//! Typed decoding of the tfevents `Event` proto and the summary messages
//! inside it.
//!
//! Only the fields a viewer needs are decoded — TF1 and TF2 scalars,
//! histograms, images, text, and summary metadata; everything else is skipped
//! at the wire level, which is how protobuf forward compatibility is meant to
//! work. Field numbers follow the frozen TensorFlow protos (`event.proto`,
//! `summary.proto`, `tensor.proto`).

use crate::wire::{DecodeError, MessageReader, WireValue, packed_f32, packed_f64};

/// One record of a tfevents file: a timestamped, stepped envelope around a
/// payload (usually a [`Summary`](EventPayload::Summary)).
#[derive(Clone, PartialEq, Debug)]
pub struct Event {
    /// Seconds since the Unix epoch, as written by the trainer.
    pub wall_time: f64,
    /// Global step. Defaults to zero when absent.
    pub step: i64,
    /// The event's payload, if it is one this decoder understands.
    pub payload: EventPayload,
}

/// The `what` oneof of an `Event`.
#[derive(Clone, PartialEq, Debug)]
#[non_exhaustive]
pub enum EventPayload {
    /// `file_version` — `"brain.Event:2"` for every file written this decade.
    FileVersion(String),
    /// `summary` — the tagged values that carry all logged data.
    Summary(Vec<SummaryValue>),
    /// A payload this decoder deliberately ignores (graph defs, session
    /// logs, run metadata) or an absent oneof.
    Other,
}

/// One `Summary.Value`: a tag and the datum logged under it.
#[derive(Clone, PartialEq, Debug)]
pub struct SummaryValue {
    /// The series tag, e.g. `train/loss`.
    pub tag: String,
    /// Plugin metadata. Writers attach it to the first point of each series
    /// only, so absence on later points is normal.
    pub metadata: Option<SummaryMetadata>,
    /// The datum itself.
    pub payload: SummaryPayload,
}

impl SummaryValue {
    /// The scalar carried by this value, under either the TF1 convention
    /// (`simple_value`) or the TF2 convention (a rank-0 float or double
    /// tensor). `None` when the value is not a scalar.
    pub fn scalar(&self) -> Option<f64> {
        match &self.payload {
            SummaryPayload::Simple(v) => Some(f64::from(*v)),
            SummaryPayload::Tensor(tensor) => tensor.scalar(),
            _ => None,
        }
    }
}

/// The value oneof of a `Summary.Value`.
#[derive(Clone, PartialEq, Debug)]
#[non_exhaustive]
pub enum SummaryPayload {
    /// `simple_value` — the TF1 scalar, an `f32`.
    Simple(f32),
    /// `tensor` — the TF2 carrier for scalars, histograms, images, and text.
    Tensor(Tensor),
    /// `image` — the TF1 encoded-image message.
    Image(Image),
    /// `histo` — the TF1 histogram message.
    Histogram(Histogram),
    /// A value kind this decoder does not understand, or an absent oneof
    /// (normal for metadata-only summaries such as hparams markers).
    Other,
}

/// `SummaryMetadata`: which plugin owns a series and how its data is classed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SummaryMetadata {
    /// The owning plugin's name, e.g. `scalars`, `histograms`, `hparams`.
    pub plugin_name: String,
    /// Plugin-specific payload (e.g. serialized hparams session info).
    pub plugin_content: Vec<u8>,
    /// The data class, set by TF2 writers; `Unknown` on TF1 files.
    pub data_class: DataClass,
}

/// `SummaryMetadata.DataClass`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum DataClass {
    /// Unset (TF1 writers, metadata-only summaries).
    Unknown,
    /// Rank-0 tensors: the scalars pipeline.
    Scalar,
    /// Small tensors: histograms, text.
    Tensor,
    /// Blob sequences: images, audio, graphs.
    BlobSequence,
    /// A value outside the known enum, preserved verbatim.
    Unrecognized(i32),
}

impl From<i32> for DataClass {
    fn from(value: i32) -> DataClass {
        match value {
            0 => DataClass::Unknown,
            1 => DataClass::Scalar,
            2 => DataClass::Tensor,
            3 => DataClass::BlobSequence,
            other => DataClass::Unrecognized(other),
        }
    }
}

/// `TensorProto`, decoded to the fields TF2 summaries actually use.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Tensor {
    /// Element type.
    pub dtype: DataType,
    /// Dimension sizes, in order. Empty for rank-0.
    pub shape: Vec<i64>,
    /// Raw little-endian element bytes (one of the two storage forms).
    pub tensor_content: Vec<u8>,
    /// Explicit `float` elements (the other storage form, for DT_FLOAT).
    pub float_val: Vec<f32>,
    /// Explicit `double` elements (for DT_DOUBLE).
    pub double_val: Vec<f64>,
    /// Explicit string/bytes elements (for DT_STRING: text, encoded images).
    pub string_val: Vec<Vec<u8>>,
}

impl Tensor {
    /// The tensor's single value as `f64`, for the TF2 scalar convention
    /// (rank-0 `DT_FLOAT` or `DT_DOUBLE`, value in the explicit list or in
    /// `tensor_content`). Lenient about a spurious `[1]` shape.
    pub fn scalar(&self) -> Option<f64> {
        if self.shape.iter().product::<i64>() != 1 {
            return None;
        }
        match self.dtype {
            DataType::Float => self
                .float_val
                .first()
                .copied()
                .or_else(|| {
                    let bytes: [u8; 4] = self.tensor_content.get(..4)?.try_into().ok()?;
                    Some(f32::from_le_bytes(bytes))
                })
                .map(f64::from),
            DataType::Double => self.double_val.first().copied().or_else(|| {
                let bytes: [u8; 8] = self.tensor_content.get(..8)?.try_into().ok()?;
                Some(f64::from_le_bytes(bytes))
            }),
            _ => None,
        }
    }

    /// All elements as `f64` for a `DT_DOUBLE` tensor (the TF2 histogram
    /// carrier: shape `[k, 3]`, rows of `(left, right, count)`).
    pub fn doubles(&self) -> Option<Vec<f64>> {
        if self.dtype != DataType::Double {
            return None;
        }
        if !self.double_val.is_empty() {
            return Some(self.double_val.clone());
        }
        packed_f64(&self.tensor_content).ok()
    }

    /// The string elements of a `DT_STRING` tensor (text summaries; image
    /// summaries are `[width, height, png, ...]`).
    pub fn strings(&self) -> Option<&[Vec<u8>]> {
        (self.dtype == DataType::String).then_some(&self.string_val[..])
    }
}

/// `tensorflow::DataType`, reduced to the cases summaries use.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[non_exhaustive]
pub enum DataType {
    /// Unset.
    #[default]
    Unknown,
    /// `DT_FLOAT` (1).
    Float,
    /// `DT_DOUBLE` (2).
    Double,
    /// `DT_INT32` (3).
    Int32,
    /// `DT_STRING` (7).
    String,
    /// `DT_INT64` (9).
    Int64,
    /// Anything else, preserved verbatim.
    Other(i32),
}

impl From<i32> for DataType {
    fn from(value: i32) -> DataType {
        match value {
            0 => DataType::Unknown,
            1 => DataType::Float,
            2 => DataType::Double,
            3 => DataType::Int32,
            7 => DataType::String,
            9 => DataType::Int64,
            other => DataType::Other(other),
        }
    }
}

/// The TF1 `Summary.Image` message.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Image {
    /// Height in pixels.
    pub height: i32,
    /// Width in pixels.
    pub width: i32,
    /// Colorspace code (1 grayscale … 6 RGBA).
    pub colorspace: i32,
    /// The encoded image bytes (PNG unless the writer chose otherwise).
    pub encoded_image: Vec<u8>,
}

/// The TF1 `HistogramProto` message.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Histogram {
    /// Minimum observed value.
    pub min: f64,
    /// Maximum observed value.
    pub max: f64,
    /// Count of observed values.
    pub num: f64,
    /// Sum of observed values.
    pub sum: f64,
    /// Sum of squares of observed values.
    pub sum_squares: f64,
    /// Right edge of each bucket.
    pub bucket_limit: Vec<f64>,
    /// Count in each bucket.
    pub bucket: Vec<f64>,
}

impl Event {
    /// Decodes one record payload as an `Event`.
    pub fn decode(buf: &[u8]) -> Result<Event, DecodeError> {
        let mut event = Event {
            wall_time: 0.0,
            step: 0,
            payload: EventPayload::Other,
        };
        let mut reader = MessageReader::new(buf);
        while let Some((field, value)) = reader.next_field()? {
            match field {
                1 => event.wall_time = value.double().unwrap_or(event.wall_time),
                2 => event.step = value.int64().unwrap_or(event.step),
                3 => {
                    if let Some(bytes) = value.bytes() {
                        event.payload =
                            EventPayload::FileVersion(String::from_utf8_lossy(bytes).into_owned());
                    }
                }
                5 => {
                    if let Some(bytes) = value.bytes() {
                        event.payload = EventPayload::Summary(decode_summary(bytes)?);
                    }
                }
                _ => {}
            }
        }
        Ok(event)
    }
}

fn decode_summary(buf: &[u8]) -> Result<Vec<SummaryValue>, DecodeError> {
    let mut values = Vec::new();
    let mut reader = MessageReader::new(buf);
    while let Some((field, value)) = reader.next_field()? {
        if field == 1
            && let Some(bytes) = value.bytes()
        {
            values.push(decode_summary_value(bytes)?);
        }
    }
    Ok(values)
}

fn decode_summary_value(buf: &[u8]) -> Result<SummaryValue, DecodeError> {
    let mut out = SummaryValue {
        tag: String::new(),
        metadata: None,
        payload: SummaryPayload::Other,
    };
    let mut reader = MessageReader::new(buf);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => {
                if let Some(bytes) = value.bytes() {
                    out.tag = String::from_utf8_lossy(bytes).into_owned();
                }
            }
            2 => {
                if let Some(v) = value.float() {
                    out.payload = SummaryPayload::Simple(v);
                }
            }
            4 => {
                if let Some(bytes) = value.bytes() {
                    out.payload = SummaryPayload::Image(decode_image(bytes)?);
                }
            }
            5 => {
                if let Some(bytes) = value.bytes() {
                    out.payload = SummaryPayload::Histogram(decode_histogram(bytes)?);
                }
            }
            8 => {
                if let Some(bytes) = value.bytes() {
                    out.payload = SummaryPayload::Tensor(decode_tensor(bytes)?);
                }
            }
            9 => {
                if let Some(bytes) = value.bytes() {
                    out.metadata = Some(decode_metadata(bytes)?);
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn decode_metadata(buf: &[u8]) -> Result<SummaryMetadata, DecodeError> {
    let mut out = SummaryMetadata {
        plugin_name: String::new(),
        plugin_content: Vec::new(),
        data_class: DataClass::Unknown,
    };
    let mut reader = MessageReader::new(buf);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => {
                if let Some(bytes) = value.bytes() {
                    let mut inner = MessageReader::new(bytes);
                    while let Some((field, value)) = inner.next_field()? {
                        match field {
                            1 => {
                                if let Some(bytes) = value.bytes() {
                                    out.plugin_name =
                                        String::from_utf8_lossy(bytes).into_owned();
                                }
                            }
                            2 => {
                                if let Some(bytes) = value.bytes() {
                                    out.plugin_content = bytes.to_vec();
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            4 => {
                if let Some(v) = value.varint() {
                    out.data_class = DataClass::from(v as i32);
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn decode_tensor(buf: &[u8]) -> Result<Tensor, DecodeError> {
    let mut out = Tensor::default();
    let mut reader = MessageReader::new(buf);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => {
                if let Some(v) = value.varint() {
                    out.dtype = DataType::from(v as i32);
                }
            }
            2 => {
                if let Some(bytes) = value.bytes() {
                    out.shape = decode_shape(bytes)?;
                }
            }
            4 => {
                if let Some(bytes) = value.bytes() {
                    out.tensor_content = bytes.to_vec();
                }
            }
            5 => match value {
                WireValue::Bytes(bytes) => out.float_val.extend(packed_f32(bytes)?),
                WireValue::Fixed32(v) => out.float_val.push(f32::from_bits(v)),
                _ => {}
            },
            6 => match value {
                WireValue::Bytes(bytes) => out.double_val.extend(packed_f64(bytes)?),
                WireValue::Fixed64(v) => out.double_val.push(f64::from_bits(v)),
                _ => {}
            },
            8 => {
                if let Some(bytes) = value.bytes() {
                    out.string_val.push(bytes.to_vec());
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn decode_shape(buf: &[u8]) -> Result<Vec<i64>, DecodeError> {
    // TensorShapeProto: repeated Dim dim = 2; Dim { int64 size = 1; }
    let mut shape = Vec::new();
    let mut reader = MessageReader::new(buf);
    while let Some((field, value)) = reader.next_field()? {
        if field == 2
            && let Some(bytes) = value.bytes()
        {
            let mut size = 0i64;
            let mut inner = MessageReader::new(bytes);
            while let Some((field, value)) = inner.next_field()? {
                if field == 1
                    && let Some(v) = value.int64()
                {
                    size = v;
                }
            }
            shape.push(size);
        }
    }
    Ok(shape)
}

fn decode_image(buf: &[u8]) -> Result<Image, DecodeError> {
    let mut out = Image::default();
    let mut reader = MessageReader::new(buf);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => out.height = value.int64().unwrap_or(0) as i32,
            2 => out.width = value.int64().unwrap_or(0) as i32,
            3 => out.colorspace = value.int64().unwrap_or(0) as i32,
            4 => {
                if let Some(bytes) = value.bytes() {
                    out.encoded_image = bytes.to_vec();
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn decode_histogram(buf: &[u8]) -> Result<Histogram, DecodeError> {
    let mut out = Histogram::default();
    let mut reader = MessageReader::new(buf);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => out.min = value.double().unwrap_or(0.0),
            2 => out.max = value.double().unwrap_or(0.0),
            3 => out.num = value.double().unwrap_or(0.0),
            4 => out.sum = value.double().unwrap_or(0.0),
            5 => out.sum_squares = value.double().unwrap_or(0.0),
            6 => match value {
                WireValue::Bytes(bytes) => out.bucket_limit.extend(packed_f64(bytes)?),
                WireValue::Fixed64(v) => out.bucket_limit.push(f64::from_bits(v)),
                _ => {}
            },
            7 => match value {
                WireValue::Bytes(bytes) => out.bucket.extend(packed_f64(bytes)?),
                WireValue::Fixed64(v) => out.bucket.push(f64::from_bits(v)),
                _ => {}
            },
            _ => {}
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::testutil::*;

    fn tf1_scalar_event(wall_time: f64, step: i64, tag: &str, value: f32) -> Vec<u8> {
        let mut summary_value = Vec::new();
        field_bytes(1, tag.as_bytes(), &mut summary_value);
        field_float(2, value, &mut summary_value);
        let mut summary = Vec::new();
        field_bytes(1, &summary_value, &mut summary);
        let mut event = Vec::new();
        field_double(1, wall_time, &mut event);
        field_varint(2, step as u64, &mut event);
        field_bytes(5, &summary, &mut event);
        event
    }

    #[test]
    fn decodes_file_version() {
        let mut buf = Vec::new();
        field_double(1, 123.5, &mut buf);
        field_bytes(3, b"brain.Event:2", &mut buf);
        let event = Event::decode(&buf).unwrap();
        assert_eq!(event.wall_time, 123.5);
        assert_eq!(event.step, 0);
        assert_eq!(
            event.payload,
            EventPayload::FileVersion("brain.Event:2".into())
        );
    }

    #[test]
    fn decodes_tf1_scalar() {
        let buf = tf1_scalar_event(1700000000.25, 42, "train/loss", 0.125);
        let event = Event::decode(&buf).unwrap();
        assert_eq!(event.wall_time, 1700000000.25);
        assert_eq!(event.step, 42);
        let EventPayload::Summary(values) = &event.payload else {
            panic!("expected summary");
        };
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].tag, "train/loss");
        assert_eq!(values[0].scalar(), Some(0.125));
    }

    #[test]
    fn decodes_negative_step() {
        let buf = tf1_scalar_event(0.0, -1, "t", 0.0);
        assert_eq!(Event::decode(&buf).unwrap().step, -1);
    }

    fn tf2_scalar_tensor(value: f32, via_content: bool) -> Vec<u8> {
        let mut tensor = Vec::new();
        field_varint(1, 1, &mut tensor); // dtype = DT_FLOAT
        if via_content {
            field_bytes(4, &value.to_le_bytes(), &mut tensor);
        } else {
            // packed float_val
            field_bytes(5, &value.to_le_bytes(), &mut tensor);
        }
        tensor
    }

    fn event_with_tensor(tag: &str, tensor: &[u8], metadata: Option<&[u8]>) -> Vec<u8> {
        let mut summary_value = Vec::new();
        field_bytes(1, tag.as_bytes(), &mut summary_value);
        field_bytes(8, tensor, &mut summary_value);
        if let Some(metadata) = metadata {
            field_bytes(9, metadata, &mut summary_value);
        }
        let mut summary = Vec::new();
        field_bytes(1, &summary_value, &mut summary);
        let mut event = Vec::new();
        field_bytes(5, &summary, &mut event);
        event
    }

    #[test]
    fn decodes_tf2_scalar_both_storage_forms() {
        for via_content in [false, true] {
            let buf = event_with_tensor("loss", &tf2_scalar_tensor(1.5, via_content), None);
            let event = Event::decode(&buf).unwrap();
            let EventPayload::Summary(values) = &event.payload else {
                panic!("expected summary");
            };
            assert_eq!(values[0].scalar(), Some(1.5), "via_content={via_content}");
        }
    }

    #[test]
    fn scalar_rejects_higher_rank_tensors() {
        // A [2]-shaped DT_FLOAT tensor is not a scalar.
        let mut tensor = Vec::new();
        field_varint(1, 1, &mut tensor);
        let mut dim = Vec::new();
        field_varint(1, 2, &mut dim);
        let mut shape = Vec::new();
        field_bytes(2, &dim, &mut shape);
        field_bytes(2, &shape, &mut tensor);
        let mut floats = Vec::new();
        floats.extend_from_slice(&1.0f32.to_le_bytes());
        floats.extend_from_slice(&2.0f32.to_le_bytes());
        field_bytes(5, &floats, &mut tensor);

        let buf = event_with_tensor("vec", &tensor, None);
        let event = Event::decode(&buf).unwrap();
        let EventPayload::Summary(values) = &event.payload else {
            panic!("expected summary");
        };
        assert_eq!(values[0].scalar(), None);
    }

    #[test]
    fn decodes_metadata_and_data_class() {
        let mut plugin_data = Vec::new();
        field_bytes(1, b"scalars", &mut plugin_data);
        field_bytes(2, b"content", &mut plugin_data);
        let mut metadata = Vec::new();
        field_bytes(1, &plugin_data, &mut metadata);
        field_varint(4, 1, &mut metadata); // DATA_CLASS_SCALAR

        let buf = event_with_tensor("loss", &tf2_scalar_tensor(0.5, false), Some(&metadata));
        let event = Event::decode(&buf).unwrap();
        let EventPayload::Summary(values) = &event.payload else {
            panic!("expected summary");
        };
        let metadata = values[0].metadata.as_ref().unwrap();
        assert_eq!(metadata.plugin_name, "scalars");
        assert_eq!(metadata.plugin_content, b"content");
        assert_eq!(metadata.data_class, DataClass::Scalar);
    }

    #[test]
    fn decodes_tf2_histogram_tensor() {
        // Shape [2, 3] DT_DOUBLE with rows (left, right, count).
        let rows = [0.0f64, 1.0, 5.0, 1.0, 2.0, 7.0];
        let mut tensor = Vec::new();
        field_varint(1, 2, &mut tensor); // DT_DOUBLE
        let mut shape = Vec::new();
        for size in [2u64, 3] {
            let mut dim = Vec::new();
            field_varint(1, size, &mut dim);
            field_bytes(2, &dim, &mut shape);
        }
        field_bytes(2, &shape, &mut tensor);
        let mut packed = Vec::new();
        for v in rows {
            packed.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        field_bytes(6, &packed, &mut tensor);

        let buf = event_with_tensor("weights", &tensor, None);
        let event = Event::decode(&buf).unwrap();
        let EventPayload::Summary(values) = &event.payload else {
            panic!("expected summary");
        };
        let SummaryPayload::Tensor(tensor) = &values[0].payload else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.doubles().unwrap(), rows);
        assert_eq!(values[0].scalar(), None);
    }

    #[test]
    fn decodes_tf1_histogram() {
        let mut histo = Vec::new();
        field_double(1, -1.0, &mut histo);
        field_double(2, 3.0, &mut histo);
        field_double(3, 10.0, &mut histo);
        let mut limits = Vec::new();
        for v in [0.0f64, 1.0, 2.0] {
            limits.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        field_bytes(6, &limits, &mut histo);
        let mut counts = Vec::new();
        for v in [3.0f64, 4.0, 3.0] {
            counts.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        field_bytes(7, &counts, &mut histo);

        let mut summary_value = Vec::new();
        field_bytes(1, b"grads", &mut summary_value);
        field_bytes(5, &histo, &mut summary_value);
        let mut summary = Vec::new();
        field_bytes(1, &summary_value, &mut summary);
        let mut event = Vec::new();
        field_bytes(5, &summary, &mut event);

        let decoded = Event::decode(&event).unwrap();
        let EventPayload::Summary(values) = &decoded.payload else {
            panic!("expected summary");
        };
        let SummaryPayload::Histogram(h) = &values[0].payload else {
            panic!("expected histogram");
        };
        assert_eq!((h.min, h.max, h.num), (-1.0, 3.0, 10.0));
        assert_eq!(h.bucket_limit, vec![0.0, 1.0, 2.0]);
        assert_eq!(h.bucket, vec![3.0, 4.0, 3.0]);
    }

    #[test]
    fn decodes_string_tensor() {
        let mut tensor = Vec::new();
        field_varint(1, 7, &mut tensor); // DT_STRING
        field_bytes(8, b"hello", &mut tensor);
        field_bytes(8, b"world", &mut tensor);
        let buf = event_with_tensor("notes/text_summary", &tensor, None);
        let event = Event::decode(&buf).unwrap();
        let EventPayload::Summary(values) = &event.payload else {
            panic!("expected summary");
        };
        let SummaryPayload::Tensor(tensor) = &values[0].payload else {
            panic!("expected tensor");
        };
        assert_eq!(
            tensor.strings().unwrap(),
            &[b"hello".to_vec(), b"world".to_vec()]
        );
    }

    #[test]
    fn decodes_tf1_image() {
        let mut image = Vec::new();
        field_varint(1, 2, &mut image);
        field_varint(2, 3, &mut image);
        field_varint(3, 3, &mut image);
        field_bytes(4, b"\x89PNGfake", &mut image);
        let mut summary_value = Vec::new();
        field_bytes(1, b"samples/image/0", &mut summary_value);
        field_bytes(4, &image, &mut summary_value);
        let mut summary = Vec::new();
        field_bytes(1, &summary_value, &mut summary);
        let mut event = Vec::new();
        field_bytes(5, &summary, &mut event);

        let decoded = Event::decode(&event).unwrap();
        let EventPayload::Summary(values) = &decoded.payload else {
            panic!("expected summary");
        };
        let SummaryPayload::Image(image) = &values[0].payload else {
            panic!("expected image");
        };
        assert_eq!((image.height, image.width, image.colorspace), (2, 3, 3));
        assert_eq!(image.encoded_image, b"\x89PNGfake");
    }

    #[test]
    fn skips_unknown_fields_everywhere() {
        let mut summary_value = Vec::new();
        field_bytes(1, b"tag", &mut summary_value);
        field_varint(77, 99, &mut summary_value); // unknown field
        field_float(2, 1.0, &mut summary_value);
        let mut summary = Vec::new();
        field_bytes(1, &summary_value, &mut summary);
        field_varint(50, 1, &mut summary); // unknown field
        let mut event = Vec::new();
        field_varint(2, 7, &mut event);
        field_bytes(5, &summary, &mut event);
        field_double(90, 3.5, &mut event); // unknown field

        let decoded = Event::decode(&event).unwrap();
        assert_eq!(decoded.step, 7);
        let EventPayload::Summary(values) = &decoded.payload else {
            panic!("expected summary");
        };
        assert_eq!(values[0].scalar(), Some(1.0));
    }

    #[test]
    fn malformed_bytes_error_rather_than_panic() {
        // A summary field whose bytes are not a valid message.
        let mut event = Vec::new();
        field_bytes(5, &[0xFF, 0xFF, 0xFF], &mut event);
        assert!(Event::decode(&event).is_err());
    }

    #[test]
    fn ignored_payloads_decode_as_other() {
        let mut event = Vec::new();
        field_double(1, 5.0, &mut event);
        field_bytes(4, b"graphdef-bytes", &mut event); // graph_def: ignored
        let decoded = Event::decode(&event).unwrap();
        assert_eq!(decoded.payload, EventPayload::Other);
    }
}
