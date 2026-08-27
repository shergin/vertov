//! A minimal tfevents *writer* — for tests, fixtures, and benchmarks.
//!
//! vertov itself never writes into a run directory (it is a read-only
//! observer); this module exists so that code exercising the readers —
//! including other crates' tests — can produce byte-exact synthetic files
//! without a protobuf library. It covers only what those tests need: TFRecord
//! framing and TF1-style scalar events.

use crate::crc::MaskedCrc;

/// Appends one TFRecord (length, masked length CRC, payload, masked payload
/// CRC) to `out`.
pub fn write_record(out: &mut Vec<u8>, payload: &[u8]) {
    let length = payload.len() as u64;
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(&MaskedCrc::compute(&length.to_le_bytes()).0.to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&MaskedCrc::compute(payload).0.to_le_bytes());
}

/// Encodes a `file_version` event (`brain.Event:2`), the first record of
/// every real events file.
pub fn file_version_event() -> Vec<u8> {
    let mut event = Vec::new();
    field_bytes(3, b"brain.Event:2", &mut event);
    event
}

/// Encodes one TF1-style scalar event: a `Summary` with a single
/// `simple_value` under `tag`.
pub fn scalar_event(wall_time: f64, step: i64, tag: &str, value: f32) -> Vec<u8> {
    let mut summary_value = Vec::new();
    field_bytes(1, tag.as_bytes(), &mut summary_value);
    varint(2 << 3 | 5, &mut summary_value);
    summary_value.extend_from_slice(&value.to_bits().to_le_bytes());
    let mut summary = Vec::new();
    field_bytes(1, &summary_value, &mut summary);

    let mut event = Vec::new();
    varint(1 << 3 | 1, &mut event);
    event.extend_from_slice(&wall_time.to_bits().to_le_bytes());
    varint(2 << 3, &mut event);
    varint(step as u64, &mut event);
    field_bytes(5, &summary, &mut event);
    event
}

/// Encodes one TF1-style histogram event: a `HistogramProto` with the given
/// `(left, right, count)` buckets under `tag`. Interior edges become
/// `bucket_limit`s; the outer edges become `min`/`max`.
pub fn histogram_event(
    wall_time: f64,
    step: i64,
    tag: &str,
    buckets: &[(f64, f64, f64)],
) -> Vec<u8> {
    let mut histogram = Vec::new();
    let min = buckets.first().map_or(0.0, |bucket| bucket.0);
    let max = buckets.last().map_or(0.0, |bucket| bucket.1);
    let total: f64 = buckets.iter().map(|bucket| bucket.2).sum();
    for (field, value) in [(1, min), (2, max), (3, total)] {
        varint(field << 3 | 1, &mut histogram);
        histogram.extend_from_slice(&f64::to_bits(value).to_le_bytes());
    }
    let mut limits = Vec::new();
    let mut counts = Vec::new();
    for &(_, right, count) in buckets {
        limits.extend_from_slice(&right.to_bits().to_le_bytes());
        counts.extend_from_slice(&count.to_bits().to_le_bytes());
    }
    field_bytes(6, &limits, &mut histogram);
    field_bytes(7, &counts, &mut histogram);

    let mut summary_value = Vec::new();
    field_bytes(1, tag.as_bytes(), &mut summary_value);
    field_bytes(5, &histogram, &mut summary_value);
    let mut summary = Vec::new();
    field_bytes(1, &summary_value, &mut summary);

    let mut event = Vec::new();
    varint(1 << 3 | 1, &mut event);
    event.extend_from_slice(&wall_time.to_bits().to_le_bytes());
    varint(2 << 3, &mut event);
    varint(step as u64, &mut event);
    field_bytes(5, &summary, &mut event);
    event
}

/// A complete little events file: `file_version` followed by the given
/// pre-encoded event payloads, each framed as a TFRecord.
pub fn events_file(events: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    write_record(&mut out, &file_version_event());
    for event in events {
        write_record(&mut out, event);
    }
    out
}

fn varint(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn field_bytes(field: u32, value: &[u8], out: &mut Vec<u8>) {
    varint(u64::from(field) << 3 | 2, out);
    varint(value.len() as u64, out);
    out.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventPayload;
    use crate::reader::{EventFileReader, ReadEventError};

    #[test]
    fn written_files_read_back() {
        let file = events_file(&[
            scalar_event(1000.0, 0, "loss", 4.0),
            scalar_event(1001.0, 1, "loss", 2.0),
        ]);
        let mut reader = EventFileReader::new(std::io::Cursor::new(&file));
        assert!(matches!(
            reader.next_event().unwrap().payload,
            EventPayload::FileVersion(_)
        ));
        for (step, expected) in [(0, 4.0), (1, 2.0)] {
            let event = reader.next_event().unwrap();
            assert_eq!(event.step, step);
            assert_eq!(event.wall_time, 1000.0 + step as f64);
            let EventPayload::Summary(values) = &event.payload else {
                panic!("expected summary");
            };
            assert_eq!(values[0].tag, "loss");
            assert_eq!(values[0].scalar(), Some(expected));
        }
        assert!(matches!(reader.next_event(), Err(ReadEventError::Truncated)));
    }
}
