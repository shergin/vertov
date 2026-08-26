//! Minimal protobuf wire-format reading: varints, field keys, and the four
//! wire types the tfevents protos use. No schema, no codegen — just enough to
//! walk a message and pull out the fields the decoder knows, skipping the
//! rest (forward compatibility comes free with the wire format).

/// A protobuf payload could not be decoded as a valid wire-format message.
///
/// This means either genuine corruption (distinguish by validating the record
/// checksum) or a proto feature outside this crate's minimal decoder.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecodeError {
    /// The buffer ended inside a varint, fixed-width value, or
    /// length-delimited field.
    Truncated,
    /// A varint ran past the maximum encoded length of ten bytes.
    VarintTooLong,
    /// A field key carried a wire type this decoder does not handle
    /// (including the long-deprecated group types).
    UnsupportedWireType(u8),
    /// A field key had field number zero, which is invalid.
    InvalidFieldNumber,
    /// A packed numeric field's byte length was not a multiple of the
    /// element width.
    MisalignedPackedField,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Truncated => write!(f, "message truncated mid-field"),
            DecodeError::VarintTooLong => write!(f, "varint longer than ten bytes"),
            DecodeError::UnsupportedWireType(t) => write!(f, "unsupported wire type {t}"),
            DecodeError::InvalidFieldNumber => write!(f, "field number zero is invalid"),
            DecodeError::MisalignedPackedField => {
                write!(f, "packed field length is not a multiple of the element size")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// One field's value, tagged by wire type.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum WireValue<'a> {
    /// Wire type 0: int32/int64/uint/bool/enum.
    Varint(u64),
    /// Wire type 1: fixed64/double.
    Fixed64(u64),
    /// Wire type 2: strings, bytes, sub-messages, packed repeated numerics.
    Bytes(&'a [u8]),
    /// Wire type 5: fixed32/float.
    Fixed32(u32),
}

impl<'a> WireValue<'a> {
    /// The value as a varint, if that is its wire type.
    pub(crate) fn varint(self) -> Option<u64> {
        match self {
            WireValue::Varint(v) => Some(v),
            _ => None,
        }
    }

    /// The value as length-delimited bytes, if that is its wire type.
    pub(crate) fn bytes(self) -> Option<&'a [u8]> {
        match self {
            WireValue::Bytes(b) => Some(b),
            _ => None,
        }
    }

    /// The value as an `f64`, if it is a fixed64.
    pub(crate) fn double(self) -> Option<f64> {
        match self {
            WireValue::Fixed64(v) => Some(f64::from_bits(v)),
            _ => None,
        }
    }

    /// The value as an `f32`, if it is a fixed32.
    pub(crate) fn float(self) -> Option<f32> {
        match self {
            WireValue::Fixed32(v) => Some(f32::from_bits(v)),
            _ => None,
        }
    }

    /// The value as an `i64` (standard two's-complement varint encoding).
    pub(crate) fn int64(self) -> Option<i64> {
        self.varint().map(|v| v as i64)
    }
}

/// A cursor over one serialized message.
pub(crate) struct MessageReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> MessageReader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> MessageReader<'a> {
        MessageReader { buf, pos: 0 }
    }

    /// Reads the next `(field number, value)` pair, or `None` at the end of
    /// the message. Unknown fields are the caller's to ignore — every wire
    /// type is fully consumed here.
    pub(crate) fn next_field(&mut self) -> Result<Option<(u32, WireValue<'a>)>, DecodeError> {
        if self.pos == self.buf.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = u32::try_from(key >> 3).map_err(|_| DecodeError::InvalidFieldNumber)?;
        if field == 0 {
            return Err(DecodeError::InvalidFieldNumber);
        }
        let value = match (key & 0x7) as u8 {
            0 => WireValue::Varint(self.varint()?),
            1 => WireValue::Fixed64(u64::from_le_bytes(
                self.take(8)?.try_into().expect("8 bytes"),
            )),
            2 => {
                let len = self.varint()?;
                let len = usize::try_from(len).map_err(|_| DecodeError::Truncated)?;
                WireValue::Bytes(self.take(len)?)
            }
            5 => WireValue::Fixed32(u32::from_le_bytes(
                self.take(4)?.try_into().expect("4 bytes"),
            )),
            other => return Err(DecodeError::UnsupportedWireType(other)),
        };
        Ok(Some((field, value)))
    }

    fn varint(&mut self) -> Result<u64, DecodeError> {
        let mut result = 0u64;
        for shift in (0..64).step_by(7) {
            let &byte = self.buf.get(self.pos).ok_or(DecodeError::Truncated)?;
            self.pos += 1;
            // The tenth byte (shift 63) may only contribute the final bit;
            // higher bits would overflow u64.
            if shift == 63 && byte & 0x7E != 0 {
                return Err(DecodeError::VarintTooLong);
            }
            result |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
        }
        Err(DecodeError::VarintTooLong)
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(len).ok_or(DecodeError::Truncated)?;
        if end > self.buf.len() {
            return Err(DecodeError::Truncated);
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }
}

/// Decodes a packed repeated `double` field (also accepting the single
/// unpacked element handed in by the caller via [`WireValue::Fixed64`]).
pub(crate) fn packed_f64(bytes: &[u8]) -> Result<Vec<f64>, DecodeError> {
    if bytes.len() % 8 != 0 {
        return Err(DecodeError::MisalignedPackedField);
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("8 bytes")))
        .collect())
}

/// Decodes a packed repeated `float` field.
pub(crate) fn packed_f32(bytes: &[u8]) -> Result<Vec<f32>, DecodeError> {
    if bytes.len() % 4 != 0 {
        return Err(DecodeError::MisalignedPackedField);
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4 bytes")))
        .collect())
}

#[cfg(test)]
pub(crate) mod testutil {
    //! Hand-encoding helpers so tests build wire-format messages without a
    //! protobuf library.

    pub(crate) fn varint(mut value: u64, out: &mut Vec<u8>) {
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

    pub(crate) fn field_varint(field: u32, value: u64, out: &mut Vec<u8>) {
        varint(u64::from(field) << 3, out);
        varint(value, out);
    }

    pub(crate) fn field_double(field: u32, value: f64, out: &mut Vec<u8>) {
        varint(u64::from(field) << 3 | 1, out);
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    pub(crate) fn field_float(field: u32, value: f32, out: &mut Vec<u8>) {
        varint(u64::from(field) << 3 | 5, out);
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    pub(crate) fn field_bytes(field: u32, value: &[u8], out: &mut Vec<u8>) {
        varint(u64::from(field) << 3 | 2, out);
        varint(value.len() as u64, out);
        out.extend_from_slice(value);
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::*;
    use super::*;

    #[test]
    fn walks_all_wire_types() {
        let mut buf = Vec::new();
        field_varint(1, 150, &mut buf);
        field_double(2, -0.5, &mut buf);
        field_bytes(3, b"payload", &mut buf);
        field_float(4, 2.25, &mut buf);

        let mut reader = MessageReader::new(&buf);
        assert_eq!(
            reader.next_field().unwrap(),
            Some((1, WireValue::Varint(150)))
        );
        let (field, value) = reader.next_field().unwrap().unwrap();
        assert_eq!((field, value.double()), (2, Some(-0.5)));
        let (field, value) = reader.next_field().unwrap().unwrap();
        assert_eq!((field, value.bytes()), (3, Some(&b"payload"[..])));
        let (field, value) = reader.next_field().unwrap().unwrap();
        assert_eq!((field, value.float()), (4, Some(2.25)));
        assert_eq!(reader.next_field().unwrap(), None);
    }

    #[test]
    fn negative_int64_roundtrips() {
        let mut buf = Vec::new();
        field_varint(2, (-7i64) as u64, &mut buf);
        let mut reader = MessageReader::new(&buf);
        let (_, value) = reader.next_field().unwrap().unwrap();
        assert_eq!(value.int64(), Some(-7));
    }

    #[test]
    fn truncated_inputs_error_cleanly() {
        let mut buf = Vec::new();
        field_bytes(1, b"hello", &mut buf);
        for end in 0..buf.len() {
            let mut reader = MessageReader::new(&buf[..end]);
            match reader.next_field() {
                Err(DecodeError::Truncated) | Ok(None) => {}
                other => panic!("prefix {end}: unexpected {other:?}"),
            }
        }
    }

    #[test]
    fn rejects_groups_and_bad_field_numbers() {
        // Wire type 3 (start group).
        let mut reader = MessageReader::new(&[0x0B]);
        assert_eq!(
            reader.next_field(),
            Err(DecodeError::UnsupportedWireType(3))
        );
        // Field number 0, wire type 0.
        let mut reader = MessageReader::new(&[0x00, 0x01]);
        assert_eq!(reader.next_field(), Err(DecodeError::InvalidFieldNumber));
    }

    #[test]
    fn varint_limits() {
        // u64::MAX is ten bytes and valid.
        let mut buf = Vec::new();
        field_varint(1, u64::MAX, &mut buf);
        let mut reader = MessageReader::new(&buf);
        assert_eq!(
            reader.next_field().unwrap(),
            Some((1, WireValue::Varint(u64::MAX)))
        );
        // Eleven continuation bytes are rejected.
        let mut long = vec![0x08];
        long.extend_from_slice(&[0x80; 10]);
        long.push(0x00);
        let mut reader = MessageReader::new(&long);
        assert_eq!(reader.next_field(), Err(DecodeError::VarintTooLong));
        // A tenth byte with overflow bits is rejected.
        let mut overflow = vec![0x08];
        overflow.extend_from_slice(&[0xFF; 9]);
        overflow.push(0x02);
        let mut reader = MessageReader::new(&overflow);
        assert_eq!(reader.next_field(), Err(DecodeError::VarintTooLong));
    }

    #[test]
    fn packed_decoding() {
        let mut bytes = Vec::new();
        for v in [1.0f64, -2.5, f64::NAN] {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        let values = packed_f64(&bytes).unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], -2.5);
        assert!(values[2].is_nan());
        assert_eq!(packed_f64(&bytes[..7]), Err(DecodeError::MisalignedPackedField));
        assert_eq!(packed_f32(&1.5f32.to_bits().to_le_bytes()), Ok(vec![1.5]));
    }
}
