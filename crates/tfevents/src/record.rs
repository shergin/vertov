//! TFRecord framing: length-prefixed records with masked CRC-32C checksums.
//!
//! Each record on disk is `u64 LE length` + `u32 masked CRC of the length
//! bytes` + `payload` + `u32 masked CRC of the payload`.
//!
//! The reader is a resumable state machine over any [`Read`]: hitting
//! end-of-file mid-record returns [`ReadRecordError::Truncated`] and keeps the
//! partial state, so the next call continues where the last one stopped once
//! the writer has appended more bytes. A bad length CRC means the framing
//! itself cannot be trusted — the file is dead past that point, its valid
//! prefix retained. Payload CRCs are deliberately *not* verified here (the hot
//! path skips them); [`Record::checksum`] validates on demand, typically only
//! after a payload fails to parse, to distinguish corruption from a decoder
//! gap.

use std::io::{ErrorKind, Read};

use crate::crc::MaskedCrc;

/// A complete record read from TFRecord framing.
///
/// The payload CRC has been read but not verified; call [`Record::checksum`]
/// to validate it (off the hot path — typically only when the payload fails
/// to parse downstream).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Record {
    /// The record payload.
    pub data: Vec<u8>,
    /// The masked CRC-32C of the payload as stored on disk.
    pub data_crc: MaskedCrc,
}

impl Record {
    /// Verifies the payload against its stored CRC.
    pub fn checksum(&self) -> Result<(), ChecksumError> {
        let got = MaskedCrc::compute(&self.data);
        if got == self.data_crc {
            Ok(())
        } else {
            Err(ChecksumError { want: self.data_crc, got })
        }
    }
}

/// A payload failed CRC validation: the bytes on disk are corrupt.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ChecksumError {
    /// The checksum stored on disk.
    pub want: MaskedCrc,
    /// The checksum of the bytes actually read.
    pub got: MaskedCrc,
}

impl std::fmt::Display for ChecksumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "record checksum mismatch: stored {:#010x}, computed {:#010x}",
            self.want.0, self.got.0
        )
    }
}

impl std::error::Error for ChecksumError {}

/// Failure modes of [`RecordReader::read_record`].
#[derive(Debug)]
pub enum ReadRecordError {
    /// The stream ended mid-record. This is a normal state, not damage: a
    /// live writer may still be appending, so keep the reader and retry once
    /// the file has grown. Partial progress is retained.
    Truncated,
    /// The length header failed its CRC: the framing is untrustworthy from
    /// `offset` on, and no further records can be recovered from this stream.
    /// Everything read before `offset` remains valid.
    BadLengthCrc {
        /// Byte offset of the start of the unrecoverable record.
        offset: u64,
        /// The checksum stored on disk.
        want: MaskedCrc,
        /// The checksum computed from the length bytes actually read.
        got: MaskedCrc,
    },
    /// An I/O error other than end-of-file.
    Io(std::io::Error),
}

impl std::fmt::Display for ReadRecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadRecordError::Truncated => write!(f, "record truncated (stream ended mid-record)"),
            ReadRecordError::BadLengthCrc { offset, want, got } => write!(
                f,
                "bad length CRC at offset {offset}: stored {:#010x}, computed {:#010x}",
                want.0, got.0
            ),
            ReadRecordError::Io(err) => write!(f, "I/O error reading record: {err}"),
        }
    }
}

impl std::error::Error for ReadRecordError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReadRecordError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ReadRecordError {
    fn from(err: std::io::Error) -> Self {
        ReadRecordError::Io(err)
    }
}

const LENGTH_HEADER: usize = 8 + 4; // u64 length + masked CRC of the length bytes
const DATA_CRC: usize = 4;

// Cap on upfront buffer reservation. A crafted header can claim any length
// with a valid CRC; the buffer grows with bytes actually read, so a huge
// claimed length costs no more memory than the stream actually delivers.
const RESERVE_CAP: usize = 1 << 20;

enum State {
    /// Accumulating the 12-byte length header.
    Header { buf: Vec<u8> },
    /// Accumulating `length` payload bytes plus the trailing payload CRC.
    Payload { length: usize, buf: Vec<u8> },
    /// A bad length CRC was seen: the framing is dead, every subsequent call
    /// reports the same error without touching the stream.
    Dead {
        offset: u64,
        want: MaskedCrc,
        got: MaskedCrc,
    },
}

/// A resumable reader for TFRecord framing over any [`Read`].
///
/// The reader owns only parse state, not the stream: pass the stream to each
/// [`read_record`](RecordReader::read_record) call. This keeps it usable both
/// for one-shot reads and for live tailing, where the same `File` is polled
/// as it grows.
pub struct RecordReader {
    state: State,
    /// Total bytes consumed from the stream, including any partial record.
    consumed: u64,
    /// Bytes consumed through the end of the last complete record — the
    /// offset to resume from if the reader is dropped and later recreated.
    committed: u64,
}

impl RecordReader {
    /// Creates a reader expecting a record boundary at the start of the
    /// stream.
    pub fn new() -> RecordReader {
        RecordReader {
            state: State::Header { buf: Vec::new() },
            consumed: 0,
            committed: 0,
        }
    }

    /// Creates a reader that accounts for `offset` bytes already consumed
    /// before the stream's current position (for resuming a previously
    /// committed offset: seek the file, then hand it to this reader).
    pub fn resume_at(offset: u64) -> RecordReader {
        RecordReader {
            state: State::Header { buf: Vec::new() },
            consumed: offset,
            committed: offset,
        }
    }

    /// Byte offset just past the last complete record. Seeking a fresh stream
    /// here and reading with [`RecordReader::resume_at`] continues exactly
    /// where this reader left off.
    pub fn committed_offset(&self) -> u64 {
        self.committed
    }

    /// Reads the next record, resuming any partial progress.
    ///
    /// On [`ReadRecordError::Truncated`] the partial state is kept; call
    /// again with the same (grown) stream to continue. On
    /// [`ReadRecordError::BadLengthCrc`] the reader is permanently dead and
    /// repeats the same error.
    pub fn read_record<R: Read>(&mut self, reader: &mut R) -> Result<Record, ReadRecordError> {
        loop {
            match &mut self.state {
                State::Dead { offset, want, got } => {
                    return Err(ReadRecordError::BadLengthCrc {
                        offset: *offset,
                        want: *want,
                        got: *got,
                    });
                }
                State::Header { buf } => {
                    fill(reader, buf, LENGTH_HEADER, &mut self.consumed)?;
                    let length_bytes: [u8; 8] = buf[..8].try_into().expect("header is 12 bytes");
                    let stored =
                        MaskedCrc(u32::from_le_bytes(buf[8..12].try_into().expect("4 bytes")));
                    let computed = MaskedCrc::compute(&length_bytes);
                    if computed != stored {
                        let offset = self.consumed - LENGTH_HEADER as u64;
                        self.state = State::Dead {
                            offset,
                            want: stored,
                            got: computed,
                        };
                        continue;
                    }
                    let length = usize::try_from(u64::from_le_bytes(length_bytes))
                        .unwrap_or(usize::MAX);
                    let buf =
                        Vec::with_capacity(length.saturating_add(DATA_CRC).min(RESERVE_CAP));
                    self.state = State::Payload { length, buf };
                }
                State::Payload { length, buf } => {
                    let want = length.saturating_add(DATA_CRC);
                    fill(reader, buf, want, &mut self.consumed)?;
                    let data_crc = MaskedCrc(u32::from_le_bytes(
                        buf[*length..].try_into().expect("trailing CRC is 4 bytes"),
                    ));
                    let mut data = std::mem::take(buf);
                    data.truncate(*length);
                    self.committed = self.consumed;
                    self.state = State::Header { buf: Vec::new() };
                    return Ok(Record { data, data_crc });
                }
            }
        }
    }
}

impl Default for RecordReader {
    fn default() -> Self {
        RecordReader::new()
    }
}

/// Extends `buf` from `reader` until it holds `want` bytes, tracking consumed
/// bytes. Returns `Truncated` on end-of-file short of the goal.
fn fill<R: Read>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    want: usize,
    consumed: &mut u64,
) -> Result<(), ReadRecordError> {
    let mut chunk = [0u8; 8192];
    while buf.len() < want {
        let goal = (want - buf.len()).min(chunk.len());
        match reader.read(&mut chunk[..goal]) {
            Ok(0) => return Err(ReadRecordError::Truncated),
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                *consumed += n as u64;
            }
            Err(err) if err.kind() == ErrorKind::Interrupted => {}
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn write_record(out: &mut Vec<u8>, payload: &[u8]) {
    let length = payload.len() as u64;
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(&MaskedCrc::compute(&length.to_le_bytes()).0.to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&MaskedCrc::compute(payload).0.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn file_with(payloads: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for payload in payloads {
            write_record(&mut out, payload);
        }
        out
    }

    #[test]
    fn reads_records_in_order() {
        let file = file_with(&[b"first", b"", b"third record"]);
        let mut cursor = Cursor::new(&file);
        let mut reader = RecordReader::new();
        for expected in [&b"first"[..], b"", b"third record"] {
            let record = reader.read_record(&mut cursor).unwrap();
            assert_eq!(record.data, expected);
            record.checksum().unwrap();
        }
        assert!(matches!(
            reader.read_record(&mut cursor),
            Err(ReadRecordError::Truncated)
        ));
        assert_eq!(reader.committed_offset(), file.len() as u64);
    }

    #[test]
    fn resumes_across_truncation_at_every_byte() {
        // Feed the file one byte at a time; the reader must report Truncated
        // at every incomplete prefix and produce both records exactly once.
        let file = file_with(&[b"hello", b"world!"]);
        let mut reader = RecordReader::new();
        let mut records = Vec::new();
        for (position, byte) in file.iter().enumerate() {
            let mut cursor = Cursor::new(std::slice::from_ref(byte));
            loop {
                match reader.read_record(&mut cursor) {
                    Ok(record) => records.push(record.data),
                    Err(ReadRecordError::Truncated) => break,
                    Err(err) => panic!("unexpected error at byte {position}: {err}"),
                }
            }
        }
        assert_eq!(records, vec![b"hello".to_vec(), b"world!".to_vec()]);
    }

    #[test]
    fn bad_length_crc_kills_the_stream() {
        let mut file = file_with(&[b"good"]);
        let offset = file.len() as u64;
        let mut second = file_with(&[b"bad"]);
        second[8] ^= 0xFF; // corrupt the length CRC
        file.extend_from_slice(&second);
        file.extend_from_slice(&file_with(&[b"unreachable"]));

        let mut cursor = Cursor::new(&file);
        let mut reader = RecordReader::new();
        assert_eq!(reader.read_record(&mut cursor).unwrap().data, b"good");
        for _ in 0..2 {
            match reader.read_record(&mut cursor) {
                Err(ReadRecordError::BadLengthCrc { offset: at, .. }) => assert_eq!(at, offset),
                other => panic!("expected BadLengthCrc, got {other:?}"),
            }
        }
        assert_eq!(reader.committed_offset(), offset);
    }

    #[test]
    fn corrupt_payload_is_detected_lazily() {
        let mut file = file_with(&[b"payload"]);
        file[12] ^= 0x01; // flip a payload bit; framing stays valid
        let mut cursor = Cursor::new(&file);
        let mut reader = RecordReader::new();
        let record = reader.read_record(&mut cursor).unwrap();
        assert!(record.checksum().is_err());
        // The framing is still sound: a following record reads fine.
    }

    #[test]
    fn corrupt_record_does_not_block_successors() {
        let mut file = file_with(&[b"broken"]);
        file[12] ^= 0x01;
        write_record(&mut file, b"fine");
        let mut cursor = Cursor::new(&file);
        let mut reader = RecordReader::new();
        assert!(reader.read_record(&mut cursor).unwrap().checksum().is_err());
        let record = reader.read_record(&mut cursor).unwrap();
        assert_eq!(record.data, b"fine");
        record.checksum().unwrap();
    }

    #[test]
    fn resume_at_continues_from_committed_offset() {
        let file = file_with(&[b"one", b"two"]);
        let mut reader = RecordReader::new();
        let mut cursor = Cursor::new(&file);
        assert_eq!(reader.read_record(&mut cursor).unwrap().data, b"one");
        let offset = reader.committed_offset();

        let mut resumed = RecordReader::resume_at(offset);
        let mut cursor = Cursor::new(&file[offset as usize..]);
        assert_eq!(resumed.read_record(&mut cursor).unwrap().data, b"two");
        assert_eq!(resumed.committed_offset(), file.len() as u64);
    }

    #[test]
    fn huge_claimed_length_does_not_allocate_upfront() {
        // A valid header claiming u64::MAX bytes must yield Truncated (and
        // bounded memory), not an allocation of the claimed size.
        let length = u64::MAX.to_le_bytes();
        let mut file = Vec::new();
        file.extend_from_slice(&length);
        file.extend_from_slice(&MaskedCrc::compute(&length).0.to_le_bytes());
        file.extend_from_slice(b"only a few actual bytes");
        let mut cursor = Cursor::new(&file);
        let mut reader = RecordReader::new();
        assert!(matches!(
            reader.read_record(&mut cursor),
            Err(ReadRecordError::Truncated)
        ));
    }
}
