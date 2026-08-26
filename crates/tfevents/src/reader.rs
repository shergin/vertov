//! The top layer: a resumable reader that turns a growing tfevents stream
//! into decoded [`Event`]s, with an honest taxonomy of everything that can go
//! wrong in a file a trainer is still writing (or died while writing).

use std::io::{Read, Seek, SeekFrom};

use crate::crc::MaskedCrc;
use crate::event::Event;
use crate::record::{ChecksumError, ReadRecordError, RecordReader};
use crate::wire::DecodeError;

/// Failure modes of [`EventFileReader::next_event`].
///
/// Only [`Truncated`](ReadEventError::Truncated) is a stopping state, and
/// even it is temporary while a writer lives. `Corrupt` and `Malformed` spoil
/// one record — report them and keep reading; `BadLengthCrc` ends the file,
/// and `Io` is whatever the filesystem made of your day.
#[derive(Debug)]
pub enum ReadEventError {
    /// The stream ended mid-record: either the true end of a live file or a
    /// torn write from a dead one. Keep the reader; retry when the file
    /// grows.
    Truncated,
    /// The framing is untrustworthy from `offset` on; no further records can
    /// be recovered. Data before `offset` remains valid.
    BadLengthCrc {
        /// Byte offset of the start of the unrecoverable record.
        offset: u64,
        /// The checksum stored on disk.
        want: MaskedCrc,
        /// The checksum computed from the length bytes actually read.
        got: MaskedCrc,
    },
    /// The record at `offset` failed its payload checksum: bytes were lost
    /// or mangled on disk. The framing is intact — subsequent records still
    /// read. This is visible data loss, not a silent gap.
    Corrupt {
        /// Byte offset of the start of the corrupt record.
        offset: u64,
        /// The stored-vs-computed checksum pair.
        error: ChecksumError,
    },
    /// The record at `offset` passed its checksum but is not a decodable
    /// `Event` — most likely a proto feature outside this decoder. The
    /// stream continues.
    Malformed {
        /// Byte offset of the start of the undecodable record.
        offset: u64,
        /// What the wire decoder objected to.
        error: DecodeError,
    },
    /// An I/O error other than end-of-file.
    Io(std::io::Error),
}

impl std::fmt::Display for ReadEventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadEventError::Truncated => write!(f, "event truncated (stream ended mid-record)"),
            ReadEventError::BadLengthCrc { offset, .. } => {
                write!(f, "framing dead at offset {offset}: bad length CRC")
            }
            ReadEventError::Corrupt { offset, error } => {
                write!(f, "corrupt record at offset {offset}: {error}")
            }
            ReadEventError::Malformed { offset, error } => {
                write!(f, "undecodable record at offset {offset}: {error}")
            }
            ReadEventError::Io(err) => write!(f, "I/O error reading event: {err}"),
        }
    }
}

impl std::error::Error for ReadEventError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReadEventError::Corrupt { error, .. } => Some(error),
            ReadEventError::Malformed { error, .. } => Some(error),
            ReadEventError::Io(err) => Some(err),
            _ => None,
        }
    }
}

/// Reads decoded [`Event`]s from a tfevents stream, resumably.
///
/// The reader owns the stream and its parse state. For live tailing, keep
/// calling [`next_event`](EventFileReader::next_event) after `Truncated` —
/// the open handle and the internal offset *are* the resume state. To resume
/// a previously closed file cheaply, persist
/// [`committed_offset`](EventFileReader::committed_offset) and reopen with
/// [`EventFileReader::resume`].
pub struct EventFileReader<R> {
    stream: R,
    records: RecordReader,
}

impl<R: Read> EventFileReader<R> {
    /// Wraps a stream positioned at the start of a tfevents file.
    pub fn new(stream: R) -> EventFileReader<R> {
        EventFileReader {
            stream,
            records: RecordReader::new(),
        }
    }

    /// Byte offset just past the last complete record — the value to persist
    /// for [`EventFileReader::resume`].
    pub fn committed_offset(&self) -> u64 {
        self.records.committed_offset()
    }

    /// Reads the next event.
    ///
    /// Checksums are not verified on the happy path (framing already
    /// guarantees length integrity); a payload checksum is computed only when
    /// decoding fails, to tell [`Corrupt`](ReadEventError::Corrupt) apart
    /// from [`Malformed`](ReadEventError::Malformed).
    pub fn next_event(&mut self) -> Result<Event, ReadEventError> {
        let offset = self.records.committed_offset();
        let record = match self.records.read_record(&mut self.stream) {
            Ok(record) => record,
            Err(ReadRecordError::Truncated) => return Err(ReadEventError::Truncated),
            Err(ReadRecordError::BadLengthCrc { offset, want, got }) => {
                return Err(ReadEventError::BadLengthCrc { offset, want, got });
            }
            Err(ReadRecordError::Io(err)) => return Err(ReadEventError::Io(err)),
        };
        match Event::decode(&record.data) {
            Ok(event) => Ok(event),
            Err(decode_error) => match record.checksum() {
                Err(error) => Err(ReadEventError::Corrupt { offset, error }),
                Ok(()) => Err(ReadEventError::Malformed {
                    offset,
                    error: decode_error,
                }),
            },
        }
    }

    /// Consumes the reader, returning the underlying stream.
    pub fn into_inner(self) -> R {
        self.stream
    }
}

impl<R: Read + Seek> EventFileReader<R> {
    /// Resumes reading at `offset`, which must be a record boundary
    /// previously obtained from [`committed_offset`](Self::committed_offset)
    /// (on this file, unchanged since).
    pub fn resume(mut stream: R, offset: u64) -> std::io::Result<EventFileReader<R>> {
        stream.seek(SeekFrom::Start(offset))?;
        Ok(EventFileReader {
            stream,
            records: RecordReader::resume_at(offset),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventPayload;
    use crate::record::write_record;
    use crate::wire::testutil::*;
    use std::io::Cursor;

    fn scalar_event_payload(step: i64, tag: &str, value: f32) -> Vec<u8> {
        let mut summary_value = Vec::new();
        field_bytes(1, tag.as_bytes(), &mut summary_value);
        field_float(2, value, &mut summary_value);
        let mut summary = Vec::new();
        field_bytes(1, &summary_value, &mut summary);
        let mut event = Vec::new();
        field_double(1, 1000.0 + step as f64, &mut event);
        field_varint(2, step as u64, &mut event);
        field_bytes(5, &summary, &mut event);
        event
    }

    fn scalar_of(event: &Event) -> (i64, f64) {
        let EventPayload::Summary(values) = &event.payload else {
            panic!("expected summary");
        };
        (event.step, values[0].scalar().unwrap())
    }

    #[test]
    fn reads_events_end_to_end() {
        let mut file = Vec::new();
        for step in 0..3 {
            write_record(&mut file, &scalar_event_payload(step, "loss", step as f32));
        }
        let mut reader = EventFileReader::new(Cursor::new(&file));
        for step in 0..3 {
            let event = reader.next_event().unwrap();
            assert_eq!(scalar_of(&event), (step, f64::from(step as f32)));
        }
        assert!(matches!(reader.next_event(), Err(ReadEventError::Truncated)));
        assert_eq!(reader.committed_offset(), file.len() as u64);
    }

    #[test]
    fn tailing_a_growing_file() {
        // Simulate a live writer: the reader sees a torn tail, then the rest
        // of the bytes arrive and the event completes.
        let mut full = Vec::new();
        write_record(&mut full, &scalar_event_payload(1, "loss", 0.5));
        write_record(&mut full, &scalar_event_payload(2, "loss", 0.25));
        let cut = full.len() - 5;

        struct GrowingFile {
            data: Vec<u8>,
            pos: usize,
        }
        impl Read for GrowingFile {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let n = (self.data.len() - self.pos).min(buf.len());
                buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
                self.pos += n;
                Ok(n)
            }
        }

        let mut reader = EventFileReader::new(GrowingFile {
            data: full[..cut].to_vec(),
            pos: 0,
        });
        assert_eq!(scalar_of(&reader.next_event().unwrap()), (1, 0.5));
        assert!(matches!(reader.next_event(), Err(ReadEventError::Truncated)));
        assert!(matches!(reader.next_event(), Err(ReadEventError::Truncated)));

        reader.stream.data.extend_from_slice(&full[cut..]);
        assert_eq!(scalar_of(&reader.next_event().unwrap()), (2, 0.25));
    }

    #[test]
    fn corrupt_record_is_reported_and_skipped() {
        let mut file = Vec::new();
        write_record(&mut file, &scalar_event_payload(1, "loss", 1.0));
        let corrupt_at = file.len() as u64;
        write_record(&mut file, &scalar_event_payload(2, "loss", 2.0));
        file[corrupt_at as usize + 12] ^= 0x80; // flip a payload bit
        write_record(&mut file, &scalar_event_payload(3, "loss", 3.0));

        let mut reader = EventFileReader::new(Cursor::new(&file));
        assert_eq!(scalar_of(&reader.next_event().unwrap()).0, 1);
        match reader.next_event() {
            Err(ReadEventError::Corrupt { offset, .. }) => assert_eq!(offset, corrupt_at),
            other => panic!("expected Corrupt, got {other:?}"),
        }
        assert_eq!(scalar_of(&reader.next_event().unwrap()).0, 3);
    }

    #[test]
    fn malformed_record_with_valid_checksum() {
        // 0xFF opens field 31 wire type 7 — invalid, but the checksum is
        // genuine, so this is Malformed, not Corrupt.
        let mut file = Vec::new();
        write_record(&mut file, &[0xFF, 0xFF]);
        write_record(&mut file, &scalar_event_payload(1, "loss", 1.0));
        let mut reader = EventFileReader::new(Cursor::new(&file));
        assert!(matches!(
            reader.next_event(),
            Err(ReadEventError::Malformed { offset: 0, .. })
        ));
        assert_eq!(scalar_of(&reader.next_event().unwrap()).0, 1);
    }

    #[test]
    fn resume_from_committed_offset() {
        let mut file = Vec::new();
        write_record(&mut file, &scalar_event_payload(1, "loss", 1.0));
        write_record(&mut file, &scalar_event_payload(2, "loss", 2.0));

        let mut reader = EventFileReader::new(Cursor::new(&file));
        reader.next_event().unwrap();
        let offset = reader.committed_offset();
        drop(reader);

        let mut resumed = EventFileReader::resume(Cursor::new(&file), offset).unwrap();
        assert_eq!(scalar_of(&resumed.next_event().unwrap()).0, 2);
        assert!(matches!(
            resumed.next_event(),
            Err(ReadEventError::Truncated)
        ));
    }
}
