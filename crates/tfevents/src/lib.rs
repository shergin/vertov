//! Read TensorBoard `tfevents` files: TFRecord framing and minimal Event
//! decoding. Zero dependencies, hand-rolled varint/CRC32C.
//!
//! The reader is built for the files trainers actually leave behind: a
//! truncated tail is a normal state (a writer may still be appending — or may
//! have crashed mid-record), not an error; corrupt records are reported as
//! visible data loss, never silent gaps; readers are resumable from a byte
//! offset so live tailing is an incremental read, not a re-parse.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod crc;

pub use crc::MaskedCrc;
