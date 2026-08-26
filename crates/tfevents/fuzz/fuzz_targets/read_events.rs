//! End-to-end: arbitrary bytes as a whole events file, through framing,
//! decoding, and every public accessor. The contract under fuzz: no panic,
//! no hang, no unbounded allocation — corrupt input surfaces as the error
//! taxonomy, never as a crash.

#![no_main]

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;
use tfevents::{EventFileReader, EventPayload, ReadEventError, SummaryPayload};

fuzz_target!(|data: &[u8]| {
    let mut reader = EventFileReader::new(Cursor::new(data));
    loop {
        match reader.next_event() {
            Ok(event) => {
                if let EventPayload::Summary(values) = &event.payload {
                    for value in values {
                        let _ = value.scalar();
                        let _ = tfevents::session_start_hparams(value);
                        if let SummaryPayload::Tensor(tensor) = &value.payload {
                            let _ = tensor.doubles();
                            let _ = tensor.strings();
                        }
                    }
                }
            }
            // One spoiled record; the stream continues.
            Err(ReadEventError::Corrupt { .. } | ReadEventError::Malformed { .. }) => {}
            // Truncated, BadLengthCrc, Io: the stream is done.
            Err(_) => break,
        }
    }
});
