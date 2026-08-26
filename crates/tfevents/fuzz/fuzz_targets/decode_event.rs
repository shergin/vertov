//! The proto layer alone: arbitrary bytes as one record payload, bypassing
//! framing so coverage guidance is not walled off behind a 32-bit CRC.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tfevents::{Event, EventPayload, SummaryPayload};

fuzz_target!(|data: &[u8]| {
    let Ok(event) = Event::decode(data) else {
        return;
    };
    if let EventPayload::Summary(values) = &event.payload {
        for value in values {
            let _ = value.scalar();
            let _ = tfevents::session_start_hparams(value);
            if let SummaryPayload::Tensor(tensor) = &value.payload {
                let _ = tensor.scalar();
                let _ = tensor.doubles();
                let _ = tensor.strings();
            }
        }
    }
});
