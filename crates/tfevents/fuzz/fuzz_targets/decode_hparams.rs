//! The hparams plugin decoder alone: arbitrary bytes as
//! `SummaryMetadata.plugin_data.content`, two message layers below anything
//! the other targets can steer into cheaply.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tfevents::hparams::SESSION_START_INFO_TAG;
use tfevents::{DataClass, SummaryMetadata, SummaryPayload, SummaryValue};

fuzz_target!(|data: &[u8]| {
    let value = SummaryValue {
        tag: SESSION_START_INFO_TAG.to_owned(),
        metadata: Some(SummaryMetadata {
            plugin_name: "hparams".to_owned(),
            plugin_content: data.to_vec(),
            data_class: DataClass::Unknown,
        }),
        payload: SummaryPayload::Other,
    };
    let _ = tfevents::session_start_hparams(&value);
});
