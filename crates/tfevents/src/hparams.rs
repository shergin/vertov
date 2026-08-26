//! The hparams plugin: hyperparameter values ride in summary *metadata*
//! under fixed tags (`_hparams_/session_start_info` carries the per-run
//! values), with no tensor payload at all.
//!
//! Wire path: `SummaryMetadata.plugin_data.content` (plugin name `hparams`)
//! is an `HParamsPluginData` whose field 3 is a `SessionStartInfo`; its
//! field 1 is a `map<string, google.protobuf.Value>` of the run's
//! hyperparameters. Values are typed string / f64 / bool; anything else the
//! plugin cannot produce.

use std::collections::BTreeMap;

use crate::event::SummaryValue;
use crate::wire::{DecodeError, MessageReader};

/// The tag whose metadata carries a run's hyperparameter values.
pub const SESSION_START_INFO_TAG: &str = "_hparams_/session_start_info";

/// A typed hyperparameter value.
#[derive(Clone, PartialEq, Debug)]
pub enum HparamValue {
    /// A numeric hyperparameter (`number_value`).
    F64(f64),
    /// A string hyperparameter (`string_value`).
    String(String),
    /// A boolean hyperparameter (`bool_value`).
    Bool(bool),
}

/// Extracts a run's hyperparameters from a summary value, if it is the
/// hparams plugin's session-start marker.
///
/// Returns `None` for every other summary value; `Some(Err(_))` only when
/// the marker is present but its plugin content does not decode.
pub fn session_start_hparams(
    value: &SummaryValue,
) -> Option<Result<BTreeMap<String, HparamValue>, DecodeError>> {
    let metadata = value.metadata.as_ref()?;
    if metadata.plugin_name != "hparams" {
        return None;
    }
    // Match on the fixed tag, not just the plugin name: the same plugin owns
    // the experiment and session-end markers too.
    if value.tag != SESSION_START_INFO_TAG {
        return None;
    }
    Some(decode_plugin_data(&metadata.plugin_content))
}

/// Decodes `HParamsPluginData`, returning the hparams of its
/// `session_start_info` (field 3), or an empty map when the content carries
/// one of the other markers.
fn decode_plugin_data(buf: &[u8]) -> Result<BTreeMap<String, HparamValue>, DecodeError> {
    let mut reader = MessageReader::new(buf);
    while let Some((field, value)) = reader.next_field()? {
        if field == 3
            && let Some(bytes) = value.bytes()
        {
            return decode_session_start_info(bytes);
        }
    }
    Ok(BTreeMap::new())
}

/// Decodes `SessionStartInfo`: field 1 is the hparams map.
fn decode_session_start_info(buf: &[u8]) -> Result<BTreeMap<String, HparamValue>, DecodeError> {
    let mut hparams = BTreeMap::new();
    let mut reader = MessageReader::new(buf);
    while let Some((field, value)) = reader.next_field()? {
        if field == 1
            && let Some(bytes) = value.bytes()
        {
            // One map entry: key = 1, value = 2 (google.protobuf.Value).
            let mut key = String::new();
            let mut entry_value = None;
            let mut entry = MessageReader::new(bytes);
            while let Some((field, value)) = entry.next_field()? {
                match field {
                    1 => {
                        if let Some(bytes) = value.bytes() {
                            key = String::from_utf8_lossy(bytes).into_owned();
                        }
                    }
                    2 => {
                        if let Some(bytes) = value.bytes() {
                            entry_value = decode_value(bytes)?;
                        }
                    }
                    _ => {}
                }
            }
            if let Some(entry_value) = entry_value {
                hparams.insert(key, entry_value);
            }
        }
    }
    Ok(hparams)
}

/// Decodes a `google.protobuf.Value` into the three kinds hparams can carry.
/// Null, struct, and list values decode to `None` — the plugin never writes
/// them for hyperparameters.
fn decode_value(buf: &[u8]) -> Result<Option<HparamValue>, DecodeError> {
    let mut out = None;
    let mut reader = MessageReader::new(buf);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            2 => {
                if let Some(v) = value.double() {
                    out = Some(HparamValue::F64(v));
                }
            }
            3 => {
                if let Some(bytes) = value.bytes() {
                    out = Some(HparamValue::String(
                        String::from_utf8_lossy(bytes).into_owned(),
                    ));
                }
            }
            4 => {
                if let Some(v) = value.varint() {
                    out = Some(HparamValue::Bool(v != 0));
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{Event, EventPayload};
    use crate::wire::testutil::*;

    fn session_start_event(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut info = Vec::new();
        for (key, value) in entries {
            let mut entry = Vec::new();
            field_bytes(1, key.as_bytes(), &mut entry);
            field_bytes(2, value, &mut entry);
            field_bytes(1, &entry, &mut info);
        }
        let mut plugin_data = Vec::new();
        field_varint(1, 0, &mut plugin_data); // version
        field_bytes(3, &info, &mut plugin_data);

        let mut inner = Vec::new();
        field_bytes(1, b"hparams", &mut inner);
        field_bytes(2, &plugin_data, &mut inner);
        let mut metadata = Vec::new();
        field_bytes(1, &inner, &mut metadata);

        let mut summary_value = Vec::new();
        field_bytes(1, SESSION_START_INFO_TAG.as_bytes(), &mut summary_value);
        field_bytes(9, &metadata, &mut summary_value);
        let mut summary = Vec::new();
        field_bytes(1, &summary_value, &mut summary);
        let mut event = Vec::new();
        field_bytes(5, &summary, &mut event);
        event
    }

    #[test]
    fn decodes_typed_hparams() {
        let mut lr = Vec::new();
        field_double(2, 0.001, &mut lr);
        let mut optimizer = Vec::new();
        field_bytes(3, b"adam", &mut optimizer);
        let mut amsgrad = Vec::new();
        field_varint(4, 1, &mut amsgrad);
        let event_bytes = session_start_event(&[
            ("lr", lr),
            ("optimizer", optimizer),
            ("amsgrad", amsgrad),
        ]);

        let event = Event::decode(&event_bytes).unwrap();
        let EventPayload::Summary(values) = &event.payload else {
            panic!("expected summary");
        };
        let hparams = session_start_hparams(&values[0]).unwrap().unwrap();
        assert_eq!(hparams["lr"], HparamValue::F64(0.001));
        assert_eq!(hparams["optimizer"], HparamValue::String("adam".into()));
        assert_eq!(hparams["amsgrad"], HparamValue::Bool(true));
    }

    #[test]
    fn other_summaries_are_not_hparams() {
        let mut summary_value = Vec::new();
        field_bytes(1, b"train/loss", &mut summary_value);
        field_float(2, 1.0, &mut summary_value);
        let mut summary = Vec::new();
        field_bytes(1, &summary_value, &mut summary);
        let mut event = Vec::new();
        field_bytes(5, &summary, &mut event);

        let event = Event::decode(&event).unwrap();
        let EventPayload::Summary(values) = &event.payload else {
            panic!("expected summary");
        };
        assert!(session_start_hparams(&values[0]).is_none());
    }
}
