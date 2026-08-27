//! The wandb offline `.wandb` file: a 7-byte header, then LevelDB-log
//! framing (32 KiB blocks of CRC'd chunks; records span chunks, chunks
//! never span blocks), each record a `wandb_internal.Record` proto. The
//! format has strong de-facto stability — a version byte gates hard breaks,
//! and the replay contract keeps old files readable forever.
//!
//! Everything here is pure over byte slices; the model owns files and
//! offsets. Live tailing follows LEET's recipe: parse to the last complete
//! record, remember that offset, retry when the file grows.

use crate::ParamValue;

/// The 7-byte file header: `":W&B"` + `0xBEE1` LE + version.
pub const HEADER_LEN: u64 = 7;

const BLOCK: usize = 32 * 1024;
const CHUNK_HEADER: usize = 7;

/// Checks the file header. `Ok(())` for version 0; newer versions fail
/// loudly (the format's own contract for breaking changes) and other bytes
/// mean this is not a `.wandb` file at all.
pub fn check_header(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() < HEADER_LEN as usize {
        return Err("file shorter than the wandb header".to_owned());
    }
    if &bytes[..4] != b":W&B" || bytes[4..6] != [0xE1, 0xBE] {
        return Err("not a wandb log file".to_owned());
    }
    if bytes[6] != 0 {
        return Err(format!(
            "wandb log version {} is newer than this reader understands",
            bytes[6]
        ));
    }
    Ok(())
}

/// Assembled record payloads from `buf`, which starts at absolute file
/// position `base` (a previously committed offset; the caller strips the
/// file header on first read). Returns the payloads and the absolute offset
/// just past the last *complete* record — a torn tail stays unconsumed, a
/// state to resume from.
pub fn read_records(buf: &[u8], base: u64) -> (Vec<Vec<u8>>, u64) {
    let mut records = Vec::new();
    let mut committed = base;
    let mut position = 0usize;
    // A record under assembly from FIRST/MIDDLE chunks.
    let mut partial: Vec<u8> = Vec::new();
    let mut in_partial = false;

    loop {
        // Chunks never cross a 32 KiB block boundary (relative to the file
        // start): fewer than 7 bytes left in a block is zero padding.
        let block_used = ((base as usize) + position) % BLOCK;
        let block_left = BLOCK - block_used;
        if block_left < CHUNK_HEADER {
            if position + block_left > buf.len() {
                break;
            }
            position += block_left;
            // Padding carries no data; a complete record boundary here
            // moves the committed offset past it.
            if !in_partial {
                committed = base + position as u64;
            }
            continue;
        }
        if position + CHUNK_HEADER > buf.len() {
            break;
        }
        let header = &buf[position..position + CHUNK_HEADER];
        let stored_crc = u32::from_le_bytes(header[..4].try_into().expect("4 bytes"));
        let len = u16::from_le_bytes(header[4..6].try_into().expect("2 bytes")) as usize;
        let kind = header[6];
        // An all-zero chunk header is mmap preallocation: skip to the next
        // block boundary.
        if stored_crc == 0 && len == 0 && kind == 0 {
            if position + block_left > buf.len() {
                break;
            }
            position += block_left;
            if !in_partial {
                committed = base + position as u64;
            }
            continue;
        }
        if len > block_left - CHUNK_HEADER {
            // A chunk claiming to cross a block: framing is untrustworthy
            // from here on; keep the valid prefix.
            break;
        }
        if position + CHUNK_HEADER + len > buf.len() {
            break;
        }
        let payload = &buf[position + CHUNK_HEADER..position + CHUNK_HEADER + len];
        // Plain CRC-32/IEEE over type byte then payload — not the masked
        // CRC32C tfevents uses.
        let mut crc = Crc32::new();
        crc.update(&[kind]);
        crc.update(payload);
        if crc.finish() != stored_crc {
            break;
        }
        position += CHUNK_HEADER + len;

        match kind {
            1 => {
                // FULL
                records.push(payload.to_vec());
                partial.clear();
                in_partial = false;
                committed = base + position as u64;
            }
            2 => {
                // FIRST
                partial.clear();
                partial.extend_from_slice(payload);
                in_partial = true;
            }
            3 if in_partial => {
                // MIDDLE
                partial.extend_from_slice(payload);
            }
            4 if in_partial => {
                // LAST
                partial.extend_from_slice(payload);
                records.push(std::mem::take(&mut partial));
                in_partial = false;
                committed = base + position as u64;
            }
            _ => break,
        }
    }
    (records, committed)
}

/// What one `wandb_internal.Record` means to a viewer.
#[derive(Clone, PartialEq, Debug)]
pub enum WandbRecord {
    /// A metrics row: step, wall seconds, and the numeric values by key.
    History {
        /// Global step (`_step` when logged, else the history step).
        step: i64,
        /// Seconds since the epoch from `_timestamp`, 0.0 when absent.
        wall: f64,
        /// Numeric metrics; non-numeric history values are skipped.
        values: Vec<(String, f64)>,
    },
    /// Config updates: typed values by key.
    Config(Vec<(String, ParamValue)>),
    /// The run exited cleanly — its absence marks a crashed run.
    Exit,
    /// Anything else (summary, stats, console output, run info, header).
    Other,
}

/// Decodes one record payload. `None` when the bytes are not a decodable
/// record — visible loss for the caller to count.
pub fn parse_record(payload: &[u8]) -> Option<WandbRecord> {
    let mut reader = Wire::new(payload);
    while let Some((field, value)) = reader.next()? {
        match field {
            2 => return parse_history(value.bytes()?),
            5 => return parse_config(value.bytes()?),
            18 => return Some(WandbRecord::Exit),
            _ => {}
        }
    }
    Some(WandbRecord::Other)
}

fn parse_history(buf: &[u8]) -> Option<WandbRecord> {
    let mut step = None;
    let mut wall = 0.0;
    let mut values = Vec::new();
    let mut reader = Wire::new(buf);
    while let Some((field, value)) = reader.next()? {
        match field {
            // repeated HistoryItem item = 1
            1 => {
                let (key, json) = parse_item(value.bytes()?)?;
                match key.as_str() {
                    "_step" => {
                        if let Some(number) = json_number(&json) {
                            step = Some(number as i64);
                        }
                    }
                    "_timestamp" => {
                        if let Some(number) = json_number(&json) {
                            wall = number;
                        }
                    }
                    "_runtime" => {}
                    _ => {
                        if let Some(number) = json_number(&json) {
                            values.push((key, number));
                        }
                    }
                }
            }
            // HistoryStep step = 2 { int64 num = 1 }
            2 => {
                let mut inner = Wire::new(value.bytes()?);
                while let Some((field, value)) = inner.next()? {
                    if field == 1 {
                        step.get_or_insert(value.varint()? as i64);
                    }
                }
            }
            _ => {}
        }
    }
    Some(WandbRecord::History {
        step: step.unwrap_or(0),
        wall,
        values,
    })
}

fn parse_config(buf: &[u8]) -> Option<WandbRecord> {
    let mut updates = Vec::new();
    let mut reader = Wire::new(buf);
    while let Some((field, value)) = reader.next()? {
        // repeated ConfigItem update = 1
        if field == 1 {
            let (key, json) = parse_item(value.bytes()?)?;
            if key.starts_with('_') {
                continue;
            }
            if let Some(param) = json_param(&json) {
                updates.push((key, param));
            }
        }
    }
    Some(WandbRecord::Config(updates))
}

/// HistoryItem / ConfigItem: key = 1, nested_key = 2, value_json = 16.
fn parse_item(buf: &[u8]) -> Option<(String, String)> {
    let mut key = String::new();
    let mut nested: Vec<String> = Vec::new();
    let mut json = String::new();
    let mut reader = Wire::new(buf);
    while let Some((field, value)) = reader.next()? {
        match field {
            1 => key = String::from_utf8_lossy(value.bytes()?).into_owned(),
            2 => nested.push(String::from_utf8_lossy(value.bytes()?).into_owned()),
            16 => json = String::from_utf8_lossy(value.bytes()?).into_owned(),
            _ => {}
        }
    }
    if !nested.is_empty() {
        key = nested.join("/");
    }
    Some((key, json))
}

/// A JSON number, tolerating Python's bare `NaN`/`Infinity` tokens (which
/// the format really does emit).
fn json_number(json: &str) -> Option<f64> {
    match json.trim() {
        "NaN" | "nan" => Some(f64::NAN),
        "Infinity" | "inf" => Some(f64::INFINITY),
        "-Infinity" | "-inf" => Some(f64::NEG_INFINITY),
        "true" | "false" | "null" => None,
        trimmed => trimmed.parse().ok(),
    }
}

/// A JSON scalar as a typed param. Config values sometimes arrive wrapped
/// as `{"value": X}` (the config.yaml shape); one unwrap level handles it.
fn json_param(json: &str) -> Option<ParamValue> {
    let trimmed = json.trim();
    if let Some(inner) = trimmed
        .strip_prefix('{')
        .and_then(|rest| rest.strip_suffix('}'))
        && let Some((key, value)) = inner.split_once(':')
        && key.trim() == "\"value\""
    {
        return json_param(value);
    }
    match trimmed {
        "true" => return Some(ParamValue::Bool(true)),
        "false" => return Some(ParamValue::Bool(false)),
        "null" => return None,
        _ => {}
    }
    if let Some(number) = json_number(trimmed) {
        return Some(ParamValue::Number(number));
    }
    let unquoted = trimmed
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))?;
    Some(ParamValue::Text(
        unquoted.replace("\\\"", "\"").replace("\\\\", "\\"),
    ))
}

// ---- minimal protobuf wire walking (self-contained on purpose) ----

enum WireValue<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
}

impl<'a> WireValue<'a> {
    fn bytes(&self) -> Option<&'a [u8]> {
        match self {
            WireValue::Bytes(bytes) => Some(bytes),
            WireValue::Varint(_) => None,
        }
    }

    fn varint(&self) -> Option<u64> {
        match self {
            WireValue::Varint(value) => Some(*value),
            WireValue::Bytes(_) => None,
        }
    }
}

struct Wire<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Wire<'a> {
    fn new(buf: &'a [u8]) -> Wire<'a> {
        Wire { buf, pos: 0 }
    }

    /// `Some(None)` at end, `None` on malformed bytes.
    #[allow(clippy::option_option)]
    fn next(&mut self) -> Option<Option<(u32, WireValue<'a>)>> {
        if self.pos == self.buf.len() {
            return Some(None);
        }
        let key = self.varint_raw()?;
        let field = u32::try_from(key >> 3).ok()?;
        let value = match key & 7 {
            0 => WireValue::Varint(self.varint_raw()?),
            1 => {
                self.take(8)?;
                WireValue::Varint(0)
            }
            2 => {
                let len = usize::try_from(self.varint_raw()?).ok()?;
                WireValue::Bytes(self.take(len)?)
            }
            5 => {
                self.take(4)?;
                WireValue::Varint(0)
            }
            _ => return None,
        };
        Some(Some((field, value)))
    }

    fn varint_raw(&mut self) -> Option<u64> {
        let mut result = 0u64;
        for shift in (0..64).step_by(7) {
            let &byte = self.buf.get(self.pos)?;
            self.pos += 1;
            if shift == 63 && byte & 0x7E != 0 {
                return None;
            }
            result |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Some(result);
            }
        }
        None
    }

    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(len)?;
        let slice = self.buf.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }
}

// ---- CRC-32/IEEE ----

struct Crc32(u32);

impl Crc32 {
    fn new() -> Crc32 {
        Crc32(!0)
    }

    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u32::from(byte);
            for _ in 0..8 {
                self.0 = if self.0 & 1 != 0 {
                    (self.0 >> 1) ^ 0xEDB8_8320
                } else {
                    self.0 >> 1
                };
            }
        }
    }

    fn finish(self) -> u32 {
        !self.0
    }
}

/// Test/fixture aid: writes a whole `.wandb` file (header + framed
/// records). vertov never writes into a run directory; this exists so the
/// readers can be exercised without a wandb installation.
pub mod writer {
    use super::{BLOCK, CHUNK_HEADER, Crc32};

    /// A complete file from pre-encoded record payloads.
    pub fn wandb_file(records: &[Vec<u8>]) -> Vec<u8> {
        let mut out = vec![b':', b'W', b'&', b'B', 0xE1, 0xBE, 0x00];
        for record in records {
            append_record(&mut out, record);
        }
        out
    }

    /// Appends one record, splitting into FIRST/MIDDLE/LAST chunks at block
    /// boundaries exactly like the real writer.
    pub fn append_record(out: &mut Vec<u8>, payload: &[u8]) {
        let mut remaining = payload;
        let mut first = true;
        loop {
            let mut room = BLOCK - (out.len() % BLOCK);
            if room < CHUNK_HEADER {
                out.resize(out.len() + room, 0);
                room = BLOCK;
            }
            let take = remaining.len().min(room - CHUNK_HEADER);
            let last = take == remaining.len();
            let kind = match (first, last) {
                (true, true) => 1,
                (true, false) => 2,
                (false, false) => 3,
                (false, true) => 4,
            };
            let (chunk, rest) = remaining.split_at(take);
            let mut crc = Crc32::new();
            crc.update(&[kind]);
            crc.update(chunk);
            out.extend_from_slice(&crc.finish().to_le_bytes());
            out.extend_from_slice(&(take as u16).to_le_bytes());
            out.push(kind);
            out.extend_from_slice(chunk);
            if last {
                return;
            }
            remaining = rest;
            first = false;
        }
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

    fn item(key: &str, json: &str) -> Vec<u8> {
        let mut out = Vec::new();
        field_bytes(1, key.as_bytes(), &mut out);
        field_bytes(16, json.as_bytes(), &mut out);
        out
    }

    /// A history record: `_step`/`_timestamp` plus the given metrics.
    pub fn history_record(step: i64, wall: f64, values: &[(&str, &str)]) -> Vec<u8> {
        let mut history = Vec::new();
        field_bytes(1, &item("_step", &step.to_string()), &mut history);
        if wall != 0.0 {
            field_bytes(1, &item("_timestamp", &wall.to_string()), &mut history);
        }
        for (key, json) in values {
            field_bytes(1, &item(key, json), &mut history);
        }
        let mut record = Vec::new();
        field_bytes(2, &history, &mut record);
        record
    }

    /// A config record.
    pub fn config_record(updates: &[(&str, &str)]) -> Vec<u8> {
        let mut config = Vec::new();
        for (key, json) in updates {
            field_bytes(1, &item(key, json), &mut config);
        }
        let mut record = Vec::new();
        field_bytes(5, &config, &mut record);
        record
    }

    /// An exit record.
    pub fn exit_record() -> Vec<u8> {
        let mut record = Vec::new();
        field_bytes(18, &[], &mut record);
        record
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_gates_versions() {
        let file = writer::wandb_file(&[]);
        assert!(check_header(&file).is_ok());
        let mut newer = file.clone();
        newer[6] = 1;
        assert!(check_header(&newer).unwrap_err().contains("version 1"));
        assert!(check_header(b":NOPE\x00\x00").is_err());
        assert!(check_header(b":W").is_err());
    }

    #[test]
    fn records_roundtrip_and_resume() {
        let history = writer::history_record(3, 1.7e9, &[("loss", "0.5"), ("acc", "0.9")]);
        let config = writer::config_record(&[("lr", "0.001"), ("opt", "\"adam\"")]);
        let file = writer::wandb_file(&[history, config, writer::exit_record()]);

        let (records, committed) = read_records(&file[HEADER_LEN as usize..], HEADER_LEN);
        assert_eq!(records.len(), 3);
        assert_eq!(committed, file.len() as u64);

        match parse_record(&records[0]).unwrap() {
            WandbRecord::History { step, wall, values } => {
                assert_eq!(step, 3);
                assert_eq!(wall, 1.7e9);
                assert_eq!(
                    values,
                    vec![("loss".to_owned(), 0.5), ("acc".to_owned(), 0.9)]
                );
            }
            other => panic!("expected history, got {other:?}"),
        }
        match parse_record(&records[1]).unwrap() {
            WandbRecord::Config(updates) => {
                assert_eq!(updates[0], ("lr".to_owned(), ParamValue::Number(0.001)));
                assert_eq!(updates[1], ("opt".to_owned(), ParamValue::Text("adam".into())));
            }
            other => panic!("expected config, got {other:?}"),
        }
        assert_eq!(parse_record(&records[2]).unwrap(), WandbRecord::Exit);
    }

    #[test]
    fn torn_tail_is_a_state() {
        let record = writer::history_record(0, 0.0, &[("loss", "1.0")]);
        let file = writer::wandb_file(&[record.clone(), record]);
        // Cut mid-second-record: only the first parses, and the committed
        // offset points at the boundary between them.
        for cut in (HEADER_LEN as usize + 1)..file.len() {
            let (records, committed) = read_records(&file[HEADER_LEN as usize..cut], HEADER_LEN);
            assert!(records.len() <= 2);
            if records.len() == 1 {
                // Feeding the remainder from the committed offset yields
                // exactly the second record.
                let (rest, end) = read_records(&file[committed as usize..], committed);
                assert_eq!(rest.len(), 1);
                assert_eq!(end, file.len() as u64);
            }
        }
    }

    #[test]
    fn large_records_span_blocks() {
        // A metric whose JSON is ~100 KiB forces FIRST/MIDDLE/LAST chunks
        // across four blocks.
        let big = format!("\"{}\"", "x".repeat(100_000));
        let record = writer::history_record(1, 0.0, &[("note", &big), ("loss", "2.5")]);
        let tail = writer::history_record(2, 0.0, &[("loss", "1.5")]);
        let file = writer::wandb_file(&[record, tail]);
        assert!(file.len() > BLOCK * 3);

        let (records, committed) = read_records(&file[HEADER_LEN as usize..], HEADER_LEN);
        assert_eq!(records.len(), 2);
        assert_eq!(committed, file.len() as u64);
        match parse_record(&records[1]).unwrap() {
            WandbRecord::History { step, values, .. } => {
                assert_eq!(step, 2);
                assert_eq!(values, vec![("loss".to_owned(), 1.5)]);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn zero_chunks_and_padding_skip() {
        let record = writer::history_record(0, 0.0, &[("loss", "1.0")]);
        let mut file = writer::wandb_file(std::slice::from_ref(&record));
        // mmap preallocation: zeros to the end of the block, then a real
        // record in the next block.
        let pad = BLOCK - (file.len() % BLOCK);
        file.resize(file.len() + pad, 0);
        writer::append_record(&mut file, &record);
        let (records, committed) = read_records(&file[HEADER_LEN as usize..], HEADER_LEN);
        assert_eq!(records.len(), 2);
        assert_eq!(committed, file.len() as u64);
    }

    #[test]
    fn nan_tokens_parse() {
        assert!(json_number("NaN").unwrap().is_nan());
        assert_eq!(json_number("Infinity"), Some(f64::INFINITY));
        assert_eq!(json_number("-Infinity"), Some(f64::NEG_INFINITY));
        assert_eq!(json_number("1.5e-3"), Some(0.0015));
        assert_eq!(json_number("\"text\""), None);
    }

    #[test]
    fn config_value_wrapper_unwraps() {
        assert_eq!(
            json_param("{\"value\": 0.01}"),
            Some(ParamValue::Number(0.01))
        );
        assert_eq!(json_param("true"), Some(ParamValue::Bool(true)));
        assert_eq!(json_param("null"), None);
    }
}
