//! Cold-load benchmark: read and decode a synthetic in-memory events file of
//! one million TF1 scalar points — the Phase 1 budget from `BENCHMARKS.md`.
//!
//! In memory rather than on disk so the number measures the parser, not the
//! page cache; real cold starts add I/O on top.

use std::hint::black_box;
use std::io::Cursor;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tfevents::{EventFileReader, EventPayload, MaskedCrc, ReadEventError, RecordReader};

// Bench-local writers: the crate deliberately ships no writer API.

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

fn scalar_event(step: u64, tag: &str, value: f32) -> Vec<u8> {
    let mut summary_value = Vec::new();
    field_bytes(1, tag.as_bytes(), &mut summary_value);
    varint(2 << 3 | 5, &mut summary_value);
    summary_value.extend_from_slice(&value.to_bits().to_le_bytes());
    let mut summary = Vec::new();
    field_bytes(1, &summary_value, &mut summary);
    let mut event = Vec::new();
    varint(1 << 3 | 1, &mut event);
    event.extend_from_slice(&(1.7e9 + step as f64).to_bits().to_le_bytes());
    varint(2 << 3, &mut event);
    varint(step, &mut event);
    field_bytes(5, &summary, &mut event);
    event
}

fn write_record(out: &mut Vec<u8>, payload: &[u8]) {
    let length = payload.len() as u64;
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(&MaskedCrc::compute(&length.to_le_bytes()).0.to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&MaskedCrc::compute(payload).0.to_le_bytes());
}

fn million_point_file() -> Vec<u8> {
    let mut file = Vec::new();
    for step in 0..1_000_000u64 {
        write_record(&mut file, &scalar_event(step, "train/loss", step as f32));
    }
    file
}

fn cold_load(criterion: &mut Criterion) {
    let file = million_point_file();
    let mut group = criterion.benchmark_group("cold_load");
    group.throughput(Throughput::Bytes(file.len() as u64));
    group.sample_size(20);

    // Framing only: records surfaced, payloads undecoded, checksums skipped.
    group.bench_function("frames_1m", |bencher| {
        bencher.iter(|| {
            let mut cursor = Cursor::new(&file);
            let mut reader = RecordReader::new();
            let mut count = 0u64;
            while let Ok(record) = reader.read_record(&mut cursor) {
                count += record.data.len() as u64;
            }
            black_box(count)
        });
    });

    // The full path: framing + Event decode + scalar extraction.
    group.bench_function("events_1m", |bencher| {
        bencher.iter(|| {
            let mut reader = EventFileReader::new(Cursor::new(&file));
            let mut sum = 0.0f64;
            loop {
                match reader.next_event() {
                    Ok(event) => {
                        if let EventPayload::Summary(values) = &event.payload {
                            for value in values {
                                if let Some(scalar) = value.scalar() {
                                    sum += scalar;
                                }
                            }
                        }
                    }
                    Err(ReadEventError::Truncated) => break,
                    Err(err) => panic!("{err}"),
                }
            }
            black_box(sum)
        });
    });
    group.finish();
}

criterion_group!(benches, cold_load);
criterion_main!(benches);
