//! Parses the recorded fixtures in `fixtures/` — files written by real
//! writers, checked in with the scripts that made them. Synthetic bytes test
//! the decoder's logic; these test its agreement with reality.

use std::fs::File;
use std::path::PathBuf;

use tfevents::{
    Event, EventFileReader, EventPayload, ReadEventError, RecordReader, SummaryPayload,
};

fn fixture_events_file(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("fixture dir {} missing ({err}); run fixtures/generate/", dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.contains("tfevents"))
        })
        .collect();
    files.sort();
    assert_eq!(files.len(), 1, "expected exactly one events file in {}", dir.display());
    files.remove(0)
}

fn read_all(path: &PathBuf) -> Vec<Event> {
    let file = File::open(path).unwrap();
    let size = file.metadata().unwrap().len();
    let mut reader = EventFileReader::new(file);
    let mut events = Vec::new();
    loop {
        match reader.next_event() {
            Ok(event) => events.push(event),
            Err(ReadEventError::Truncated) => break,
            Err(err) => panic!("unexpected error in {}: {err}", path.display()),
        }
    }
    // A finished writer leaves no torn tail: the reader must have consumed
    // the file exactly.
    assert_eq!(reader.committed_offset(), size);
    events
}

#[test]
fn tensorboardx_fixture_parses_fully() {
    let path = fixture_events_file("tensorboardx");
    let events = read_all(&path);

    // First event announces the format version.
    assert_eq!(
        events[0].payload,
        EventPayload::FileVersion("brain.Event:2".into())
    );

    let mut losses = Vec::new();
    let mut accuracies = Vec::new();
    let mut histogram_steps = Vec::new();
    let mut texts = Vec::new();
    let mut images = Vec::new();
    for event in &events {
        let EventPayload::Summary(values) = &event.payload else {
            continue;
        };
        assert!(event.wall_time > 1.7e9, "wall time looks like epoch seconds");
        for value in values {
            match (value.tag.as_str(), &value.payload) {
                ("train/loss", _) => losses.push((event.step, value.scalar().unwrap())),
                ("train/accuracy", _) => {
                    accuracies.push((event.step, value.scalar().unwrap()));
                }
                ("params/weights", SummaryPayload::Histogram(histogram)) => {
                    assert!(histogram.num > 0.0);
                    assert!(!histogram.bucket.is_empty());
                    assert_eq!(histogram.bucket_limit.len(), histogram.bucket.len());
                    histogram_steps.push(event.step);
                }
                ("notes/text_summary", SummaryPayload::Tensor(tensor)) => {
                    texts.push(tensor.strings().unwrap()[0].clone());
                }
                ("samples/red", SummaryPayload::Image(image)) => images.push(image.clone()),
                (tag, payload) => panic!("unexpected value {tag}: {payload:?}"),
            }
        }
    }

    // Scalar series: 20 points each, steps in order, values per the
    // generator's formulas (computed in f64, recorded as f32).
    assert_eq!(losses.len(), 20);
    assert_eq!(accuracies.len(), 20);
    for (index, &(step, value)) in losses.iter().enumerate() {
        assert_eq!(step, index as i64);
        let expected = if step == 13 {
            25.0
        } else {
            4.0 * (-0.25 * step as f64).exp() + 0.5
        };
        assert!(
            (value - expected).abs() <= 1e-6 * expected.abs(),
            "loss at step {step}: recorded {value}, expected {expected}"
        );
    }
    // The spike survives, exactly.
    assert_eq!(losses[13].1, 25.0);

    assert_eq!(histogram_steps, vec![0, 5, 10, 15]);
    assert_eq!(texts.len(), 1);
    assert_eq!(
        String::from_utf8_lossy(&texts[0]),
        "fixture recorded by tensorboardx.py"
    );
    assert_eq!(images.len(), 1);
    assert_eq!((images[0].height, images[0].width), (4, 4));
    assert!(images[0].encoded_image.starts_with(b"\x89PNG"));
}

#[test]
fn tensorboardx_hparams_fixture_yields_typed_values() {
    let path = fixture_events_file("tensorboardx-hparams/hparam-session");
    let events = read_all(&path);

    let mut hparams = None;
    let mut final_loss = None;
    for event in &events {
        let EventPayload::Summary(values) = &event.payload else {
            continue;
        };
        for value in values {
            if let Some(decoded) = tfevents::session_start_hparams(value) {
                hparams = Some(decoded.unwrap());
            }
            if value.tag == "metrics/final_loss" {
                final_loss = value.scalar();
            }
        }
    }

    let hparams = hparams.expect("session_start_info present");
    use tfevents::HparamValue;
    assert_eq!(hparams["lr"], HparamValue::F64(0.001));
    assert_eq!(hparams["optimizer"], HparamValue::String("adam".into()));
    assert_eq!(hparams["amsgrad"], HparamValue::Bool(true));
    assert_eq!(hparams["layers"], HparamValue::F64(4.0));
    assert_eq!(final_loss, Some(0.75));
}

#[test]
fn tensorboardx_fixture_checksums_are_valid() {
    // The reader skips payload checksums on the hot path; verify here that
    // every record in the recorded file actually carries a valid one.
    let path = fixture_events_file("tensorboardx");
    let mut file = File::open(&path).unwrap();
    let mut records = RecordReader::new();
    let mut count = 0usize;
    loop {
        match records.read_record(&mut file) {
            Ok(record) => {
                record.checksum().unwrap();
                count += 1;
            }
            Err(tfevents::ReadRecordError::Truncated) => break,
            Err(err) => panic!("{err}"),
        }
    }
    assert!(count > 40, "expected a realistic record count, got {count}");
}
