//! Generates a playground logdir exercising every feature at once: several
//! tfevents runs (spikes, NaN gaps, a restart with ghost data, a token
//! counter, histograms, long and non-ASCII tags), plus a dvclive run, an
//! MLflow run, and a wandb offline run — a mixed root for manual QA.
//!
//! Usage: `cargo run -p vertov-model --example playground -- <dir>`

use std::fs;
use std::path::Path;

use tfevents::writer::{events_file, histogram_event, scalar_event, write_record};

fn main() {
    let root = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: playground <dir>");
        std::process::exit(2);
    });
    let root = Path::new(&root);
    let _ = fs::remove_dir_all(root);
    fs::create_dir_all(root).unwrap();

    let wall = |step: i64| 1.7e9 + step as f64 * 30.0;

    // Three tfevents runs sweeping a learning rate; each logs loss (with a
    // spike), accuracy, a NaN gap, gradient histograms, and a token counter.
    for (index, lr) in [1e-2, 3e-3, 1e-3].iter().enumerate() {
        let name = format!("sweep/lr-{lr}");
        let dir = root.join(&name);
        fs::create_dir_all(&dir).unwrap();
        let mut events = Vec::new();
        for step in 0..120i64 {
            let noise = ((step * 2654435761 % 97) as f64 / 97.0 - 0.5) * 0.2;
            let mut loss = 5.0 * (-0.03 * step as f64 * (1.0 + index as f64)).exp() + 0.4 + noise;
            if step == 57 {
                loss = 24.0; // the spike no viewer may hide
            }
            let value = if step == 80 { f32::NAN } else { loss as f32 };
            events.push(scalar_event(wall(step), step, "train/loss", value));
            events.push(scalar_event(
                wall(step),
                step,
                "train/accuracy",
                (1.0 - 0.9 * (-0.02 * step as f64).exp()) as f32,
            ));
            events.push(scalar_event(
                wall(step),
                step,
                "tokens",
                (step * 4096) as f32,
            ));
            if step % 20 == 0 {
                let spread = 1.5 - step as f64 / 120.0;
                let buckets: Vec<(f64, f64, f64)> = (0..14)
                    .map(|bucket| {
                        let left = -2.0 + bucket as f64 * 0.3;
                        let center = (left + 0.15) / spread;
                        (left, left + 0.3, 40.0 * (-center * center).exp())
                    })
                    .collect();
                events.push(histogram_event(wall(step), step, "grads/layer0", &buckets));
            }
        }
        fs::write(dir.join("events.out.tfevents.1000.host"), events_file(&events)).unwrap();
    }

    // A restarted run: crashed at step 90, resumed from its step-60
    // checkpoint in a second file — ghost data, segments, a boundary.
    {
        let dir = root.join("restarted");
        fs::create_dir_all(&dir).unwrap();
        let first: Vec<Vec<u8>> = (0..90)
            .map(|step| {
                scalar_event(
                    wall(step),
                    step,
                    "train/loss",
                    (4.0 * (-0.02 * step as f64).exp() + 0.5) as f32,
                )
            })
            .collect();
        let second: Vec<Vec<u8>> = (60..150)
            .map(|step| {
                scalar_event(
                    wall(200 + step),
                    step,
                    "train/loss",
                    (4.0 * (-0.02 * step as f64).exp() + 0.7) as f32,
                )
            })
            .collect();
        fs::write(dir.join("events.out.tfevents.1000.host"), events_file(&first)).unwrap();
        fs::write(dir.join("events.out.tfevents.2000.host"), events_file(&second)).unwrap();
    }

    // A torn tail: a run whose last record is half-written (crashed writer).
    {
        let dir = root.join("torn-tail");
        fs::create_dir_all(&dir).unwrap();
        let mut bytes = events_file(
            &(0..40)
                .map(|step| scalar_event(wall(step), step, "train/loss", 2.0 - step as f32 * 0.01))
                .collect::<Vec<_>>(),
        );
        let mut torn = Vec::new();
        write_record(&mut torn, &scalar_event(wall(40), 40, "train/loss", 1.0));
        bytes.extend_from_slice(&torn[..torn.len() - 6]);
        fs::write(dir.join("events.out.tfevents.1000.host"), bytes).unwrap();
    }

    // Non-ASCII tags and a deep/long name.
    {
        let dir = root.join("unicode/эксперимент-длинное-имя-для-обрезки-колонок");
        fs::create_dir_all(&dir).unwrap();
        let events: Vec<Vec<u8>> = (0..30)
            .map(|step| {
                scalar_event(
                    wall(step),
                    step,
                    "метрики/потеря",
                    (1.0 / (step + 1) as f64) as f32,
                )
            })
            .collect();
        fs::write(dir.join("events.out.tfevents.1000.host"), events_file(&events)).unwrap();
    }

    // dvclive.
    {
        let live = root.join("dvc-exp/dvclive");
        fs::create_dir_all(live.join("plots/metrics")).unwrap();
        fs::write(live.join("params.yaml"), "lr: 0.0007\noptimizer: lion\n").unwrap();
        let mut tsv = String::from("timestamp\tstep\tloss\n");
        for step in 0..60i64 {
            use std::fmt::Write as _;
            let _ = writeln!(
                tsv,
                "{}\t{step}\t{}",
                (wall(step) * 1000.0) as i64,
                3.0 * (-0.05 * step as f64).exp() + 0.3
            );
        }
        fs::write(live.join("plots/metrics/loss.tsv"), tsv).unwrap();
    }

    // MLflow.
    {
        let run = root.join("mlruns/0/qa0run");
        fs::create_dir_all(run.join("metrics")).unwrap();
        fs::create_dir_all(run.join("params")).unwrap();
        fs::write(run.join("meta.yaml"), "run_id: qa0run\nrun_name: mlflow-qa\n").unwrap();
        let mut lines = String::new();
        for step in 0..50i64 {
            use std::fmt::Write as _;
            let _ = writeln!(
                lines,
                "{} {} {step}",
                (wall(step) * 1000.0) as i64,
                2.0 * (-0.04 * step as f64).exp() + 0.2
            );
        }
        fs::write(run.join("metrics/loss"), lines).unwrap();
        fs::write(run.join("params/lr"), "0.005").unwrap();
    }

    // wandb offline.
    {
        use vertov_formats::wandb::writer;
        let dir = root.join("wandb/offline-run-20260826_000000-qa1");
        fs::create_dir_all(&dir).unwrap();
        let mut records = vec![writer::config_record(&[
            ("lr", "0.002"),
            ("optimizer", "\"adam\""),
        ])];
        for step in 0..70i64 {
            records.push(writer::history_record(
                step,
                wall(step),
                &[("train/loss", &format!("{}", 2.5 * (-0.03 * step as f64).exp() + 0.25))],
            ));
        }
        records.push(writer::exit_record());
        fs::write(dir.join("run-qa1.wandb"), writer::wandb_file(&records)).unwrap();
    }

    println!("playground written to {}", root.display());
}
