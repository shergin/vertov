//! dvclive and MLflow backend tests: discovery, summaries, hparams, live
//! tailing with torn lines, frontier-consistent materialization, and mixed
//! roots — every backend normalized into the same catalog.

use std::fs;
use std::path::{Path, PathBuf};

use vertov_model::{Backend, Project, SeriesClass};

struct Logdir(PathBuf);

impl Logdir {
    fn new(name: &str) -> Logdir {
        let dir = std::env::temp_dir().join(format!(
            "vertov-backends-test-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Logdir(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn write(&self, relative: &str, text: &str) {
        let path = self.0.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    fn append(&self, relative: &str, text: &str) {
        use std::io::Write as _;
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(self.0.join(relative))
            .unwrap();
        file.write_all(text.as_bytes()).unwrap();
    }
}

impl Drop for Logdir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn dvclive_run_end_to_end() {
    let logdir = Logdir::new("dvclive");
    logdir.write(
        "exp1/dvclive/params.yaml",
        "lr: 0.001\noptimizer: adam\nmodel:\n  layers: 4\n",
    );
    logdir.write(
        "exp1/dvclive/plots/metrics/train/loss.tsv",
        "timestamp\tstep\tloss\n1700000000000\t0\t4.0\n1700000010000\t1\t2.0\n",
    );
    logdir.write(
        "exp1/dvclive/plots/metrics/acc.tsv",
        "step\tacc\n0\t0.5\n1\t0.75\n",
    );

    let mut project = Project::new(logdir.path());
    let report = project.refresh().unwrap();
    assert_eq!(report.new_points, 4);

    let run = &project.runs["exp1"];
    assert_eq!(run.backend, Backend::Dvclive);
    assert_eq!(run.hparams["lr"], vertov_model::HparamValue::F64(0.001));
    assert_eq!(
        run.hparams["model.layers"],
        vertov_model::HparamValue::F64(4.0)
    );
    let loss = &run.series["train/loss"];
    assert_eq!(loss.class, SeriesClass::Scalar);
    assert_eq!(loss.summary.count(), 2);
    assert_eq!(loss.summary.min(), Some(2.0));
    assert_eq!(run.first_wall, Some(1.7e9));
    assert_eq!(run.series["acc"].summary.last().unwrap().value, 0.75);

    let points = project.materialize("exp1", "train/loss").unwrap().unwrap();
    assert_eq!(points.steps, vec![0, 1]);
    assert_eq!(points.values, vec![4.0, 2.0]);
    assert_eq!(points.walls, vec![1.7e9, 1.7e9 + 10.0]);
}

#[test]
fn dvclive_tails_with_torn_lines() {
    let logdir = Logdir::new("dvclive-tail");
    let tsv = "run/dvclive/plots/metrics/loss.tsv";
    logdir.write(tsv, "step\tloss\n0\t4.0\n");

    let mut project = Project::new(logdir.path());
    project.refresh().unwrap();
    project.materialize("run", "loss").unwrap().unwrap();

    // A complete row plus a torn one arrive.
    logdir.append(tsv, "1\t2.0\n2\t1.");
    let report = project.refresh().unwrap();
    assert_eq!(report.new_points, 1);
    assert_eq!(project.points("run", "loss").unwrap().steps, vec![0, 1]);

    // The torn row completes.
    logdir.append(tsv, "5\n");
    let report = project.refresh().unwrap();
    assert_eq!(report.new_points, 1);
    let points = project.points("run", "loss").unwrap();
    assert_eq!(points.steps, vec![0, 1, 2]);
    assert_eq!(points.values, vec![4.0, 2.0, 1.5]);
    assert_eq!(project.runs["run"].series["loss"].summary.count(), 3);
}

#[test]
fn dvclive_shrunk_file_resets_honestly() {
    let logdir = Logdir::new("dvclive-shrink");
    let tsv = "run/dvclive/plots/metrics/loss.tsv";
    logdir.write(tsv, "step\tloss\n0\t4.0\n1\t3.0\n2\t2.0\n");
    let mut project = Project::new(logdir.path());
    project.refresh().unwrap();
    assert_eq!(project.runs["run"].series["loss"].summary.count(), 3);

    // The experiment is re-run from scratch: the file is rewritten shorter.
    logdir.write(tsv, "step\tloss\n0\t9.0\n");
    project.refresh().unwrap();
    let summary = &project.runs["run"].series["loss"].summary;
    assert_eq!(summary.count(), 1);
    assert_eq!(summary.last().unwrap().value, 9.0);
}

#[test]
fn mlflow_run_end_to_end() {
    let logdir = Logdir::new("mlflow");
    logdir.write(
        "mlruns/0/a1b2/meta.yaml",
        "run_id: a1b2\nrun_name: brave-owl-7\nstatus: FINISHED\n",
    );
    logdir.write(
        "mlruns/0/a1b2/metrics/loss",
        "1700000000000 4.0 0\n1700000010000 2.0 1\n",
    );
    logdir.write("mlruns/0/a1b2/params/lr", "0.01");
    logdir.write("mlruns/0/a1b2/params/optimizer", "sgd");
    // The experiment-level meta.yaml must not become a run.
    logdir.write("mlruns/0/meta.yaml", "experiment_id: 0\nname: Default\n");

    let mut project = Project::new(logdir.path());
    let report = project.refresh().unwrap();
    assert_eq!(report.new_points, 2);
    assert_eq!(project.runs.len(), 1);

    let run = &project.runs["brave-owl-7"];
    assert_eq!(run.backend, Backend::Mlflow);
    assert_eq!(run.hparams["lr"], vertov_model::HparamValue::F64(0.01));
    assert_eq!(
        run.hparams["optimizer"],
        vertov_model::HparamValue::String("sgd".into())
    );
    assert_eq!(run.series["loss"].summary.count(), 2);

    let points = project.materialize("brave-owl-7", "loss").unwrap().unwrap();
    assert_eq!(points.steps, vec![0, 1]);
    assert_eq!(points.values, vec![4.0, 2.0]);

    // Live growth appends into the materialized series.
    logdir.append("mlruns/0/a1b2/metrics/loss", "1700000020000 1.0 2\n");
    let report = project.refresh().unwrap();
    assert_eq!(report.new_points, 1);
    assert_eq!(
        project.points("brave-owl-7", "loss").unwrap().values,
        vec![4.0, 2.0, 1.0]
    );
}

#[test]
fn mixed_root_carries_all_backends() {
    let logdir = Logdir::new("mixed");
    // A tfevents run.
    let events = tfevents::writer::events_file(&[tfevents::writer::scalar_event(
        1.7e9, 0, "loss", 1.0,
    )]);
    fs::create_dir_all(logdir.path().join("tf-run")).unwrap();
    fs::write(
        logdir.path().join("tf-run/events.out.tfevents.1000.host"),
        events,
    )
    .unwrap();
    // A dvclive run.
    logdir.write("dvc-run/dvclive/plots/metrics/loss.tsv", "step\tloss\n0\t2.0\n");
    // An MLflow run.
    logdir.write("mlruns/0/x/meta.yaml", "run_id: x\nrun_name: ml-run\n");
    logdir.write("mlruns/0/x/metrics/loss", "1700000000000 3.0 0\n");

    let mut project = Project::new(logdir.path());
    project.refresh().unwrap();
    let backends: Vec<(String, Backend)> = project
        .runs
        .iter()
        .map(|(name, run)| (name.clone(), run.backend))
        .collect();
    assert_eq!(
        backends,
        vec![
            ("dvc-run".to_owned(), Backend::Dvclive),
            ("ml-run".to_owned(), Backend::Mlflow),
            ("tf-run".to_owned(), Backend::Tfevents),
        ]
    );
    // Every backend materializes through the same call.
    for run in ["dvc-run", "ml-run", "tf-run"] {
        assert_eq!(
            project.materialize(run, "loss").unwrap().unwrap().len(),
            1,
            "{run}"
        );
    }
}

#[test]
fn real_dvclive_fixture_parses() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let mut project = Project::new(fixtures.join("dvclive"));
    project.refresh().unwrap();

    let run = &project.runs["exp"];
    assert_eq!(run.backend, Backend::Dvclive);
    assert_eq!(run.hparams["lr"], vertov_model::HparamValue::F64(0.005));
    assert_eq!(
        run.hparams["optimizer"],
        vertov_model::HparamValue::String("adamw".into())
    );
    for tag in ["train/loss", "train/accuracy"] {
        assert_eq!(run.series[tag].summary.count(), 12, "{tag}");
    }
    let points = project.materialize("exp", "train/loss").unwrap().unwrap();
    assert_eq!(points.steps, (0..12).collect::<Vec<i64>>());
    for (index, &value) in points.values.iter().enumerate() {
        let expected = 6.0 * (-0.4 * index as f64).exp() + 0.25;
        assert!(
            (value - expected).abs() < 1e-12,
            "step {index}: {value} vs {expected}"
        );
    }
}

#[test]
fn real_mlflow_fixture_parses() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let mut project = Project::new(fixtures.join("mlflow"));
    project.refresh().unwrap();

    assert_eq!(project.runs.len(), 1);
    let run = &project.runs["warm-start-7"];
    assert_eq!(run.backend, Backend::Mlflow);
    assert_eq!(run.hparams["lr"], vertov_model::HparamValue::F64(0.02));
    assert_eq!(
        run.hparams["optimizer"],
        vertov_model::HparamValue::String("sgd".into())
    );
    // Metric names with slashes land as nested files and read back whole.
    assert_eq!(run.series["val/accuracy"].summary.count(), 10);
    // Wall times came from the millisecond timestamps.
    assert!(run.first_wall.unwrap() > 1.7e9);

    let points = project.materialize("warm-start-7", "loss").unwrap().unwrap();
    assert_eq!(points.steps, (0..10).collect::<Vec<i64>>());
    for (index, &value) in points.values.iter().enumerate() {
        let expected = 3.0 * (-0.5 * index as f64).exp() + 0.1;
        assert!(
            (value - expected).abs() < 1e-12,
            "step {index}: {value} vs {expected}"
        );
    }
}

#[test]
fn wandb_run_end_to_end() {
    use vertov_formats::wandb::writer;
    let logdir = Logdir::new("wandb");
    let dir = logdir.path().join("offline-run-20260826_120000-abc123");
    fs::create_dir_all(&dir).unwrap();
    let wandb_path = dir.join("run-abc123.wandb");
    let mut file = writer::wandb_file(&[
        writer::config_record(&[("lr", "{\"value\": 0.001}"), ("optimizer", "\"adam\"")]),
        writer::history_record(0, 1.7e9, &[("loss", "4.0"), ("acc", "0.5")]),
        writer::history_record(1, 1.7e9 + 10.0, &[("loss", "NaN"), ("acc", "0.75")]),
    ]);
    fs::write(&wandb_path, &file).unwrap();

    let mut project = Project::new(logdir.path());
    let report = project.refresh().unwrap();
    assert_eq!(report.new_points, 4);
    let name = "offline-run-20260826_120000-abc123";
    let run = &project.runs[name];
    assert_eq!(run.backend, Backend::Wandb);
    assert_eq!(run.hparams["lr"], vertov_model::HparamValue::F64(0.001));
    assert_eq!(
        run.hparams["optimizer"],
        vertov_model::HparamValue::String("adam".into())
    );
    // No exit record yet: not finished; mtime says active.
    assert_eq!(run.finished, None);
    assert_eq!(run.series["loss"].summary.count(), 2);
    assert_eq!(run.series["loss"].summary.segments[0].non_finite, 1);
    assert_eq!(run.first_wall, Some(1.7e9));

    let points = project.materialize(name, "loss").unwrap().unwrap();
    assert_eq!(points.steps, vec![0, 1]);
    assert_eq!(points.values[0], 4.0);
    assert!(points.values[1].is_nan());

    // The trainer logs one more row and exits cleanly.
    writer::append_record(
        &mut file,
        &writer::history_record(2, 1.7e9 + 20.0, &[("loss", "2.0"), ("acc", "0.9")]),
    );
    writer::append_record(&mut file, &writer::exit_record());
    fs::write(&wandb_path, &file).unwrap();
    let report = project.refresh().unwrap();
    assert_eq!(report.new_points, 2);
    assert_eq!(project.points(name, "loss").unwrap().steps, vec![0, 1, 2]);
    let run = &project.runs[name];
    assert_eq!(run.finished, Some(true));
    assert_eq!(
        run.status(std::time::SystemTime::now(), std::time::Duration::from_secs(60)),
        vertov_model::RunStatus::Finished
    );
}

#[test]
fn wandb_newer_version_is_a_dead_file() {
    let logdir = Logdir::new("wandb-version");
    let dir = logdir.path().join("offline-run-1-x");
    fs::create_dir_all(&dir).unwrap();
    let mut file = vertov_formats::wandb::writer::wandb_file(&[]);
    file[6] = 9; // a future version byte
    fs::write(dir.join("run-x.wandb"), &file).unwrap();

    let mut project = Project::new(logdir.path());
    let report = project.refresh().unwrap();
    assert_eq!(report.dead_files, 1);
    assert_eq!(project.dead_files, 1);
    // The run exists (visible) but carries no data — fail loudly, keep
    // other backends working.
    assert!(project.runs["offline-run-1-x"].series.is_empty());
    // A dead file is reported once, not every tick.
    let report = project.refresh().unwrap();
    assert_eq!(report.dead_files, 0);
    assert_eq!(project.dead_files, 1);
}

#[test]
fn real_wandb_fixture_parses() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let mut project = Project::new(fixtures.join("wandb"));
    project.refresh().unwrap();

    assert_eq!(project.runs.len(), 1);
    let name = project.runs.keys().next().unwrap().clone();
    let run = &project.runs[&name];
    assert_eq!(run.backend, Backend::Wandb);
    assert_eq!(run.finished, Some(true), "the exit record marks a clean end");
    assert_eq!(run.hparams["lr"], vertov_model::HparamValue::F64(0.003));
    assert_eq!(
        run.hparams["optimizer"],
        vertov_model::HparamValue::String("adam".into())
    );
    assert_eq!(run.hparams["amsgrad"], vertov_model::HparamValue::Bool(true));
    assert_eq!(run.series["train/accuracy"].summary.count(), 10);

    let points = project.materialize(&name, "train/loss").unwrap().unwrap();
    assert_eq!(points.steps, (0..10).collect::<Vec<i64>>());
    for (index, &value) in points.values.iter().enumerate() {
        let expected = 5.0 * (-0.35 * index as f64).exp() + 0.2;
        assert!(
            (value - expected).abs() < 1e-12,
            "step {index}: {value} vs {expected}"
        );
    }
    assert!(project.dead_files == 0 && project.dropped_records == 0);
}

#[test]
fn vanished_dvclive_run_is_dropped() {
    let logdir = Logdir::new("dvclive-vanish");
    logdir.write("run/dvclive/plots/metrics/loss.tsv", "step\tloss\n0\t1.0\n");
    let mut project = Project::new(logdir.path());
    project.refresh().unwrap();
    assert_eq!(project.runs.len(), 1);
    fs::remove_dir_all(logdir.path().join("run")).unwrap();
    project.refresh().unwrap();
    assert!(project.runs.is_empty());
}
