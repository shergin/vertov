//! Golden tests: the binary's output, byte-for-byte, over a deterministic
//! generated logdir — the payoff of views being pure functions.
//!
//! Charts render through `Frame::detect` on a piped stdout: quadrant
//! charset, no color, 80×16 unless sized — deterministic by construction.
//! Every invocation passes `--no-cache` so tests never touch (or depend
//! on) the user's real summary cache.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tfevents::writer::{events_file, scalar_event};

struct Logdir(PathBuf);

impl Logdir {
    fn new(name: &str) -> Logdir {
        let dir = std::env::temp_dir().join(format!(
            "vertov-golden-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Logdir(dir)
    }

    fn write(&self, relative: &str, bytes: &[u8]) {
        let path = self.0.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
}

impl Drop for Logdir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn wall(step: i64) -> f64 {
    1.7e9 + step as f64 * 10.0
}

/// Two runs, two tags each, deterministic values; `sgd` has a restart.
fn standard_logdir(name: &str) -> Logdir {
    let logdir = Logdir::new(name);
    let adam: Vec<Vec<u8>> = (0..10)
        .flat_map(|step| {
            [
                scalar_event(wall(step), step, "train/loss", 8.0 / (step + 1) as f32),
                scalar_event(wall(step), step, "train/acc", 0.1 * step as f32),
            ]
        })
        .collect();
    logdir.write("adam/events.out.tfevents.1000.host", &events_file(&adam));

    let sgd_first: Vec<Vec<u8>> = (0..6)
        .map(|step| scalar_event(wall(step), step, "train/loss", 9.0 - step as f32))
        .collect();
    let sgd_second: Vec<Vec<u8>> = (4..8)
        .map(|step| scalar_event(wall(10 + step), step, "train/loss", 6.5 - step as f32))
        .collect();
    logdir.write(
        "sgd/events.out.tfevents.1000.host",
        &events_file(&sgd_first),
    );
    logdir.write(
        "sgd/events.out.tfevents.2000.host",
        &events_file(&sgd_second),
    );
    logdir
}

fn vertov(logdir: &Path, args: &[&str]) -> (String, String, bool) {
    let output = Command::new(env!("CARGO_BIN_EXE_vertov"))
        .arg(args[0])
        .arg(logdir)
        .args(&args[1..])
        .arg("--no-cache")
        .output()
        .expect("binary runs");
    (
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
        output.status.success(),
    )
}

#[test]
fn ls_text_golden() {
    let logdir = standard_logdir("ls-text");
    let (stdout, _, ok) = vertov(&logdir.0, &["ls"]);
    assert!(ok);
    assert_eq!(
        stdout,
        "run   status  series  points  restarts  step  duration\n\
         adam  active       2      20         0     9  90s\n\
         sgd   active       1      10         1     7  2m50s\n"
    );
}

#[test]
fn ls_json_golden() {
    let logdir = standard_logdir("ls-json");
    let (stdout, _, ok) = vertov(&logdir.0, &["ls", "--json"]);
    assert!(ok);
    assert_eq!(
        stdout,
        "[\n  {\"run\": \"adam\", \"status\": \"active\", \"series\": 2, \"points\": 20, \"restarts\": 0, \"step\": 9, \"duration\": \"90s\"},\n  {\"run\": \"sgd\", \"status\": \"active\", \"series\": 1, \"points\": 10, \"restarts\": 1, \"step\": 7, \"duration\": \"2m50s\"}\n]\n"
    );
}

#[test]
fn summary_text_golden() {
    let logdir = standard_logdir("summary");
    let (stdout, _, ok) = vertov(&logdir.0, &["summary", "sgd"]);
    assert!(ok);
    assert_eq!(
        stdout,
        "tag         class   count   min  max  mean  last  step  segments\n\
         train/loss  scalar     10  -0.5    9   4.3  -0.5     7         2\n"
    );
}

#[test]
fn export_csv_golden() {
    let logdir = standard_logdir("export");
    let (stdout, _, ok) = vertov(&logdir.0, &["export", "--csv"]);
    assert!(ok);
    assert_eq!(
        stdout,
        "run,train/acc,train/loss\n\
         adam,0.9000000357627869,0.800000011920929\n\
         sgd,,-0.5\n"
    );
}

#[test]
fn show_chart_golden() {
    let logdir = standard_logdir("show");
    let (stdout, _, ok) = vertov(
        &logdir.0,
        &["show", "-t", "train/loss", "--width", "60", "--height", "14"],
    );
    assert!(ok);
    // The exact frame is asserted structurally rather than byte-for-byte
    // here (glyph-level goldens live in malevich); what vertov owns is the
    // composition: both series in the legend, the restart rule, the title.
    assert!(stdout.contains("train/loss · 2 runs"), "title: {stdout}");
    assert!(stdout.contains("adam train/loss"), "legend: {stdout}");
    assert!(stdout.contains("sgd train/loss"), "legend: {stdout}");
    assert!(stdout.contains("restart"), "restart rule: {stdout}");
    let width = stdout.lines().map(|line| line.chars().count()).max().unwrap();
    assert!(width <= 60, "frame width {width} exceeds 60");
}

#[test]
fn tokens_axis_maps_and_reports_gaps() {
    let logdir = Logdir::new("tokens-axis");
    // `metered` logs a token counter covering steps 0..=6; its loss goes to
    // step 9, so three points sit outside coverage. `plain` has no counter.
    let mut metered: Vec<Vec<u8>> = (0..10)
        .map(|step| scalar_event(wall(step), step, "loss", 5.0 - 0.5 * step as f32))
        .collect();
    for step in [0, 2, 4, 6] {
        metered.push(scalar_event(wall(step), step, "tokens", step as f32 * 1024.0));
    }
    logdir.write(
        "metered/events.out.tfevents.1000.host",
        &events_file(&metered),
    );
    let plain: Vec<Vec<u8>> = (0..5)
        .map(|step| scalar_event(wall(step), step, "loss", 1.0))
        .collect();
    logdir.write("plain/events.out.tfevents.1000.host", &events_file(&plain));

    let (stdout, _, ok) = vertov(
        &logdir.0,
        &["show", "-t", "loss", "-x", "tokens", "--width", "70", "--height", "12"],
    );
    assert!(ok, "{stdout}");
    // The metered run draws (its counter reaches 6144 tokens); the plain
    // run is skipped loudly, and out-of-coverage points are counted.
    assert!(stdout.contains("1 runs lack a token counter"), "{stdout}");
    assert!(stdout.contains("3 pts outside counter"), "{stdout}");
    assert!(stdout.contains("metered loss"), "{stdout}");
    // Interpolated odd steps land between counter points: max x is 6144.
    assert!(stdout.contains("6"), "{stdout}");
}

#[test]
fn missing_tag_lists_alternatives() {
    let logdir = standard_logdir("missing-tag");
    let (_, stderr, ok) = vertov(&logdir.0, &["show", "-t", "nope"]);
    assert!(!ok);
    assert!(stderr.contains("no scalar tag matching `nope`"));
    assert!(stderr.contains("train/acc"));
    assert!(stderr.contains("train/loss"));
}

#[test]
fn summary_unknown_run_fails_with_suggestions() {
    let logdir = standard_logdir("unknown-run");
    let (_, stderr, ok) = vertov(&logdir.0, &["summary", "nadam"]);
    assert!(!ok);
    assert!(stderr.contains("no run `nadam`"));
    assert!(stderr.contains("adam"));
}

#[test]
fn runs_filter_limits_every_table() {
    let logdir = standard_logdir("runs-filter");
    let (stdout, _, ok) = vertov(&logdir.0, &["ls", "--runs", "sgd", "--csv"]);
    assert!(ok);
    assert_eq!(
        stdout,
        "run,status,series,points,restarts,step,duration\nsgd,active,1,10,1,7,2m50s\n"
    );
}
