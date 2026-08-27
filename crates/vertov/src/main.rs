//! vertov — a terminal viewer for ML training runs.
//!
//! Headless commands (kaz's philosophy: data on stdout, zero rendering
//! logic outside the library): `show` renders matching scalar series once,
//! `tail` live-plots them, `ls`/`summary`/`export` print tables in text,
//! CSV, or JSON. Data comes from the files trainers already write; vertov
//! only ever reads.

mod chart;
mod export;
mod ls;
mod summary;
mod table;
mod tail;
mod tui;

use std::process::ExitCode;
use std::time::Duration;

use malevich::Frame;
use vertov_model::{Project, Run, RunStatus, SeriesClass};

use chart::{ChartData, ChartOptions, XAxis};
use table::Format;

const HELP: &str = "\
vertov — a terminal viewer for ML training runs

Usage:
  vertov <logdir>                  # the TUI: runs table, scalars, live tail
  vertov show <logdir> -t <tag> [chart options]
  vertov tail <logdir> -t <tag> [--interval SECS] [chart options]
  vertov ls <logdir> [--runs SUBSTR] [--csv | --json]
  vertov summary <logdir> <run> [--csv | --json]
  vertov export <logdir> [--runs SUBSTR] [--csv | --json]

Commands:
  (none)   A logdir alone opens the TUI. `?` inside lists the keys.
  show     Render matching scalar series to stdout, once.
  tail     Live chart on stderr, repainted in place as the logdir grows.
           Ctrl-C stops; the final frame stays in your scrollback.
  ls       Runs table: status, series, points, restarts, last step, duration.
  summary  One run, every series: exact accumulators, never samples.
  export   Flat runs × (params + metrics) table; last value per scalar tag.

Chart options (show, tail):
  --smooth <F>      EWMA smoothing factor in [0,1) — smoothed line over the
                    faded raw one, TensorBoard's exact debiasing.
  -x, --x <AXIS>    X axis: step (default), wall, relative.
  --runs <FILTER>   A predicate over hparams, metrics, status, and name
                    ('lr > 1e-3 and status == active'), or — when the text
                    is not a predicate — a substring of the run name.
                    Also accepted by ls and export.
      --width <N>   Frame width in cells (default: detected).
      --height <N>  Frame height in cells (default: detected).

Options:
  -t, --tag <TAG>       Tag filter: matches any scalar tag containing TAG.
      --interval <SECS> Poll interval for tail (default 5; NFS-friendly
                        polling, no inotify required).
      --csv, --json     Table output for ls/summary/export (default: text).
      --no-cache        Skip the summary cache (~/.cache/vertov/); always
                        re-read from the logdir. The cache is disposable —
                        deleting it by hand is always safe too.
      --pixels <MODE>   auto (default): TUI charts render as real images
                        where the terminal speaks sixel/kitty/iTerm2, cell
                        glyphs elsewhere. never: always cell glyphs.
  -h, --help            This help.

Examples:
  vertov show runs/ -t loss --smooth 0.97
  vertov tail runs/ -t 'train/loss' --interval 2
  vertov export runs/ --csv > runs.csv
";

struct Args {
    command: Command,
    logdir: String,
    tag: String,
    runs_filter: Option<String>,
    format: Format,
    smooth: Option<f64>,
    x_axis: XAxis,
    interval: Duration,
    width: Option<usize>,
    height: Option<usize>,
    no_cache: bool,
    no_pixels: bool,
}

enum Command {
    Tui,
    Show,
    Tail,
    Ls,
    Summary { run: String },
    Export,
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(Some(args)) => args,
        Ok(None) => {
            print!("{HELP}");
            return ExitCode::SUCCESS;
        }
        Err(err) => {
            eprintln!("vertov: {err}");
            eprintln!("Run `vertov --help` for usage.");
            return ExitCode::FAILURE;
        }
    };
    let result = match &args.command {
        Command::Tui => tui::run(&args),
        Command::Show => show(&args),
        Command::Tail => tail::run(&args),
        Command::Ls => ls::run(&args),
        Command::Summary { run } => summary::run(&args, &run.clone()),
        Command::Export => export::run(&args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("vertov: {err}");
            ExitCode::FAILURE
        }
    }
}

fn parse_args() -> Result<Option<Args>, lexopt::Error> {
    use lexopt::prelude::*;

    let mut command_name: Option<String> = None;
    let mut positionals: Vec<String> = Vec::new();
    let mut tag = None;
    let mut runs_filter = None;
    let mut format = Format::Text;
    let mut smooth = None;
    let mut x_axis = XAxis::Step;
    let mut interval = Duration::from_secs(5);
    let mut width = None;
    let mut height = None;
    let mut no_cache = false;
    let mut no_pixels = false;

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Value(value) if command_name.is_none() => {
                command_name = Some(value.string()?);
            }
            Value(value) => positionals.push(value.string()?),
            Short('t') | Long("tag") => tag = Some(parser.value()?.string()?),
            Long("runs") => runs_filter = Some(parser.value()?.string()?),
            Long("csv") => format = Format::Csv,
            Long("json") => format = Format::Json,
            Long("smooth") => {
                let factor: f64 = parser.value()?.parse()?;
                if !(0.0..1.0).contains(&factor) {
                    return Err("--smooth must be in [0, 1)".to_owned().into());
                }
                smooth = Some(factor);
            }
            Short('x') | Long("x") => {
                x_axis = match parser.value()?.string()?.as_str() {
                    "step" => XAxis::Step,
                    "wall" => XAxis::Wall,
                    "relative" => XAxis::Relative,
                    other => {
                        return Err(
                            format!("unknown x axis `{other}` (step|wall|relative)").into()
                        );
                    }
                };
            }
            Long("interval") => {
                let seconds: f64 = parser.value()?.parse()?;
                if !seconds.is_finite() || seconds <= 0.0 {
                    return Err("--interval must be positive".to_owned().into());
                }
                interval = Duration::from_secs_f64(seconds);
            }
            Long("width") => width = Some(parser.value()?.parse()?),
            Long("height") => height = Some(parser.value()?.parse()?),
            Long("no-cache") => no_cache = true,
            Long("pixels") => {
                no_pixels = match parser.value()?.string()?.as_str() {
                    "auto" => false,
                    "never" => true,
                    other => {
                        return Err(format!("unknown pixels mode `{other}` (auto|never)").into());
                    }
                };
            }
            Short('h') | Long("help") => return Ok(None),
            _ => return Err(arg.unexpected()),
        }
    }

    let Some(command_name) = command_name else {
        return Ok(None);
    };
    let mut positionals = positionals.into_iter();
    // A bare path (not a command name) opens the TUI on it: `vertov runs/`.
    let known_command = matches!(
        command_name.as_str(),
        "show" | "tail" | "ls" | "summary" | "export"
    );
    let logdir = if known_command {
        positionals.next().ok_or("missing <logdir>")?
    } else {
        command_name.clone()
    };
    let command = match command_name.as_str() {
        "show" => Command::Show,
        "tail" => Command::Tail,
        "ls" => Command::Ls,
        "summary" => Command::Summary {
            run: positionals.next().ok_or("missing <run> (see `vertov ls`)")?,
        },
        "export" => Command::Export,
        _ => Command::Tui,
    };
    if let Some(extra) = positionals.next() {
        return Err(format!("unexpected argument `{extra}`").into());
    }
    let tag = match command {
        Command::Show | Command::Tail => {
            tag.ok_or("missing -t <tag> (vertov never renders all tags unasked)")?
        }
        _ => tag.unwrap_or_default(),
    };
    Ok(Some(Args {
        command,
        logdir,
        tag,
        runs_filter,
        format,
        smooth,
        x_axis,
        interval,
        width,
        height,
        no_cache,
        no_pixels,
    }))
}

/// Opens the project, warms it from the summary cache unless `--no-cache`,
/// refreshes once, and saves the cache back (best-effort — the cache is an
/// accelerator, never a requirement).
fn load_project(args: &Args) -> std::io::Result<Project> {
    let mut project = Project::new(&args.logdir);
    if !args.no_cache {
        project.load_cache();
    }
    project.refresh()?;
    if !args.no_cache {
        let _ = project.save_cache();
    }
    Ok(project)
}

/// `--runs` and the TUI filter accept either a predicate
/// (`lr > 1e-3 and status == active`) or, when the text does not parse as
/// one, a plain substring of the run name. Callers parse once and pass the
/// result down.
fn run_passes(
    filter: Option<&str>,
    predicate: Option<&vertov_model::Predicate>,
    name: &str,
    status: &str,
    run: &Run,
) -> bool {
    match (filter, predicate) {
        (None, _) => true,
        (Some(_), Some(predicate)) => predicate.matches(name, status, run),
        (Some(filter), None) => name.contains(filter),
    }
}

fn status_text(run: &Run, now: std::time::SystemTime, window: Duration) -> &'static str {
    match run.status(now, window) {
        RunStatus::Active => "active",
        RunStatus::Idle => "idle",
        RunStatus::Unknown => "?",
    }
}

fn hparam_text(value: &vertov_model::HparamValue) -> String {
    match value {
        vertov_model::HparamValue::F64(v) => v.to_string(),
        vertov_model::HparamValue::String(v) => v.clone(),
        vertov_model::HparamValue::Bool(v) => v.to_string(),
    }
}

fn chart_options(args: &Args) -> ChartOptions {
    ChartOptions {
        x_axis: args.x_axis,
        smooth: args.smooth,
        runs_filter: args.runs_filter.clone(),
        log_y: false,
    }
}

fn sized(mut frame: Frame, args: &Args) -> Frame {
    if let Some(width) = args.width {
        frame.width = width;
    }
    if let Some(height) = args.height {
        frame.height = height;
    }
    frame
}

/// A one-line summary of what the chart shows and what it had to drop —
/// staleness and loss are displayed, never hidden.
fn title(args: &Args, data: &ChartData, project: &Project, live: Option<&str>) -> String {
    use std::fmt::Write as _;
    let mut title = args.tag.clone();
    if data.run_count > 1 {
        let _ = write!(title, " · {} runs", data.run_count);
    }
    if data.cut > 0 {
        let _ = write!(title, " · +{} series not shown", data.cut);
    }
    if project.dropped_records > 0 {
        let _ = write!(title, " · {} records lost", project.dropped_records);
    }
    if project.dead_files > 0 {
        let _ = write!(title, " · {} dead files", project.dead_files);
    }
    if let Some(live) = live {
        let _ = write!(title, " · {live}");
    }
    title
}

fn show(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut project = load_project(args)?;
    let data = ChartData::collect(&mut project, &args.tag, &chart_options(args))?;
    if data.is_empty() {
        return Err(no_match_message(args, &project).into());
    }
    let frame = sized(Frame::detect(), args);
    print!(
        "{}",
        data.plot(&title(args, &data, &project, None)).render(&frame)
    );
    Ok(())
}

fn no_match_message(args: &Args, project: &Project) -> String {
    use std::fmt::Write as _;
    let seen: std::collections::BTreeSet<&String> = project
        .runs
        .values()
        .flat_map(|run| {
            run.series
                .iter()
                .filter(|(_, series)| series.class == SeriesClass::Scalar)
                .map(|(tag, _)| tag)
        })
        .collect();
    let mut message = format!("no scalar tag matching `{}` in {}", args.tag, args.logdir);
    if !seen.is_empty() {
        let _ = write!(message, "\nscalar tags found:");
        for tag in seen.iter().take(20) {
            let _ = write!(message, "\n  {tag}");
        }
        if seen.len() > 20 {
            let _ = write!(message, "\n  … and {} more", seen.len() - 20);
        }
    }
    message
}
