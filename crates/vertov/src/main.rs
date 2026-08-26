//! vertov — a terminal viewer for ML training runs.
//!
//! The Phase 0 spike: `show` renders matching scalar series once, `tail`
//! live-plots them with in-place repaint. Data comes from tfevents files the
//! trainer already writes; vertov only ever reads.

mod chart;
mod tail;

use std::process::ExitCode;
use std::time::Duration;

use malevich::Frame;
use vertov_model::{Project, SeriesClass};

use chart::ChartData;

const HELP: &str = "\
vertov — a terminal viewer for ML training runs

Usage:
  vertov show <logdir> -t <tag> [--width N] [--height N]
  vertov tail <logdir> -t <tag> [--interval SECS] [--width N] [--height N]

Commands:
  show   Render matching scalar series to stdout, once.
  tail   Live chart on stderr, repainted in place as the logdir grows.
         Ctrl-C stops; the final frame stays in your scrollback.

Options:
  -t, --tag <TAG>       Tag filter: matches any scalar tag containing TAG.
      --interval <SECS> Poll interval for tail (default 5; NFS-friendly polling,
                        no inotify required).
      --width <N>       Frame width in cells (default: detected).
      --height <N>      Frame height in cells (default: detected).
  -h, --help            This help.

Examples:
  vertov show runs/ -t loss
  vertov tail runs/ -t 'train/loss' --interval 2
";

struct Args {
    command: Command,
    logdir: String,
    tag: String,
    interval: Duration,
    width: Option<usize>,
    height: Option<usize>,
}

enum Command {
    Show,
    Tail,
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
    let result = match args.command {
        Command::Show => show(&args),
        Command::Tail => tail::run(&args),
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

    let mut command = None;
    let mut logdir = None;
    let mut tag = None;
    let mut interval = Duration::from_secs(5);
    let mut width = None;
    let mut height = None;

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Value(value) if command.is_none() => {
                command = Some(match value.to_string_lossy().as_ref() {
                    "show" => Command::Show,
                    "tail" => Command::Tail,
                    other => {
                        return Err(format!("unknown command `{other}`").into());
                    }
                });
            }
            Value(value) if logdir.is_none() => {
                logdir = Some(value.string()?);
            }
            Value(value) => {
                return Err(format!("unexpected argument `{}`", value.to_string_lossy()).into());
            }
            Short('t') | Long("tag") => tag = Some(parser.value()?.string()?),
            Long("interval") => {
                let seconds: f64 = parser.value()?.parse()?;
                if !seconds.is_finite() || seconds <= 0.0 {
                    return Err("--interval must be positive".to_owned().into());
                }
                interval = Duration::from_secs_f64(seconds);
            }
            Long("width") => width = Some(parser.value()?.parse()?),
            Long("height") => height = Some(parser.value()?.parse()?),
            Short('h') | Long("help") => return Ok(None),
            _ => return Err(arg.unexpected()),
        }
    }

    let Some(command) = command else {
        return Ok(None);
    };
    let logdir = logdir.ok_or("missing <logdir>")?;
    let tag = tag.ok_or("missing -t <tag> (vertov never renders all tags unasked)")?;
    Ok(Some(Args {
        command,
        logdir,
        tag,
        interval,
        width,
        height,
    }))
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
    let mut project = Project::new(&args.logdir);
    project.refresh()?;
    let data = ChartData::collect(&mut project, &args.tag)?;
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
