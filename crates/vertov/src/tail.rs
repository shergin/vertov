//! `vertov tail`: live chart on stderr, repainted in place as the logdir
//! grows.
//!
//! Polling is the primary mechanism (training clusters live on NFS/Lustre,
//! which emit no notification events); the chart repaints only when data
//! actually changed, and staleness is written into the title rather than
//! left to the viewer's imagination. No alt-screen: the final frame stays in
//! scrollback after Ctrl-C, malevich's live-repaint discipline.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use malevich::Frame;
use malevich::stream::Live;
use vertov_model::Project;

use crate::chart::ChartData;
use crate::{Args, sized, title};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

pub fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    install_interrupt_handler();
    let mut project = Project::new(&args.logdir);
    if !args.no_cache {
        project.load_cache();
    }

    // Hide the cursor while repainting; restored below on every exit path.
    let mut cursor = io::stderr();
    let _ = write!(cursor, "\x1b[?25l");
    let _ = cursor.flush();
    let result = repaint(args, &mut project);
    let _ = write!(cursor, "\x1b[?25h");
    let _ = cursor.flush();
    if !args.no_cache {
        // Best-effort on the way out: the cache is an accelerator, never a
        // requirement.
        let _ = project.save_cache();
    }

    match result {
        // A closed terminal is a clean stop, not a failure.
        Err(err) if err.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        other => Ok(other?),
    }
}

fn repaint(args: &Args, project: &mut Project) -> io::Result<()> {
    let mut live = Live::new(io::stderr());
    let mut last_change = Instant::now();
    let mut drawn_once = false;
    loop {
        if INTERRUPTED.load(Ordering::SeqCst) {
            return Ok(());
        }
        let report = project.refresh()?;
        let changed = report.new_points > 0 || report.new_files > 0;
        if changed {
            last_change = Instant::now();
        }
        if changed || !drawn_once {
            let data = ChartData::collect(project, &args.tag, &crate::chart_options(args))?;
            // Re-detect every draw so a terminal resize follows along.
            let frame = sized(Frame::detect_for(&io::stderr()), args);
            let status = status_line(project, args, last_change);
            live.draw(
                &data.plot(&title(args, &data, project, Some(&status))),
                &frame,
            )?;
            drawn_once = true;
        }
        // Sleep in short slices so Ctrl-C answers promptly.
        let deadline = Instant::now() + args.interval;
        while Instant::now() < deadline {
            if INTERRUPTED.load(Ordering::SeqCst) {
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}

/// `live` while data flows; `stale Ns` once nothing has arrived for two poll
/// intervals — the viewer must say when it is showing old data.
fn status_line(project: &Project, args: &Args, last_change: Instant) -> String {
    let quiet = last_change.elapsed();
    if project.runs.is_empty() {
        "waiting for data".to_owned()
    } else if quiet > 2 * args.interval {
        format!("stale {}s", quiet.as_secs())
    } else {
        "live".to_owned()
    }
}

#[cfg(unix)]
fn install_interrupt_handler() {
    // A bare Ctrl-C would leave the terminal cursor hidden; catching SIGINT
    // lets the repaint loop restore it and leave the last frame in
    // scrollback. Double Ctrl-C still kills hard (the handler resets to
    // default after the first).
    extern "C" fn on_sigint(_: libc::c_int) {
        INTERRUPTED.store(true, Ordering::SeqCst);
        unsafe {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
        }
    }
    unsafe {
        libc::signal(libc::SIGINT, on_sigint as *const () as libc::sighandler_t);
    }
}

#[cfg(not(unix))]
fn install_interrupt_handler() {
    // Windows: Ctrl-C exits without restoring the cursor. Phase 4 brings a
    // proper console handler with the TUI work.
}
