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

use crate::chart::ChartData;
use crate::logdir::Watcher;
use crate::{Args, sized, title};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

pub fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    install_interrupt_handler();
    let mut watcher = Watcher::new(&args.logdir, &args.tag);

    // Hide the cursor while repainting; restored below on every exit path.
    let mut cursor = io::stderr();
    let _ = write!(cursor, "\x1b[?25l");
    let _ = cursor.flush();
    let result = repaint(args, &mut watcher);
    let _ = write!(cursor, "\x1b[?25h");
    let _ = cursor.flush();

    match result {
        // A closed terminal is a clean stop, not a failure.
        Err(err) if err.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        other => Ok(other?),
    }
}

fn repaint(args: &Args, watcher: &mut Watcher) -> io::Result<()> {
    let mut live = Live::new(io::stderr());
    let mut last_change = Instant::now();
    let mut drawn_once = false;
    loop {
        if INTERRUPTED.load(Ordering::SeqCst) {
            return Ok(());
        }
        let appended = watcher.poll()?;
        if appended > 0 {
            last_change = Instant::now();
        }
        if appended > 0 || !drawn_once {
            let data = ChartData::collect(watcher);
            // Re-detect every draw so a terminal resize follows along.
            let frame = sized(Frame::detect_for(&io::stderr()), args);
            let status = status_line(watcher, args, last_change);
            live.draw(&data.plot(&title(args, &data, watcher, Some(&status))), &frame)?;
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
fn status_line(watcher: &Watcher, args: &Args, last_change: Instant) -> String {
    let quiet = last_change.elapsed();
    if watcher.runs.is_empty() {
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
