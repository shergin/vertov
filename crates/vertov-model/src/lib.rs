//! The unified data model behind vertov: a catalog of runs and series with
//! exact, mergeable summaries for *every* series, restart segments as
//! first-class data, and transient full-fidelity points for the series
//! actually being looked at.
//!
//! The files are the database: nothing here owns a copy of the data beyond
//! cheap summaries and resume offsets. Everything holds three commitments —
//! summaries are exact accumulators (never samples), a rewritten step
//! truncates only its own series' tail (RustBoard's preemption semantics)
//! with the pre-restart tail kept as a renderable ghost, and a torn file
//! tail is a state to resume from, not an error.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod moments;
pub mod project;
pub mod series;

pub use moments::Moments;
pub use project::{Project, RefreshReport, Run, RunStatus};
pub use series::{Ghost, PointStamp, Points, SegmentSummary, Series, SeriesClass, SeriesSummary};
