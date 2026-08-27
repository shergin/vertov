//! Series: per-tag summaries, restart segments, and materialized points.
//!
//! Every series carries an exact summary at all times, structured as one
//! accumulator per restart segment. Preemption follows RustBoard: a new
//! point at `step <= tail` truncates that series' tail — never other tags —
//! and opens a new segment. Summaries cannot un-accumulate a truncated tail
//! (that would need the points), so a preempted segment keeps its as-written
//! totals and records the preemption step; [`SeriesSummary::preempted`]
//! says when merged values include ghost data, and materialization gives
//! the exact live view on demand.

use crate::moments::Moments;

/// What kind of data a series carries, from its payloads and plugin
/// metadata.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum SeriesClass {
    /// Scalar points (TF1 `simple_value` or TF2 rank-0 tensors).
    Scalar,
    /// Histograms (TF1 proto or TF2 `[k, 3]` tensors).
    Histogram,
    /// Encoded images.
    Image,
    /// Text summaries.
    Text,
    /// Payloads this model does not classify.
    Unknown,
}

/// One observed point: step, wall time, and value (NaN for non-scalar
/// series, where only step/wall/count are meaningful).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PointStamp {
    /// Global step.
    pub step: i64,
    /// Seconds since the Unix epoch.
    pub wall: f64,
    /// The scalar value; NaN both for gaps and for non-scalar series.
    pub value: f64,
}

/// Exact accumulators over one restart segment of a series.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SegmentSummary {
    /// First point of the segment.
    pub first: PointStamp,
    /// Last point written to the segment (including any later-preempted
    /// tail — see [`SegmentSummary::preempted_at`]).
    pub last: PointStamp,
    /// Minimum finite value, NaN if none.
    pub min: f64,
    /// Maximum finite value, NaN if none.
    pub max: f64,
    /// Mean/variance over finite values.
    pub moments: Moments,
    /// Total points written, finite or not.
    pub count: u64,
    /// Points with non-finite values (NaN is a gap, never interpolated).
    pub non_finite: u64,
    /// Set when a restart preempted this segment: the step at which the
    /// following segment began. Points at or beyond it are ghost data, and
    /// this segment's accumulators still include them (exact truncation
    /// would need the points; materialize for the live view).
    pub preempted_at: Option<i64>,
}

impl SegmentSummary {
    fn new(point: PointStamp) -> SegmentSummary {
        let mut segment = SegmentSummary {
            first: point,
            last: point,
            min: f64::NAN,
            max: f64::NAN,
            moments: Moments::default(),
            count: 0,
            non_finite: 0,
            preempted_at: None,
        };
        segment.accumulate(point);
        segment
    }

    fn accumulate(&mut self, point: PointStamp) {
        self.last = point;
        self.count += 1;
        if point.value.is_finite() {
            // f64::min/max return the other operand for NaN, so the NaN
            // initial state seeds itself.
            self.min = self.min.min(point.value);
            self.max = self.max.max(point.value);
            self.moments.push(point.value);
        } else {
            self.non_finite += 1;
        }
    }
}

/// The always-current summary of a series: one exact accumulator per
/// restart segment, merged on demand.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct SeriesSummary {
    /// The segments, in write order. Never empty once a point has been
    /// observed.
    pub segments: Vec<SegmentSummary>,
}

impl SeriesSummary {
    /// Feeds one point. Returns `true` when the point preempted the series
    /// (opened a new segment by rewriting a step at or before the tail).
    pub fn observe(&mut self, point: PointStamp) -> bool {
        match self.segments.last_mut() {
            None => {
                self.segments.push(SegmentSummary::new(point));
                false
            }
            Some(tail) if point.step <= tail.last.step => {
                tail.preempted_at = Some(point.step);
                self.segments.push(SegmentSummary::new(point));
                true
            }
            Some(tail) => {
                tail.accumulate(point);
                false
            }
        }
    }

    /// Total points written across all segments (ghost tails included).
    pub fn count(&self) -> u64 {
        self.segments.iter().map(|segment| segment.count).sum()
    }

    /// First point ever written.
    pub fn first(&self) -> Option<PointStamp> {
        self.segments.first().map(|segment| segment.first)
    }

    /// Most recently written live point.
    pub fn last(&self) -> Option<PointStamp> {
        self.segments.last().map(|segment| segment.last)
    }

    /// Minimum finite value as written, NaN-free `None` when no finite
    /// values exist. Includes ghost tails when [`preempted`](Self::preempted).
    pub fn min(&self) -> Option<f64> {
        self.segments
            .iter()
            .map(|segment| segment.min)
            .filter(|min| !min.is_nan())
            .reduce(f64::min)
    }

    /// Maximum finite value as written; see [`min`](Self::min).
    pub fn max(&self) -> Option<f64> {
        self.segments
            .iter()
            .map(|segment| segment.max)
            .filter(|max| !max.is_nan())
            .reduce(f64::max)
    }

    /// Mean/variance over all finite values as written.
    pub fn moments(&self) -> Moments {
        let mut merged = Moments::default();
        for segment in &self.segments {
            merged.merge(&segment.moments);
        }
        merged
    }

    /// True when any segment was preempted: merged aggregates then include
    /// ghost tails, and the exact live view requires materialization.
    pub fn preempted(&self) -> bool {
        self.segments
            .iter()
            .any(|segment| segment.preempted_at.is_some())
    }
}

/// One series in the catalog: classification and its summary. Points live
/// separately, materialized on demand.
#[derive(Clone, PartialEq, Debug)]
pub struct Series {
    /// What the series carries.
    pub class: SeriesClass,
    /// Owning plugin, from the first point's metadata when present.
    pub plugin: Option<String>,
    /// The always-current summary.
    pub summary: SeriesSummary,
}

/// A preempted tail, kept renderable: data honesty over tidiness.
#[derive(Clone, PartialEq, Debug)]
pub struct Ghost {
    /// Index into the live columns where the truncation happened — the
    /// ghost's points originally followed `steps[..at]`.
    pub at: usize,
    /// Steps of the truncated points.
    pub steps: Vec<i64>,
    /// Wall times of the truncated points.
    pub walls: Vec<f64>,
    /// Values of the truncated points.
    pub values: Vec<f64>,
}

/// Materialized full-fidelity points of one series: parallel columns plus
/// segment boundaries and preempted ghost tails.
///
/// Invariant: live `steps` are strictly increasing (preemption enforces it).
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Points {
    /// Steps, strictly increasing.
    pub steps: Vec<i64>,
    /// Wall times, parallel to `steps`.
    pub walls: Vec<f64>,
    /// Values, parallel to `steps`; NaN is a gap.
    pub values: Vec<f64>,
    /// Indices where a new restart segment begins (index 0 is never listed).
    pub boundaries: Vec<usize>,
    /// Preempted tails, in preemption order.
    pub ghosts: Vec<Ghost>,
}

impl Points {
    /// Appends one point, applying preemption: a step at or before the tail
    /// truncates the tail into a [`Ghost`] and records a boundary.
    pub fn push(&mut self, point: PointStamp) {
        if self.steps.last().is_some_and(|&tail| point.step <= tail) {
            let cut = self.steps.partition_point(|&step| step < point.step);
            self.ghosts.push(Ghost {
                at: cut,
                steps: self.steps.split_off(cut),
                walls: self.walls.split_off(cut),
                values: self.values.split_off(cut),
            });
            self.boundaries.push(cut);
        }
        self.steps.push(point.step);
        self.walls.push(point.wall);
        self.values.push(point.value);
    }

    /// Number of live points.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// True when no live points exist.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// One histogram observation: normalized `(left, right, count)` buckets at
/// a step.
#[derive(Clone, PartialEq, Debug)]
pub struct HistogramSnapshot {
    /// Global step.
    pub step: i64,
    /// Seconds since the Unix epoch.
    pub wall: f64,
    /// Contiguous buckets, left to right.
    pub buckets: Vec<(f64, f64, f64)>,
}

/// Materialized histogram series: snapshots in step order, with restart
/// boundaries. Preempted tails are truncated (unlike scalar [`Points`], no
/// ghosts are kept — a re-read restores anything a view later wants).
#[derive(Clone, PartialEq, Debug, Default)]
pub struct HistogramSeries {
    /// Snapshots, steps strictly increasing.
    pub snapshots: Vec<HistogramSnapshot>,
    /// Indices where a new restart segment begins.
    pub boundaries: Vec<usize>,
}

impl HistogramSeries {
    /// Appends one snapshot, applying step preemption.
    pub fn push(&mut self, snapshot: HistogramSnapshot) {
        if self
            .snapshots
            .last()
            .is_some_and(|tail| snapshot.step <= tail.step)
        {
            let cut = self
                .snapshots
                .partition_point(|held| held.step < snapshot.step);
            self.snapshots.truncate(cut);
            self.boundaries.push(cut);
        }
        self.snapshots.push(snapshot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(step: i64, value: f64) -> PointStamp {
        PointStamp {
            step,
            wall: 1000.0 + step as f64,
            value,
        }
    }

    #[test]
    fn summary_accumulates_exactly() {
        let mut summary = SeriesSummary::default();
        for (step, value) in [(0, 4.0), (1, 2.0), (2, f64::NAN), (3, 8.0)] {
            assert!(!summary.observe(point(step, value)));
        }
        assert_eq!(summary.count(), 4);
        assert_eq!(summary.min(), Some(2.0));
        assert_eq!(summary.max(), Some(8.0));
        assert_eq!(summary.moments().count(), 3);
        assert!((summary.moments().mean().unwrap() - 14.0 / 3.0).abs() < 1e-12);
        assert_eq!(summary.first().unwrap().step, 0);
        assert_eq!(summary.last().unwrap().step, 3);
        assert_eq!(summary.segments.len(), 1);
        assert!(!summary.preempted());
        assert_eq!(summary.segments[0].non_finite, 1);
    }

    #[test]
    fn summary_preemption_opens_segment() {
        let mut summary = SeriesSummary::default();
        for step in 0..5 {
            summary.observe(point(step, step as f64));
        }
        // Restart resumes from step 3.
        assert!(summary.observe(point(3, 30.0)));
        assert!(!summary.observe(point(4, 40.0)));
        assert_eq!(summary.segments.len(), 2);
        assert_eq!(summary.segments[0].preempted_at, Some(3));
        assert!(summary.preempted());
        assert_eq!(summary.last().unwrap().value, 40.0);
        // As-written totals include the ghost tail — flagged by preempted().
        assert_eq!(summary.count(), 7);
    }

    #[test]
    fn equal_step_is_a_preemption() {
        let mut summary = SeriesSummary::default();
        summary.observe(point(5, 1.0));
        assert!(summary.observe(point(5, 2.0)));
        assert_eq!(summary.segments.len(), 2);
    }

    #[test]
    fn points_preemption_truncates_into_ghost() {
        let mut points = Points::default();
        for step in 0..5 {
            points.push(point(step, step as f64));
        }
        points.push(point(3, 30.0));
        points.push(point(4, 40.0));

        assert_eq!(points.steps, vec![0, 1, 2, 3, 4]);
        assert_eq!(points.values, vec![0.0, 1.0, 2.0, 30.0, 40.0]);
        assert_eq!(points.boundaries, vec![3]);
        assert_eq!(points.ghosts.len(), 1);
        assert_eq!(points.ghosts[0].at, 3);
        assert_eq!(points.ghosts[0].steps, vec![3, 4]);
        assert_eq!(points.ghosts[0].values, vec![3.0, 4.0]);
    }

    #[test]
    fn points_multiple_preemptions() {
        let mut points = Points::default();
        for step in [0, 1, 2, 1, 2, 3, 0, 1] {
            points.push(point(step, step as f64 * 10.0));
        }
        assert_eq!(points.steps, vec![0, 1]);
        assert_eq!(points.boundaries, vec![1, 0]);
        assert_eq!(points.ghosts.len(), 2);
        assert_eq!(points.ghosts[0].steps, vec![1, 2]);
        assert_eq!(points.ghosts[1].steps, vec![0, 1, 2, 3]);
    }
}
