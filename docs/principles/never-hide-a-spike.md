# Never hide a spike

What renders is pixel-identical to plotting every point. What
summarizes is exact. Nothing between the data and the eye is
lossy.

## Why

The whole job of babysitting a run is noticing the anomaly early:
the one-step loss spike, the gradient-norm excursion before the
divergence. Those events are exactly what uniform or reservoir
downsampling deletes — TensorBoard's 1000-point reservoir will,
with high probability, simply not contain the spike. A dashboard
that is blind precisely when something goes wrong teaches its
users to distrust it; lossless-at-any-zoom was a selling point
competitors built entire products on.

Smoothing has the same failure mode socially: a smoothed-only
curve looks calm because it is hiding the evidence.

## The idea

Downsampling is a rendering concern with an exact solution: M4 —
min, max, first, last per raster column, bucketed by the column
each point lands in — provably produces the same pixels as
drawing every point. So the data model never samples; the
renderer reduces, losslessly, at whatever resolution the frame
has. Zoom re-materializes from the file, so the guarantee holds
at every scale.

Tables get the same treatment from the other side: min/max/mean/
last are exact accumulators over every point ever seen, so the
extreme in the runs table is the true extreme, not the extreme of
a sample.

Smoothing is a labeled overlay: the debiased EWMA drawn over the
raw line, which stays visible behind it. The reader always sees
both the story and the evidence.

## Consequences

- No fidelity knob exists, because there is nothing to trade:
  full fidelity is the only mode.
- A spike one step wide survives every zoom level and every
  overlay.
- The summary table can be trusted for triage; its extremes are
  real.
- Displaying the smoothed line alone is not expressible.

## Not this

- Reservoir or uniform sampling at ingest.
- LTTB or other perceptual downsamplers on the truth path — fine
  for thumbnails elsewhere, not for the chart being read.
- Silently dropping points to fit a memory budget.
- Smoothing defaults that hide the raw line.

See [Vision](../vision.md) rule 4,
[The files are the database](files-are-the-database.md) for why
nothing forces sampling, and
[Honest under failure](honest-under-failure.md) for the same
stance toward broken data.

## Spelled today

malevich's M4 reduction (pixel-identical, benchmarked at ten
million points) and `stat::ewma` (TensorBoard's debiased
smoothing) are the primitives; exact summaries ride `Moments`.
Fidelity requirements are argued in [plan.md](../plan.md) §3.4
and §5.5. This section may rot; the rest must not.
