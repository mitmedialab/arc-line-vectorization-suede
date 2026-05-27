# Arc-line-vectorization-suede

This repo is a [suede dependency](https://github.com/pmalacho-mit/suede). 

To see the installable source code, please checkout the [release branch](https://github.com/mitmedialab/arc_line_vectorization_suede/tree/release).

## Installation

```bash
bash <(curl https://suede.sh/install-release) --repo mitmedialab/arc_line_vectorization_suede
```

<details>
<summary>
See alternative to using <a href="https://github.com/pmalacho-mit/suede#suedesh">suede.sh</a> script proxy
</summary>

```bash
bash <(curl https://raw.githubusercontent.com/pmalacho-mit/suede/refs/heads/main/scripts/install-release.sh) --repo mitmedialab/arc_line_vectorization_suede
```

</details>

Turn a hand-drawn PNG sketch into a sequence of motion commands a
differential-drive drawing robot can execute — and make that sequence
draw the picture *quickly*.

This README is the onboarding guide. It explains what the system does,
the robot model that shapes every design decision, the pipeline stages
end to end, and the math behind the parts that aren't obvious. By the
end you should be able to find your way around the code and understand
*why* it is shaped the way it is.

---

## 1. The problem, and why it is interesting

Input: a raster image of a line drawing (the test set is 31 sketches,
all 1024×1024). Output: an ordered list of robot commands that, when
executed, reproduce the drawing.

The robot can do exactly three things (`release/commands.py`):

- `line`  — drive straight a given distance, pen down or up.
- `spin`  — rotate in place by a signed angle (pen contributes nothing).
- `arc`   — drive a circular arc of a given radius through a signed
  sweep angle.

Two facts about this robot drive the entire design:

1. **Spinning in place is pure overhead.** It moves the pen nowhere.
   Every time the drawn path has a corner, the robot must stop and
   spin. Every separate stroke begins with a spin to face the right
   way. So a curve drawn as one `arc` is far cheaper than the same
   curve approximated by a chain of short `line` segments — the
   polyline pays one spin per vertex.

2. **Pen-up travel is overhead too.** Lifting the pen to jump from the
   end of one stroke to the start of another draws nothing but still
   costs driving and spinning time.

The objective is therefore a tension between two things that pull in
opposite directions:

- **Fidelity** — the drawn result should look like the input.
- **Draw time** — fewer primitives, fewer corners, fewer pen-ups.

Almost every threshold in the codebase is a point chosen on that
trade-off curve. When you change a fitting tolerance you are not
"fixing a bug", you are moving along that curve — which is why the
project now ships an objective fidelity metric (stage 6) so the
trade-off can be tuned with numbers instead of opinions.

A second, subtler point: a wavy line and a smooth arc can be *visually*
similar but cost very differently to draw, and a smooth arc and a
deliberately scalloped edge can fit the *same* arc within tolerance yet
mean different things. The pipeline spends a lot of effort deciding
when wobble is signal and when it is noise.

---

## 2. Pipeline at a glance

`release/__init__.py::default_pipeline(source)` runs the whole thing
and returns a 7-tuple:

```
skeleton, segment, graph,
low_geometry, high_geometry,
optimized_low_geometry, optimized_high_geometry
```

The stages:

```
 PNG
  │
  ▼
[1] Skeletonize   binarize → thin → resolve filled shapes → resolve crossings
  │               also: assign every binary pixel to a final-skeleton pixel
  │
  ▼
[2] Segment       trace polylines → fuse → repair junctions → fuse
  │               also: label every final-polyline pixel with a raw-segment id
  │
  ▼
[3] Graph         endpoints → vertices, polylines → edges
  │
  ▼
[4] Vectorize     fit Lines / Arcs / Circles to each polyline
  │   ├── low geometry   the real pipeline (joint constraint solve)
  │   │                  also: label every drawing command with raw-segment spans
  │   └── high geometry  a deliberately naive baseline for comparison
  ▼
[5] Route optimize  re-sequence primitives to minimize draw time
  │
  ▼
[6] Fidelity      rasterize the output, score it against the source
```

Stages 1–3 recover *geometry* (where the strokes are). Stage 4 turns
geometry into *primitives* (Lines/Arcs/Circles). Stage 5 turns
primitives into a fast *command order*. Stage 6 measures how well it
all worked.

A useful mental model: stages 1–3 are computer-vision/graph work and
are largely about robustness to messy input; stage 4 is geometric
fitting and optimization; stage 5 is a routing/TSP problem.

Stages 1, 2, and 4 each emit a **labeling** alongside their primary
output that points each downstream artifact back to where it came
from — see §9 for the end-to-end story. The labels are what let you
answer "this pixel of the original drawing became this drawing
command" without having to re-derive it from the geometry.

---

## 3. Stage 1 — Skeletonize  (`release/skeletonize/`)

Goal: reduce fat hand-drawn strokes to clean 1-pixel-wide centerlines.

`cleanup.py` binarizes the image and applies morphological thinning.
The result is a skeleton: every stroke becomes a 1-px-wide path.

### 3.1 Filled shapes  (`eyes.py`)

Morphological thinning collapses a hand-drawn filled disk (a pupil, a
dot) to a single pixel or a tight 2–5 pixel cluster. Downstream
segmentation either drops it as a sub-minimum-length polyline or
hands it on as a degenerate "stroke" the vectorizer can't reasonably
fit; the visual intent is lost. `eyes.py` runs after the cleaned
skeleton is available, finds each filled connected component, and
**replaces the collapsed medial-axis skeleton with the region's
1-pixel outer boundary** — a closed loop the segmenter then traces
cleanly and the vectorizer fits as a `Circle`.

Detection combines two signals per 8-connected component, applied to
the binary mask:

- ``skeleton_pixels <= max_skeleton_pixels`` — a filled disk's medial
  axis collapses to a few pixels; a normal stroke has one skeleton
  pixel per pixel of length and easily exceeds this even for short
  strokes.
- ``area / skeleton_pixels >= min_fill_ratio`` — a stroke has
  ``area / skel ≈ stroke_width`` (each skeleton pixel covers one
  column across the stroke), so a ratio threshold a few times larger
  than stroke width cleanly separates fills (10–50×) from strokes
  (~3–5×).

Both gates are needed: skeleton count alone catches short stroke
fragments (a tick mark); ratio alone catches any sufficiently fat
blob. Resolution wipes the interior skeleton pixels and writes back
the morphological-gradient ring re-skeletonized to one pixel wide —
that re-thinning matters, because the raw ring picks up 2-pixel
shoulders on irregular hand-drawn fills which would fragment the
trace into tiny pieces the min-length filter then drops.

### 3.2 Crossings  (`crossings.py`)

The other hard part is **crossings**. When two strokes briefly share
ink — an X, a T, a place where the pen doubled back — thinning
cannot tell them apart and collapses the overlap into a single
`)(`-shaped centerline. Naively tracing that produces wrong topology.

The crossing/"chromosome" resolver detects these with several
independent signals rather than one fragile threshold:

- a *local* adaptive stroke half-width (so thin and thick strokes can
  coexist in one drawing),
- counting how many skeleton arms exit a ring around the suspect
  region (via 8-connected components),
- a stroke-pairing test that looks for anti-parallel arms,
- a merged-segment-length test.

Where a real crossing is found, the resolver erases the dilated blob
and redraws two clean Bresenham strokes between the paired arm
endpoints, restoring the true topology.

### 3.3 Pixel-to-skeleton labeling  (`labeling.py`)

Once the final skeleton (`uncrossed`) is settled, every binary pixel
is assigned to its associated skeleton pixel by **geodesic flooding
through the stroke mask**, implemented as one
`skimage.segmentation.watershed` call with a flat elevation and one
marker per skeleton pixel. With a flat elevation, watershed
degenerates to multi-source BFS that assigns every reachable pixel to
the nearest marker by in-stroke distance. Two arms meeting at a
junction therefore partition the surrounding ink along the actual
stroke topology rather than along a Euclidean perpendicular bisector
that can leak across junctions on thick ink. Skimage's thinning
doesn't expose which pixel was absorbed into which surviving pixel,
so we recover the mapping after the fact.

The output `labeling` is an `(H, W)` `int32` array; pixels outside
the binary mask hold `-1`. See §9 for how downstream stages use it.

### 3.4 Output fields

`binary` (the filled stroke mask), `skeletonized` (the raw 1-px
skeleton), `collapsed` (small-hole cleanup), `eye_detection` (the
`EyeDetectResult` listing each filled region), `eyes_resolved` (the
skeleton with filled regions replaced by their boundary rings),
`detection` (the crossing detection), `uncrossed` (the final
skeleton — what stage 2 consumes), `labeling` (the per-binary-pixel
back-pointer to a skeleton pixel).

---

## 4. Stage 2 — Segment  (`release/segment/`)

Goal: turn the skeleton bitmap into a list of polylines (ordered point
sequences), one per stroke, with correct topology at junctions.

The stage runs **trace → fuse → repair → fuse**:

`trace.py` walks the skeleton pixel graph into polylines. (Note: it
works in `(row, col)` = `(y, x)` internally and converts to image-space
`(x, y)` at the boundary — see the conventions section.)

`fusion.py` joins polyline fragments that belong to the same stroke
but were split by chewed-up ink at a junction. It runs in two passes.
The first pass does **A\* search through the original binary** to test
whether two loose endpoints can be reached through real ink, scoring
candidate joins by reachability plus tangent alignment and resolving
them with greedy maximum-weight matching. The second pass runs after
repair, when endpoints are already co-located, and matches on
bridge-aware tangents alone.

`repair.py` reconstructs junctions. The key idea is a **curvature-gated
test**: a cluster of near-coincident points belonging to different
polylines is treated as a genuine X/Y/T crossing only if at least one
of those polylines shows a spike in `|r''|` (the second derivative of
position — a curvature kink) near the cluster. This distinguishes a
real crossing (kink-ratio 5–20× baseline) from a "parallel
coincidence" where two smooth strokes merely run close together
(ratio ≈ 1). When a real junction is confirmed, its location is solved
as the 2×2 least-squares intersection of the *stable* approach
tangents (sampled outside the noisy zone), and the affected region is
grown outward while `|r''|` stays elevated.

Relevant config lives in `RepairConfig`; the cascade-merge parameters
(`cascade_gap`, an index gap, and `cascade_max_jp_distance`, a pixel
distance) decide when two adjacent repairs on one polyline are merged.

### 4.1 Raw-segment labeling  (`labels.py`)

Every pixel of every final polyline (post-fuse-post-repair-post-fuse)
carries the id of the raw `trace.py`-output segment it came from.
The labels are exposed as `Segment.labeled_segments`, where each
entry pairs the final polyline with a tiled list of `RawSegmentSpan`
runs whose union exactly equals the polyline.

The mapping is recovered after the fact by **KDTree nearest-neighbour
lookup** through every raw-segment pixel, then compressing
consecutive same-id pixels into spans. This works because the
fuse/repair/post-fuse cascade moves pixels by at most
O(stroke-width) — fusion bridges hug ink between joined endpoints,
and repair's LS-solved junction point sits within the original
cluster — so each new pixel's nearest raw pixel is in the raw segment
that semantically owns it. Maintaining labels through every
concatenation/splice in the cascade would have been invasive; the
after-the-fact lookup is one KDTree build and one query per final
polyline.

Each `RawSegmentSpan` exposes both a half-open final-polyline range
(`start`, `end`) AND an inclusive raw-segment range (`raw_start`,
`raw_end`). The raw range is inclusive on both ends because a reverse
traversal (fusion can splice a raw segment in either direction) would
otherwise land at `raw_end = -1` when ending at raw index 0 —
indistinguishable from Python's "all but last" slice sentinel.
`raw_end < raw_start` therefore unambiguously encodes a reverse walk.

---

## 5. Stage 3 — Graph  (`release/graph/`)

Goal: a topological view of the drawing.

Endpoints of polylines are clustered with a tolerance so that strokes
meeting at a point share a vertex. The result, `StrokeGraph`, treats
**endpoint locations as vertices and polylines as edges**. A closed
loop (a circle) has no endpoints, so it is modeled as a "loop edge"
attached to a single virtual vertex at its rightmost point.

This graph is what stage 5's routing reasons about.

---

## 6. Stage 4 — Vectorize  (`release/vectorize/`)

Goal: replace each polyline (a dense point list) with a short chain of
primitives — `Line`, `Arc`, `Circle` — that the robot can draw.

There are two independent implementations:

- **low geometry** (`vectorize/low_geometry/`) — the real pipeline.
- **high geometry** (`vectorize/high_geometry/`) — a deliberately
  naive baseline kept only for comparison. It segments at curvature
  peaks, classifies each piece as a line or arc, and greedily orders
  the strokes. There is no joint solve and no consolidation. Treat it
  as frozen: its job is to be the "before" number that the low
  pipeline must beat.

The rest of this section is the low-geometry path.

### 6.1 Primitives

`primitives.py` defines `Line`, `Arc`, `Circle`. Arcs are stored with a
**bulge** parameterization:

```
bulge = tan(sweep / 4)
```

Bulge is signed (direction of curvature), is 0 for a straight line, and
avoids the branch-cut problems of storing a center + sweep directly.
Endpoints `p0`, `p1` plus bulge fully determine an arc; `radius()`,
`center()`, `sweep()`, `chord()` are derived.

### 6.2 Fitting a polyline to a chain  (`fitting.py`)

`fit_polyline(pts)` is the heart of the stage. It tries, roughly in
order of cheapest-good-answer first:

1. **Closed-circle shortcut.** If the polyline is near-closed,
   corner-free, fits a circle with low RMS, and has near-unit aspect
   ratio, return a single `Circle`. The aspect check matters: an oval
   (e.g. a cat's head) fits a circle with deceptively low RMS, but
   rendering it as a true circle pushes the radius to the *average* of
   the two axes and visibly overshoots — so ovals are rejected here
   and fall through to subdivision.

2. **Structural splitting.** Compute split points and recurse on each
   sub-range:
   - `find_corners` — tangent-direction discontinuities (kinks of
     ~50–80°): ear tips, heart cusps, box corners.
   - `find_inflections` — *sustained* curvature sign reversals: the
     smooth S-curve where a stroke switches from turning one way to
     the other. No kink, but no single arc can span it.
   - `find_closed_subloops` — near-closed circular sub-loops embedded
     in an open stroke (a wheel rim drawn as part of a longer
     contour). This now returns *all* non-overlapping loops, so a
     figure-8 or a chain of bubbles is handled, not just the largest
     loop.

3. **Single-primitive shortcuts.** A corner-free polyline that fits one
   `Arc` (or `Line`) tightly enough becomes that one primitive.

   The arc shortcut is **density-gated**, and the reasoning is worth
   internalizing because it is a clean example of the
   signal-vs-noise problem. The fit RMS alone cannot tell a
   clean arc carrying pen tremor from a deliberately wavy edge — both
   can land at ~9 px RMS. What *can* tell them apart is whether chain
   subdivision can be trusted to do better, and that depends on sample
   density: subdivision splits the points among 2–3 pieces and re-fits
   each, which is only reliable when each resulting piece still has
   enough points to constrain its fit and the joint solve. So a
   borderline-RMS polyline keeps the one-arc shortcut when it is too
   *sparse* to subdivide safely, and falls through to subdivision when
   it is *dense* (the cheese-block edges, ~135 points each, fit one
   arc at ~9 px RMS but are visibly better as a short arc chain). The
   relevant constants are `_SINGLE_ARC_SHORTCUT_RMS_CAP`,
   `_SINGLE_ARC_SHORTCUT_RMS_LOOSE_CAP`, `_SUBDIVISION_MIN_POINTS`.

   Separately, `fit_arc` rejects any arc whose **sagitta**
   (perpendicular bow away from the chord, `r·(1 − cos(sweep/2))`) is
   sub-pixel: such an "arc" is visually a straight line and is
   degraded to one so the solver and the firmware estimator never see
   spurious curvature.

4. **MDL chain subdivision.** When nothing above applies, subdivide the
   polyline into a chain of primitives by minimizing a
   Minimum-Description-Length-style objective: total fit residual plus
   a per-primitive penalty `λ`. More pieces fit the data better but
   cost `λ` each — the penalty is what stops the fitter from chasing
   every wiggle. Implemented as a dynamic program (`use_dp`) with a
   top-down fallback.

5. **`fuse_chain`.** Subdivision can over-segment a gentle curve into
   many tiny `Line`s; `fuse_chain` greedily re-merges consecutive
   pieces whose combined points still fit a single `Line` or `Arc`.
   This is what prevents a `line-spin-line-spin-…` staircase where one
   smooth arc belonged. It refuses to fuse across a curvature reversal
   (an S-curve), because averaging opposite curvatures yields a
   near-straight arc that mis-renders the shape.

### 6.3 The joint constraint solve  (`solve.py`, `residuals.py`)

Fitting each polyline independently leaves the drawing slightly
incoherent: strokes that should meet exactly don't, lines that should
be parallel aren't, arcs that should share a center don't. The solve
fixes this globally.

All primitive parameters are packed into one flat vector and handed to
a least-squares solver (`scipy.optimize.least_squares`, Trust-Region
Reflective). The residual vector stacks several families, each with a
calibrated weight:

- **data fidelity** — primitives stay near their source points;
- **endpoint anchors** — the chain's free ends stay put;
- **coincidence** — endpoints that should touch, touch (high weight,
  effectively a hard constraint);
- **on-curve** — a point constrained to lie on a primitive;
- **G1 smoothness** — tangents match across an internal joint; encoded
  as the *cross product* of the two unit tangents so there is no
  angle branch cut;
- **beautification** — parallel / perpendicular / equal-radius /
  concentric relations (see 6.4);
- **regularization** — a `log(radius)` term and a `bulge²` term that
  keep an under-constrained arc from running off to an enormous radius.

Because each residual depends on only a handful of parameters, the
Jacobian is extremely sparse. A hand-built **sparsity pattern** is
passed to the solver; without it the dense Jacobian would dominate
runtime.

### 6.4 Beautification  (`beautify.py`)

After the first solve, `beautify.py` scans all primitives for
near-relationships — two lines almost parallel, two arcs almost
concentric, two radii almost equal — and emits them as soft
constraints. A second solve then snaps those relations exact (or
nearly so).

`merge_arc_pairs` additionally recognizes when a circle was fitted as
several arcs and rebuilds the single `Circle`. It handles two arcs
(two halves of a loop) and, via a second pass, **three or more** arcs.
The 3+ case qualifies a group by *angular-union coverage*: project
every arc onto the shared mean center, take the union of the angular
intervals they sweep, and require it to tile ~360°. This is what still
rejects a crescent moon — a lune's two arcs sweep the *same* angular
sector at two radii, so their sweeps sum past 320° but their angular
union does not reach 360°.

### 6.5 Routing  (`routing.py`)

The fitted primitives still need an order. `routing.py` models this as
a graph problem: primitives are edges, endpoints are vertices, and the
goal is a tour that draws every primitive with the fewest **pen-ups**.
That is an Eulerian path when the graph has 0 or 2 odd-degree
vertices, and the Chinese-Postman problem otherwise — NetworkX's
`eulerize` adds duplicate edges to fix parity, which translates back
into extra pen-up moves.

Important scope note: this stage minimizes pen-ups, **not total
turning**. Among the many valid Eulerian paths it does not pick the
one with the least in-place spinning. Minimizing spin cost is stage
5's job, and stage 5 is therefore a *required* final stage — the
low-geometry command snapshots are pen-up-optimal but not
time-optimal on their own.

### 6.6 The two output snapshots

`LowGeometryVectorize` exposes two command streams:

```
snapshot                primitives                tour
----------------------  ------------------------  --------------------
commands_fitted         primitives_fitted         tour
commands_consolidated   primitives_consolidated   tour_consolidated
```

`commands_consolidated` (post-beautify, post-arc-merge) is the intended
output; `commands_fitted` is the pre-beautification snapshot kept for
diagnostics.

### 6.7 Per-command raw-segment labels  (low geometry, `labels.py`)

The shared label types `CommandSpan` and `LabeledCommand` live in
`vectorize/labels_common.py`; both vectorizers emit them, so low- and
high-geometry labels are the same type and reference the same
raw-segment ids (directly comparable). Low geometry produces them
*structurally* (this section); high geometry produces them
*geometrically* (§6.8).

When the low-geometry vectorizer is built with
`labeled_segments=segment.labeled_segments`, it produces
`labeled_commands_consolidated` — one `LabeledCommand` per emitted
command. Drawing commands (pen-down line / arc / circle) carry a list
of `CommandSpan` runs naming the raw segments their primitive came
from; spins and pen-up transit commands carry an empty span list.

Each `CommandSpan` has:

- `raw_segment_id` — pointer back to `Segment.segments`.
- `raw_start`, `raw_end` — inclusive indices in the raw segment;
  `raw_end < raw_start` means the command walks the raw segment in
  reverse (whenever the tour traverses a primitive end → start).
- `command_start_ratio`, `command_end_ratio` — fraction along the
  command's geometry (0 at the command's start, 1 at its end), so
  callers can map ratio → spatial position via the command's primitive.

The link is rebuilt in three hops, each contained in `labels.py`:

1. **command → primitive.** Replay `to_commands`'s control flow to
   find the last command emitted per tour entry — that's the drawing
   command. Spins and pen-ups are emitted before it, by construction.
2. **primitive → final-polyline range.** Each primitive came from one
   `ChainPiece` carrying `start_idx` / `end_idx` into the *subsampled*
   polyline `fit_polyline` saw. `build_chains` applies a deterministic
   `np.linspace` subsample at `polyline_subsample_cap`; we recompute
   it here to map subsampled indices back to raw (= final-segment)
   indices.
3. **final-polyline range → raw-segment spans.** The segment stage's
   `LabeledSegment` carries per-pixel `raw_ids` / `raw_indices`, so we
   walk the relevant pixel range and compress consecutive same-id
   pixels into command spans. Each run's endpoints get projected onto
   the primitive geometry (analytic projection for `Line`/`Arc`/
   `Circle`) for the ratio fields.

Most raw segments are wholly consumed by one primitive and yield
exactly one `CommandSpan` covering `raw[0:len]` and ratio
`[0.0, 1.0]`; spans that don't cover a full raw segment fall out of
the same code path when a primitive consumed only part of a raw
segment (a corner split mid-stroke).

Important scope note: the *structural* low-geometry labeler is tied to
the consolidated tour it walked, so it only labels
`commands_consolidated`. `OptimizeRoute` (stage 5) reorders commands
and re-emits transit spins/lines, so that stream is **not** structurally
labeled — but the geometric labeler in §6.8 can label any stream,
including the optimized one, since it reads only the emitted geometry.

### 6.8 Per-command raw-segment labels  (high geometry / geometric, `labels_common.py`)

The high-geometry baseline is frozen and tracks no point-range
provenance through its segment/classify/order logic, so there is no
structural command → primitive → polyline chain to walk. Its labels
are recovered **geometrically** by `label_commands_geometric`, which
needs only the emitted command stream plus the raw segments:

1. Replay the command stream to recover each pen-down command's drawn
   path (a line's two endpoints, an arc's center / radius / sweep).
2. Sample that path at ~1 point per pixel.
3. Match each sample to the nearest raw-segment pixel via a KDTree —
   the same KDTree construction the segment stage uses (§4.1).
4. Compress consecutive same-id samples into `CommandSpan` runs, taking
   the raw-index range from the matched pixels and the ratio range
   from the sample parameters.

`HighGeometryVectorize` takes an optional `raw_segments` argument
(the pipeline passes `segment.segments`); when supplied it populates
`labeled_commands` parallel to `commands`. For the geometric labeler,
`LabeledCommand.primitive_id` is the drawing command's ordinal in draw
order (a stable handle, not an index into any primitive list) and
`final_segment_index` is `None` (the baseline doesn't carry the
final-segment abstraction).

Because it reads only geometry, the same function labels the
route-optimized stream too — call it with `optimized.commands` and the
same `segment.segments` when you need labels on stage 5's output.

---

## 7. Stage 5 — Route optimization  (`release/optimize.py`)

Goal: re-sequence the pen-down primitives so total firmware draw time
is minimal. This is a TSP-like problem with a twist — each primitive
can be drawn in **either direction**, and the cost of moving between
two primitives depends on the direction of both.

### 7.1 The cost cache

`OptimizeRoute` first builds a transition-cost cache. For `n`
primitives it precomputes a 4-D matrix

```
trans[a, ra, b, rb]   # time to go from primitive a (direction ra)
                      #            to primitive b (direction rb)
```

plus `trans_start[a, ra]` for the move from the robot's initial state.
Drawing time itself is direction-independent and stored once as a
scalar. With the cache, evaluating any tour is just `n` array lookups.

The transition cost comes from a faithful port of the robot's
**firmware motion model**: a differential-drive arc has the inner and
outer wheels running at different speeds, and `estimate_arc_time`
models that; spins and straight moves have their own time formulas.
The same model powers both the optimizer and the reported draw-time
numbers, so they are consistent.

### 7.2 The search

Local search from **two seeds**, keeping whichever is cheaper:

1. a nearest-neighbour greedy tour, and
2. the input ordering itself.

Seeding from the input matters: without it a nearest-neighbour start
can converge to a local optimum that is *worse* than the already-decent
Eulerian order the optimizer was handed. Local search then runs 2-opt
(reverse a tour slice — which also flips the drawing direction of every
primitive in it) and or-opt (relocate a short run, optionally flipped).
A final guard reverts to the input if, despite everything, the result
came out slower: the optimizer's contract is *never worse than the
input*.

### 7.3 The incremental delta — and the symmetry it rests on

A 2-opt move reverses a slice and flips the direction of every
primitive in it. Naively that changes every transition inside the
slice, making each move O(n) to evaluate and a pass O(n³). But the
cost cache has a symmetry:

```
trans[a, ra, b, rb] == trans[b, 1−rb, a, 1−ra]
```

In words: reversing a directed transition and flipping both endpoints'
traversal directions leaves the cost unchanged. (It follows from the
construction of the cache — the flipped traversal's entry/exit pose is
the other endpoint's pose with heading rotated by π; the gap vector
negates but its length is invariant, and every spin magnitude is
invariant under simultaneously rotating both headings by π.)

The consequence: when a slice is reversed-and-flipped, its *internal*
transitions keep their cost — only the **two boundary transitions**
change. So a 2-opt move is an **O(1)** delta, and a pass is O(n²). The
same argument makes or-opt incremental. `_VERIFY_DELTAS` in
`optimize.py` cross-checks the delta against a full recompute when you
need to validate changes to the cost model.

---

## 8. Stage 6 — Fidelity metric  (`release/fidelity.py`)

The pipeline reports *cost* (draw time) and *complexity* (primitive
counts), but those say nothing about whether the output looks like the
input. `fidelity.py` closes that loop.

`coverage_metrics` rasterizes a command stream back into a pixel mask
and compares it against the source via Euclidean distance transforms:

- **precision** — fraction of output pixels within tolerance of source
  ink. Low precision = spurious strokes / overshoot.
- **recall** — fraction of source skeleton pixels within tolerance of
  an output pixel. Low recall = missed or over-smoothed detail.
- **f1** — harmonic mean of the two.
- **chamfer** — mean symmetric distance; a single scalar, lower better.

The tolerance defaults to the estimated stroke width, so "within a
stroke width" counts as on-target. `test.py` prints this per example.
Use it: when you change a fitting threshold, this metric tells you
whether you moved up or down the fidelity/cost curve. The cheese
over-smoothing fix in §6.2, for instance, was accepted only because
the metric confirmed 11 of 12 sampled drawings improved with none
regressing.

---

## 9. Cross-stage labeling — pixel ↔ command

Three stages emit a labeling alongside their primary output so the
pipeline can answer "which drawing command does this source pixel
become?" without re-deriving it from geometry. The chain is:

```
binary pixel  →  skeleton pixel  →  raw segment id  →  final segment span  →  drawing command
                 (§3.3)            (§4.1)            (§4.1)                 (§6.7 / §6.8)
```

Each hop is implemented independently and exposes its lookup
artifact, so downstream callers can stitch them together at whatever
granularity they need:

| Hop | Stage | Artifact | Where |
|-----|-------|----------|-------|
| binary pixel → skeleton pixel | Skeletonize | `labeling: (H, W) int32` (-1 = outside binary) | `Skeletonize.labeling` |
| skeleton/final pixel → raw segment id (+ index, + raw range) | Segment | `labeled_segments: List[LabeledSegment]` with per-pixel `raw_ids`/`raw_indices` and run-compressed `RawSegmentSpan` lists | `Segment.labeled_segments` |
| drawing command → raw segment id (+ index, + ratio), low geometry | Vectorize | `labeled_commands_consolidated: List[LabeledCommand]` (structural) | `LowGeometryVectorize.labeled_commands_consolidated` |
| drawing command → raw segment id (+ index, + ratio), high geometry | Vectorize | `labeled_commands: List[LabeledCommand]` (geometric) | `HighGeometryVectorize.labeled_commands` |

Both vectorizers emit the same `CommandSpan` / `LabeledCommand` types
(defined in `vectorize/labels_common.py`) against the same raw-segment
ids, so a low- and a high-geometry command are directly comparable.
They differ only in *how* the link is found: low geometry tracks
provenance structurally (§6.7); high geometry, which is frozen and
keeps no provenance, recovers it geometrically (§6.8).

Walking forward: a binary pixel `(y, x)` lands on
`Skeletonize.labeling[y, x]`, a row-major flat index into the True
pixels of the final skeleton. That skeleton pixel's image position
appears inside one of the final-polyline points; the segment-stage's
`raw_ids[i]` / `raw_indices[i]` tell you which raw `trace.py`
polyline (and which index in it) the final point was credited to;
the vectorize-stage's `CommandSpan` whose `[raw_start, raw_end]`
brackets that raw index gives the drawing command and its
`command_start_ratio` / `command_end_ratio` window.

Walking backward (from a command to source pixels) is the same chain
in reverse. Use this for diagnostics ("why is this part of the
drawing missing?") or for source-color-preserving renderings ("paint
each command in the colour of the source pixels it covers").

Each stage also ships a debug visualization in `release/visualize.py`
that uses a golden-ratio hue palette with adjacency-contrast
renumbering, so neighbouring labels always land at well-separated
hues. The visualizations are saved as `examples/<name>.<stage>.labeling.png`
by the test harness — see §11.

Scope limits worth knowing:

- The skeleton-pixel mapping is *geodesic-via-watershed*, not
  Euclidean. Two arms meeting at a junction partition their
  surrounding ink along the actual stroke topology rather than along
  a Euclidean perpendicular bisector that can leak across junctions
  on thick ink.
- The segment-stage mapping is *nearest-neighbour-via-KDTree*. It
  exploits that the fuse/repair/post-fuse cascade moves pixels by at
  most O(stroke-width), which is true for every transformation in
  the cascade — fusion bridges hug ink between the joined endpoints,
  and repair's LS-solved junction point sits within the original
  cluster.
- The low-geometry command mapping is *structural* and is produced
  for `commands_consolidated` only; `OptimizeRoute` reorders commands
  and re-emits transit spins/lines, so that stream isn't structurally
  labeled. The high-geometry command mapping is *geometric*
  (sample-and-match against the raw segments) and so works on any
  command stream — including the route-optimized one. Use
  `label_commands_geometric(optimized.commands, segment.segments, …)`
  when you need labels on stage 5's output.

---

## 10. Auto-configuration  (`release/auto_config.py`)

There is no per-image tuning. `derive_configs(source)` measures one
quantity — the characteristic **stroke width**, estimated as twice the
median of the distance transform over stroke pixels — and scales every
tolerance from it.

The scaling rule is explicit and worth remembering when you add a
config value:

- **pixel distances** scale linearly with `s = stroke_width / 3`;
- **areas** scale with `s²`;
- **angles, ratios, counts, index gaps, and boolean flags do NOT
  scale** — they are intrinsically scale-invariant.

If you hardcode a pixel threshold inside a function instead of routing
it through `derive_configs`, you have quietly broken scale-invariance
for large or small drawings. Put pixel thresholds in the config.

One principled exception lives in `auto_config.py` itself: the
chromosome / crossing-detection `detect` block is deliberately *not*
scaled. Two reasons. First, chicken-and-egg — a chromosome *is* a
region of wide overlapping ink, and that wide ink inflates the
distance-transform median that drives `estimate_stroke_width`, so
scaling the crossing detector by that estimate feeds its own input
pathology back into its tuning. Concretely, drawings with dense
overlapping strokes can over-estimate to two or three times the true
stroke width (and very thin drawings can under-estimate the other
way), which is enough to miss real crossings or hallucinate false
ones. Second, the chromosome detector already adapts to stroke width
*locally* via its internal `local_tau` (see §3), so a global
multiplier on top would be redundant. If you ever need genuine
multi-resolution chromosome detection, derive its thresholds from
`local_tau`, not from the global median estimate. The same caveat
applies more generally: `estimate_stroke_width` is robust to most
hand-drawn input but is not trustworthy on drawings dominated by
filled regions or dense crosshatching, so any new pixel threshold
that operates on exactly those regimes deserves a moment's thought
about whether scaling is right for it.

---

## 11. Repository layout

```
release/
  __init__.py          default_pipeline() — runs all stages
  commands.py          DrawingCommand types; the robot interface
  auto_config.py       derive_configs(): stroke-width-driven scaling
  optimize.py          stage 5 — OptimizeRoute, cost cache, firmware model
  fidelity.py          stage 6 — objective fidelity metric
  visualize.py         diagnostic renders (skeleton/segments/overlays/SVG/GIF)

  skeletonize/         stage 1
    cleanup.py           binarize + thinning
    eyes.py              filled-shape detection; swap medial axis for boundary
    crossings.py         crossing / chromosome detection and resolution
    labeling.py          binary pixel → skeleton pixel (geodesic watershed)
  segment/             stage 2
    trace.py             skeleton bitmap → polylines (raw segments)
    fusion.py            join fragments of the same stroke (A* + tangents)
    repair.py            curvature-gated junction reconstruction
    labels.py            final-polyline pixel → raw segment id (KDTree)
  graph/               stage 3 — StrokeGraph (endpoints→vertices)
  vectorize/
    labels_common.py   shared CommandSpan/LabeledCommand + geometric labeler
    low_geometry/      stage 4, the real path
      fitting.py         polyline → primitive chain (corners/inflections/MDL)
      primitives.py      Line / Arc / Circle
      solve.py           global least-squares joint constraint solve
      residuals.py       residual terms for the solve
      beautify.py        detect near-relations; merge arcs into circles
      manifest.py        constraint bundle types
      routing.py         Eulerian / Chinese-Postman ordering
      labels.py          drawing command → raw segment spans (structural)
    high_geometry/     stage 4, the naive comparison baseline
                       (command labels recovered geometrically)

test.py                runs every example end to end, prints metrics
test.sh                unit tests, then the example run
tests/                 unit tests (geometry + simulator sanity checks)
```

---

## 12. Running it

```sh
./test.sh
```

runs the unit tests (fast geometry/simulator checks — if these fail the
example run is not worth starting) and then processes every PNG in
`examples/` end to end, printing per-example primitive counts, draw
times, the fidelity metric, and the regression checks.

Useful environment variable: `PIPELINE_GIFS=1` enables the animated
comparison GIFs, which are off by default because they dominate
wall-clock time.

To run the pipeline programmatically:

```python
from release import default_pipeline
skeleton, segment, graph, low, high, opt_low, opt_high = \
    default_pipeline("examples/smile.png")
print(opt_low.commands)               # the robot command stream
print(opt_low.estimated_time_after)   # estimated draw time (s)

# Cross-stage labels (see §9):
print(skeleton.labeling.shape)                       # (H, W) int32 per-pixel
print(segment.labeled_segments[0].spans)             # raw-segment runs
print(low.labeled_commands_consolidated[0].spans)    # low-geom command labels
print(high.labeled_commands[0].spans)                # high-geom command labels

# Geometric labeler works on any command stream, incl. the optimized one:
from release.vectorize.labels_common import label_commands_geometric
opt_labels = label_commands_geometric(opt_low.commands, segment.segments)
```

A single example processes in ~10–15 s; the full suite takes several
minutes. `test.py` parallelizes with a memory-aware worker cap.

For each example the harness writes a set of debug PNGs into
`examples/`:

| Filename | Stage | Content |
|----------|-------|---------|
| `<name>.skeleton.png` | 1 | binary, cleaned skeleton, eye detection, eye-resolved skeleton, crossing detection, final skeleton (6-panel) |
| `<name>.skeleton.labeling.png` | 1 | every binary pixel coloured by its skeleton-pixel label (§3.3) |
| `<name>.segments.png` | 2 | per-stage segment renders (trace, fused, repaired, post-repair fused) |
| `<name>.segments.labeling.png` | 2 | every final-polyline span coloured by its raw-segment id (§4.1) |
| `<name>.graph.png` | 3 | stroke graph with vertices and edges |
| `<name>.vectorized.svg` | 4 | high / low / optimized 3-panel comparison |
| `<name>.commands.labeling.png` | 4 | every low-geometry drawing command coloured by its raw-segment-span ids (§6.7) |
| `<name>.commands.labeling.high.png` | 4 | same for the high-geometry baseline, recovered geometrically (§6.8) |
| `<name>.heatmap.png`, `<name>.overlay.png`, `<name>.overlay.clean.png` | 5–6 | firmware-time heatmap and source-image overlays |

Adjacent labels in the four `*.labeling*.png` renders always land at
well-separated hues (golden-ratio palette + stride renumbering), so
an over-merge shows as a single colour where two would be expected
and an over-split shows as a hue jump inside one continuous run. The
two `commands.labeling` renders share the raw-segment colour basis,
so you can compare how the low and high vectorizers carve the same
raw strokes into commands.

---

## 13. Conventions and gotchas for contributors

- **Coordinate frame.** Image space `(x, y)`, x right, y down. `trace.py`
  works in `(row, col)` = `(y, x)` internally and converts at its
  boundary — be careful when reading it.
- **Angle sign.** Positive angle = counter-clockwise in image
  coordinates (which renders as clockwise on screen, because y points
  down). For arcs, `radius` is always positive; the sign of the sweep
  selects direction.
- **Arc storage.** Arcs are `(p0, p1, bulge)` with `bulge = tan(sweep/4)`.
  Do not assume a center+sweep representation.
- **`default_pipeline` returns 7 values.** If you change it, update
  every caller — a stale 5-tuple unpack is exactly the bug that used
  to crash the test harness.
- **High geometry is frozen.** It is a baseline, not a place to add
  features. Improvements go in low geometry. (The raw-segment command
  labels it now carries are the exception that proves the rule: they're
  recovered *geometrically* from its emitted commands without touching
  the frozen segment/classify/order logic — see §6.8.)
- **Two command snapshots.** `commands_consolidated` is the real
  output; `commands_fitted` is a diagnostic snapshot. Don't confuse
  them. Neither is route-optimized — that's `OptimizeRoute`.
- **Tolerances are trade-off points, not constants of nature.** Before
  changing one, predict which way it moves the fidelity/cost curve,
  then confirm with the stage-6 metric across the whole example set —
  not by eyeballing one drawing.
- **Validate optimizer changes with `_VERIFY_DELTAS = True`.** If you
  touch the firmware cost model or the incremental deltas, flip that
  flag, run the suite once to confirm every delta matches a full
  recompute, then flip it back.
- **Most thresholds belong in `auto_config.py`.** A hardcoded pixel
  distance inside a function silently breaks scale-invariance.
- **Labeled spans use mixed conventions.** Final-polyline `start`/`end`
  on `RawSegmentSpan` are half-open (Python slice). Raw-segment
  `raw_start`/`raw_end` on `RawSegmentSpan` *and* `CommandSpan` are
  inclusive on both ends. The inclusive convention is deliberate:
  a half-open reverse walk ending at raw index 0 would land at
  `raw_end = -1`, which is indistinguishable from Python's
  "all but last" slice sentinel. To slice the raw segment regardless
  of direction, use
  `raw_segments[id][min(raw_start, raw_end) : max(raw_start, raw_end) + 1]`.
- **`OptimizeRoute` invalidates the structural command labels.**
  Stage 5 reorders primitives and re-emits transit commands, so the
  low-geometry structural labels (`low.labeled_commands_consolidated`,
  tied to `low.commands_consolidated`) don't carry over to its output.
  The geometric labeler does carry over: call
  `label_commands_geometric(opt_low.commands, segment.segments)` for
  labels on the post-optimization stream (§6.8).

---

## 14. Appendix — key formulas

**Arc geometry.** For an arc of radius `r` and signed sweep `θ`:

```
bulge   = tan(θ / 4)
chord   = 2 r sin(|θ| / 2)
sagitta = r (1 − cos(|θ| / 2))      # perpendicular bow from the chord
```

The sagitta is the "how curved is this really" measure: an arc with
sub-pixel sagitta is degraded to a line (§6.2).

**Circle fit.** Closed loops and circular sub-loops are fit with an
algebraic (Kåsa-style) circle fit — least squares on
`x² + y² + Dx + Ey + F = 0`, which is linear in `(D, E, F)` and so has
a closed-form solution; center and radius follow. RMS of the radial
residual is the quality score.

**Curvature-kink test (junction repair).** Sampling position `r(t)`
along a polyline, `|r''(t)|` spikes at a real crossing and stays flat
where two strokes merely run parallel. The spike-to-baseline ratio
(~5–20× vs ~1×) is the discriminator (§4).

**Firmware motion model (route cost).** A differential-drive arc of
radius `r` through sweep `θ` runs the outer wheel a distance
`(r + w/2)|θ|` and the inner wheel `(r − w/2)|θ|` for wheelbase `w`;
arc time is governed by the outer wheel against the max wheel speed.
Spins and straight drives have their own time formulas. See
`estimate_arc_time` and friends in `optimize.py`.

**2-opt cost symmetry.** The transition cache satisfies

```
trans[a, ra, b, rb] == trans[b, 1−rb, a, 1−ra]
```

so reversing a tour slice (with the direction flips a 2-opt move
implies) leaves every internal transition's cost unchanged — only the
two boundary transitions change. This is what makes the 2-opt /
or-opt delta O(1) (§7.3).

**MDL subdivision objective.** A chain of `k` primitives fitting a
polyline with total squared residual `R` is scored as `R + λk`; the
subdivider minimizes this. `λ` is derived from the line tolerance, so
"is one more primitive worth it" is asked in residual units.