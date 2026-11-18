# Stereo asset format, version 1

An optional export directory contains:

- `left.png`, `right.png`: RGB images on the full aligned working-resolution grid, before animation crop and output rotation.
- `stereo.npz`: float32 `disparity_left`, `disparity_right`; boolean `reliable_left`, `reliable_right`, `support_left`, `support_right`. Load using `numpy.load(path, allow_pickle=False)`.
- `manifest.json`: schema/version, conventions, analysis and camera provenance, anchor, motion settings, animation crop, and coverage statistics.
- `depth_meters.npy`: only for measured calibration; invalid or nonpositive depth is NaN.

The coordinate origin is the top-left pixel. X increases rightward; Y increases downward. Left disparity is `x_left - x_right`; right disparity is `x_right - x_left`. The two maps live on their respective eye grids. They are not negatives of one another at the same array index.

For viewpoint `t` in `[0,1]`, projected coordinates before cropping are:

```
from left:  x' = x_left  - t * disparity_left       + t * anchor_disparity
from right: x' = x_right - (1-t) * disparity_right  + t * anchor_disparity
```

The same anchor shift is applied to both projections. Anchor X/Y are in the aligned left image. `render.crop_xyxy` is an exclusive-end crop applied to every animation frame. Output rotation is applied after cropping. Full-grid maps remain unrotated and uncropped for reuse. Disparity units must scale if a consumer resizes these grids.

`reliable_*` records stereo consistency, not a probability of correctness. Dense disparity includes filtered or propagated estimates outside those masks. `support_*` records pixels that came from the source image after alignment. The renderer favors reliable source contributions; inferred geometry must not be confused with measured depth.

Nominal profiles store the right-eye affine transform and explicitly do not provide metric depth. The source resize scale is in analysis metadata. Measured profiles store calibration, rectified projection matrices, and Q at working resolution. Metric depth comes from OpenCV reprojection through Q, masked by stereo consistency.

Coverage statistics distinguish uncovered pixels before fill, after fill, and visible unfilled fraction after the common crop. Small background fills are deterministic, but are not captured-pixel evidence at the missing viewpoint. The format is not a claim of compatibility with an existing PortalEffectWeb loader.

## Measured calibration input

Supply JSON through `--profile path.json` or the GUI's **Load measured calibration** button:

```json
{
  "image_size": [3584, 2016],
  "K1": [[3000, 0, 1792], [0, 3000, 1008], [0, 0, 1]],
  "K2": [[3000, 0, 1792], [0, 3000, 1008], [0, 0, 1]],
  "D1": [0, 0, 0, 0, 0],
  "D2": [0, 0, 0, 0, 0],
  "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "T": [-0.075, 0, 0]
}
```

**These numbers are an illustrative schema example, not a W3 calibration.** Use values measured from your own camera and capture mode. K uses pixels at `image_size`; D uses OpenCV distortion conventions; R and T transform left camera coordinates into right camera coordinates, with T in meters. Only horizontal-baseline profiles are supported. A profile for a different crop/aspect ratio is rejected. Swapping eyes with a measured profile is rejected to avoid silently misusing its geometry.

## Photo edits and styles (additive metadata fields)

`render.photo_edits` records the photographic parameters and normalized crop. `render.photo_crop_xyxy` is an exclusive-end pixel rectangle on the **rotated common animation crop**, not on the full stereo-map grid. Apply the common crop, rotation, then this photo crop to locate the exported field of view. Keep disparity maps on their original full aligned grid.

`render.style` records the processing style and its parameters; `render.antialias_effective` reports `2x` or `off`, including the automatic override for Stereo Echo. `render.color_balance_gains` records the shared left-eye-derived balance estimate. Raw aligned images and disparity maps in the asset bundle remain unedited. In anaglyph mode, apply photographic corrections separately to both eyes before combining channels.

The original common-grid hole statistics precede the optional user crop and artistic effects. They do not quantify the artistic displacement, anaglyph crosstalk, or remaining geometry error.

## Independent render grid and temporal effects (v0.3)

GUI/CLI bundles now contain the **export render grid**, which can differ from the cached analysis grid. `analysis.geometry_size` retains the analysis dimensions; `analysis.working_size`, `render.render_grid`, the stored maps, anchor, alignment matrices, and measured Q refer to the render grid. `analysis.source_color_detail` indicates that colors were reopened from the original images. The disparity values are scaled to pixels of this grid. Guided enlargement conservatively propagates validity; it never upgrades an inferred contribution into a reliable measurement.

After common crop, rotation, and `photo_crop_xyxy`, resize to `render.output_width` with proportional even height. `export_supersampling_requested` is the pre-crop source-width multiplier, not a guarantee of that many independent samples after a tight crop or above native resolution.

`render.effects` contains enabled independent effects, `effects_seed` the deterministic noise seed, and `timeline_frames` the number of chronological frames in one loop. Effects use final-output pixels. A temporal sequence expands Classic holds to whole video frames once, and includes distinct forward/return frames. Persistence uses circular exponential history in linear RGB. The old `style` field remains for Python API compatibility; GUI/CLI effects now use the post-render timeline stack.

## v0.4 additive rendering fields

`motion.fusion` selects `symmetric`, `legacy`, `left`, or `right`; `motion.repair_edges` requests localized unsupported-edge repair (ignored by legacy fusion and non-interpolated modes). Neither changes the stored analysis validity masks.

`render.light_leak_frames` lists zero-based chronological playback indices affected by a fixed-position light leak. Counts refer to playback samples, not necessarily distinct encoded GIF images. Selection repeats each loop. `render.curvature_safe_scale` records the fixed normalized inset and `curvature_stage` records sampling before output resizing. Crop coordinates still refer to the selected source region; the curvature inset maps a subset of that region to the final rectangle. Screen-pattern effects remain on the final-resolution grid.
