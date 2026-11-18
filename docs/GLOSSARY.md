# Settings and features

## Import, camera, and geometry

| Setting | What it does and when to use it |
| --- | --- |
| Open MPO | Loads both eye images from one stereo file. Use this instead of the separate single-eye JPEG. |
| Open left / right JPEGs | Loads an explicit pair. Choose the left image first, then the right. PNG inputs are also accepted. |
| Swap eyes | Reverses eye assignment. Use when depth appears inverted or the pair was supplied in reverse order. |
| Camera: Auto / W3 Wide / Nintendo 3DS | Nominal profiles using image-based vertical alignment. They do not supply measured lens distortion or metric depth. Select 3DS to identify that capture type in processing/cache metadata. |
| Load measured calibration | Loads camera intrinsics, distortion, and stereo calibration from JSON. Use only a calibration measured for your camera and capture mode. See the asset-format guide. |
| Stereo: CPU SGBM | Local CPU stereo matching. Works without NVIDIA hardware and needs no model download. |
| Stereo: GPU Fast-FoundationStereo | Optional neural geometry estimation in a separate NVIDIA environment. Often useful for difficult scenes, but not guaranteed to outperform CPU on every photo. |
| Geometry width | Resolution used to estimate disparity. Larger values cost more time and memory. Geometry is cached separately from preview/export sizes. Source images are not enlarged merely for analysis. |
| Refine uncertain geometry edges | Uses nearby color-compatible reliable geometry around uncertain boundaries. It does not turn inferred geometry into measured confidence. Compare with it off when diagnosing a difficult scene. |
| Anchor | A clicked subject depth held steady through the camera move. Choose texture inside the subject rather than a silhouette. |
| Use automatic anchor | Clears the clicked choice and selects a central foreground region when possible. |

## Motion and output quality

| Setting | What it does and when to use it |
| --- | --- |
| Smooth 3D motion | Synthesizes intermediate camera viewpoints from stereo geometry. Produces a forward-and-return loop. |
| Classic two-frame | Alternates the captured eyes, aligned to the anchor. Avoids intermediate-view synthesis. |
| Anaglyph 3D | Produces one red/cyan JPEG for red-left/cyan-right glasses. Temporal effects are disabled. |
| Anaglyph method | Dubois uses a generic color projection; Color directly assigns red/cyan channels; Monochrome reduces color distractions. Glasses/display crosstalk can still vary. |
| Sweep | How far the virtual camera moves between the captured eyes. Lower values often hide difficult occlusion artifacts. Zero stays at the center viewpoint; one reaches both endpoints. |
| Views | Number of synthesized views in the forward direction. More views at the same FPS makes a longer loop. The return does not duplicate turnaround frames. |
| FPS | Playback frames per second. Higher values make a fixed number of views play faster. |
| Classic hold | How long each original eye remains visible. Video timing rounds to whole frames. |
| MP4 cycles | How many complete loops are encoded into a video. GIF repeats indefinitely instead. |
| Ease at turnarounds | Slows camera motion near both ends of the sweep. Turn it off for constant-speed sampling. |
| Preview width | Final preview width after crop. Defaults to 960 px. Lower it for quicker iteration. |
| Preview frame AA | Renders smooth preview views on a 2× sampling grid before downsampling. Enabled by default; reduces jaggies but does not repair wrong geometry. It also works with effects. |
| Export width | Final export width after crop. Defaults to 1280 px. Original source colors are used rather than enlarging the preview. |
| Export supersampling | At 2× (default), renders at a larger source grid before final downsampling. Independent of preview AA. Tight crops, rotation, and native source size can limit the effective benefit. |
| Build preview | Reuses available analysis and renders current motion/edit/effect settings. Needed after motion or effect changes. |
| Pause / Play | Stops or resumes preview playback for inspection. |
| Cancel | Requests cancellation between CPU stages/frames; GPU inference and encoding can be terminated. A currently running OpenCV stage may finish first. |
| Export GIF / MP4 / JPEG | Writes the selected output format. MP4 requires FFmpeg with libx264; JPEG is specific to Anaglyph mode. |
| Also save stereo assets | Writes aligned eye images, disparity, reliability masks, and a settings manifest beside the media. Uses dense stereo even for Classic/Anaglyph. |

## Photo edit

| Control | What it does and when to use it |
| --- | --- |
| Exposure | Brightens or darkens in stops. Useful for overall light level. |
| Warmth | Shifts the shared color balance toward warm amber or cool blue. |
| Tint | Shifts balance between green and magenta. |
| Saturation | Controls color intensity. Zero makes monochrome; one preserves the original level. |
| Contrast | Expands or compresses tonal separation around a midtone pivot. |
| Sharpness | Adds detail enhancement. High values can emphasize halos and existing stereo artifacts. Zero leaves sharpening off. |
| Auto color balance | Estimates one correction from the left image and applies it consistently to all frames. |
| Rotation | Rotates the finished view clockwise, after stereo matching. Use for sideways photos; changing rotation resets crop. |
| Crop ratio | Constrains the crop to a selected aspect ratio, original ratio, or a free rectangle. |
| Crop selection | Drag a marquee, move it by dragging inside, or resize at corners. Before building, the frame is provisional. A gold outline indicates the sampled area when curvature is active. |
| Reset crop | Restores the available crop area under the selected ratio. |
| Reset photo edits | Restores color/crop defaults while retaining rotation. |
| Save / Load preset | Stores/restores this panel's photo values, crop, rotation, and aspect as JSON. It does not save source photographs, anchors, or camera calibration. |

## Geometry diagnostics

The display selector does not change rendering. Choose an eye for disparity/reliability maps, or a virtual viewpoint for projection diagnostics. Projection diagnostics use the analysis grid; they are not a pixel-exact export preview.

| View / control | Interpretation |
| --- | --- |
| Depth / disparity: raw | Estimated horizontal displacement between eyes. This is not distance in meters. Color is normalized for visibility. |
| Depth / disparity: refined | Geometry after the optional edge refinement. If refinement is off, it matches raw. |
| Match reliability | Left/right consistency evidence; dark means unreliable. It is not a calibrated probability or a depth map. |
| Refinement difference | Where refinement changed disparity and by how much. A prettier map alone does not establish better interpolation. |
| Left / right projection | Shows what each source contributes at the selected viewpoint. Helps identify which eye contains an artifact. |
| Source contribution | Red indicates left source, blue indicates right source, and mixtures indicate blending. |
| Projection disagreement | Highlights color differences between warped eyes. Differences can arise from occlusion, lighting, moving subjects, or bad correspondence. |
| Edge repair mask | Shows where localized unsupported-edge repair actually replaced samples. |
| Symmetric source selection | Normal fusion mode. Uses confidence, visibility, viewpoint weighting, and smooth endpoint convergence. |
| Legacy / left / right priority | Comparison modes to investigate source bias. These are not recommended defaults. |
| Repair unsupported edge conflicts only | Copies nearby trusted background samples into small, conflicting, unreliable regions. Leaves ordinary image interiors intact; does not apply a global blur. |
| Mismatched-edge blending rejection | Optional comparison that avoids blending disagreeing colors. Keep off unless comparison shows a benefit. |

## Effects

Each group has its own master toggle. Disabled values remain available and are saved in your Effects preset. **Bypass all effects** preserves your configuration while rendering clean output. **Reset all effects** restores disabled groups and default parameter values. Save/Load uses your own JSON files; no artistic presets are bundled.

Most effects run after final sizing; pixel parameters therefore refer to output pixels. Curvature is sampled before final resizing. Periodic settings use whole **cycles per complete loop**, not cycles per second. Seeded static is intentionally discontinuous. Excessive periodic rates are rejected instead of silently undersampling them.

### Stereo / delayed echo

Mixes an earlier playback sample into the current view for a double-exposure trail. Start with a low amount.

| Parameter | How to use it |
| --- | --- |
| Amount | Blend strength; zero leaves the image unchanged. |
| Delay (ms) | Earlier playback image mixed into the current image, wrapping around the loop. |
| Separation (px) | Additional horizontal displacement of the echo at final output size. |

### Band tearing

Displaces horizontal bands. Enable manually for deliberate tearing; band count controls how finely the image is divided.

| Parameter | How to use it |
| --- | --- |
| Amount | Blend strength; zero leaves the image unchanged. |
| Band count | Number of horizontal bands spanning the image. |
| Cycles / loop | Whole cycles across the complete forward-and-return loop. Zero is static. |

### RGB misconvergence

Separates red and blue channels for chromatic misconvergence. Small offsets are subtle; large offsets produce distinct colored contours.

| Parameter | How to use it |
| --- | --- |
| Split (px) | Red and blue displace in opposite directions at final output size. |
| Drift | Periodic change in channel separation across the playback loop. |
| Cycles / loop | Whole cycles across the complete forward-and-return loop. Zero is static. |

### Horizontal sync wobble

Bends horizontal position over time to suggest unstable synchronization.

| Parameter | How to use it |
| --- | --- |
| Amount | Blend strength; zero leaves the image unchanged. |
| Waves | Vertical frequency of horizontal displacement. |
| Cycles / loop | Whole cycles across the complete forward-and-return loop. Zero is static. |

### Composite video

Approximates analog color bleed and signal artifacts while treating color and luminance separately.

| Parameter | How to use it |
| --- | --- |
| Chroma bleed (px) | Horizontal blur of color while preserving luminance. |
| Amount | Blend strength; zero leaves the image unchanged. |
| Amount | Blend strength; zero leaves the image unchanged. |
| Amount | Blend strength; zero leaves the image unchanged. |
| Cycles / loop | Whole cycles across the complete forward-and-return loop. Zero is static. |

### Static / dropouts

Adds repeatable noise and signal interruptions. These are intentionally abrupt rather than smooth animation.

| Parameter | How to use it |
| --- | --- |
| Amount | Blend strength; zero leaves the image unchanged. |
| Dropout strength | Seeded bright/dark horizontal signal interruptions. |

### Film light leak · selected frames

Adds a spatially fixed light leak only to selected playback frames. Applied after persistence so it does not leave decay in unselected frames.

| Parameter | How to use it |
| --- | --- |
| Amount | Blend strength; zero leaves the image unchanged. |
| Hue | Fixed leak hue. |
| Spread | Fixed leak width relative to image size. |
| Horizontal position | Fixed position: -1 left edge, +1 right edge. |
| Vertical position | Fixed position: -1 top, +1 bottom. |
| Affected frames | Exact number of playback frames per loop, capped to loop length. |
| Burst length | Prefer consecutive frames in each event; final burst may be shorter. |
| Shuffle seed | Change which frames receive the leak; its position stays fixed. |

### Exposure flicker

Varies brightness periodically over the loop. Use low amounts for subtle instability.

| Parameter | How to use it |
| --- | --- |
| Amount | Exposure variation; low strengths preserve readable stereo motion. |
| Cycles / loop | Whole cycles across the complete forward-and-return loop. Zero is static. |

### Halation / phosphor bloom

Spreads bright highlights into a glow. Warmth moves the glow toward a film-halation appearance.

| Parameter | How to use it |
| --- | --- |
| Amount | Blend strength; zero leaves the image unchanged. |
| Radius (px) | Glow radius at final output size. |
| Warmth | Shift highlight glow toward warm film halation. |

### Phosphor persistence

Accumulates decaying light from earlier frames, with independent RGB decay. Circular history avoids a first-frame startup fade.

| Parameter | How to use it |
| --- | --- |
| Amount | Blend strength; zero leaves the image unchanged. |
| Red decay (ms) | Exponential decay time; initialized from circular loop history. |
| Green decay (ms) | Independent green phosphor decay. |
| Blue decay (ms) | Independent blue phosphor decay. |

### CRT curvature

Curves the image using a fixed safe inset inside the crop. It sacrifices field of view to avoid sampling a black exterior. The crop editor outlines the source area used.

| Parameter | How to use it |
| --- | --- |
| Amount | Blend strength; zero leaves the image unchanged. |

### Vignette / bezel shadow

Darkens the perimeter and simulates a shadow near the screen border.

| Parameter | How to use it |
| --- | --- |
| Amount | Blend strength; zero leaves the image unchanged. |
| Bezel shadow | Darkens the outer screen border. |

### Glass glare

Adds a glass-like highlight that can shift periodically.

| Parameter | How to use it |
| --- | --- |
| Amount | Blend strength; zero leaves the image unchanged. |
| Cycles / loop | Whole cycles across the complete forward-and-return loop. Zero is static. |

### Rolling scanlines

Overlays rolling horizontal scanlines at final pixel resolution, with optional refresh jitter.

| Parameter | How to use it |
| --- | --- |
| Amount | Blend strength; zero leaves the image unchanged. |
| Pitch (px) | Line spacing in final output pixels; independent of render supersampling. |
| Cycles / loop | Whole cycles across the complete forward-and-return loop. Zero is static. |
| Refresh jitter | Periodic variation in scanline travel speed; wraps smoothly across the loop. |

### Phosphor mask

Adds an RGB aperture grille or dot shadow mask at final pixel resolution. Use modest strength to retain brightness.

| Parameter | How to use it |
| --- | --- |
| Amount | Blend strength; zero leaves the image unchanged. |
| Pitch (px) | RGB triad spacing in final output pixels. |
| Shadow mask | 0: aperture grille; 1: staggered dot shadow mask. |

These are artistic approximations, not calibrated CRT hardware or physically measured film simulations.
