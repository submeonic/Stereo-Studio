"""Effects on chronological playback samples, after photo edits and final sizing.

Cycle counts are integer cycles per loop. Persistence has the exact circular
initial condition; independent seeded noise is reproducible, not temporally smooth.
"""

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from .editing import linear_rgb, srgb


# Each parameter: label, minimum, maximum, default, step, help.
def param(label, low, high, default, step, help):
    return (label, low, high, default, step, help)


AMOUNT = param("Amount", 0, 1, 0.3, 0.01, "Blend strength; zero leaves the image unchanged.")
CYCLES = param(
    "Cycles / loop",
    0,
    12,
    1,
    1,
    "Whole cycles across the complete forward-and-return loop. Zero is static.",
)
SPECS = {
    "echo": (
        "Stereo / delayed echo",
        {
            "amount": AMOUNT,
            "delay": param(
                "Delay (ms)",
                0,
                2000,
                120,
                10,
                "Earlier playback image mixed into the current image, wrapping around the loop.",
            ),
            "separation": param(
                "Separation (px)",
                -40,
                40,
                0,
                1,
                "Additional horizontal displacement of the echo at final output size.",
            ),
        },
    ),
    "tearing": (
        "Band tearing",
        {
            "amount": AMOUNT,
            "bands": param(
                "Band count", 1, 120, 28, 1, "Number of horizontal bands spanning the image."
            ),
            "cycles": CYCLES,
        },
    ),
    "rgb": (
        "RGB misconvergence",
        {
            "offset": param(
                "Split (px)",
                0,
                40,
                3,
                0.25,
                "Red and blue displace in opposite directions at final output size.",
            ),
            "drift": param(
                "Drift",
                0,
                1,
                0.3,
                0.01,
                "Periodic change in channel separation across the playback loop.",
            ),
            "cycles": CYCLES,
        },
    ),
    "wobble": (
        "Horizontal sync wobble",
        {
            "amount": AMOUNT,
            "waves": param("Waves", 1, 40, 6, 1, "Vertical frequency of horizontal displacement."),
            "cycles": CYCLES,
        },
    ),
    "composite": (
        "Composite video",
        {
            "bleed": param(
                "Chroma bleed (px)",
                0,
                12,
                3,
                0.25,
                "Horizontal blur of color while preserving luminance.",
            ),
            "sharpness": AMOUNT,
            "ringing": AMOUNT,
            "crawl": AMOUNT,
            "cycles": CYCLES,
        },
    ),
    "static": (
        "Static / dropouts",
        {
            "amount": AMOUNT,
            "dropouts": param(
                "Dropout strength",
                0,
                1,
                0.2,
                0.01,
                "Seeded bright/dark horizontal signal interruptions.",
            ),
        },
    ),
    "leak": (
        "Film light leak · selected frames",
        {
            "amount": AMOUNT,
            "hue": param("Hue", 0, 359, 25, 1, "Fixed leak hue."),
            "spread": param(
                "Spread", 0.05, 1, 0.35, 0.01, "Fixed leak width relative to image size."
            ),
            "x": param(
                "Horizontal position",
                -1.5,
                1.5,
                1,
                0.05,
                "Fixed position: -1 left edge, +1 right edge.",
            ),
            "y": param(
                "Vertical position", -1.5, 1.5, 0.2, 0.05, "Fixed position: -1 top, +1 bottom."
            ),
            "frames": param(
                "Affected frames",
                0,
                600,
                3,
                1,
                "Exact number of playback frames per loop, capped to loop length.",
            ),
            "burst": param(
                "Burst length",
                1,
                60,
                1,
                1,
                "Prefer consecutive frames in each event; final burst may be shorter.",
            ),
            "seed": param(
                "Shuffle seed",
                0,
                9999,
                42,
                1,
                "Change which frames receive the leak; its position stays fixed.",
            ),
        },
    ),
    "flicker": (
        "Exposure flicker",
        {
            "amount": param(
                "Amount",
                0,
                1,
                0.08,
                0.01,
                "Exposure variation; low strengths preserve readable stereo motion.",
            ),
            "cycles": CYCLES,
        },
    ),
    "bloom": (
        "Halation / phosphor bloom",
        {
            "amount": AMOUNT,
            "radius": param("Radius (px)", 0.5, 40, 6, 0.5, "Glow radius at final output size."),
            "warmth": param(
                "Warmth", 0, 1, 0.3, 0.01, "Shift highlight glow toward warm film halation."
            ),
        },
    ),
    "persistence": (
        "Phosphor persistence",
        {
            "amount": AMOUNT,
            "red_ms": param(
                "Red decay (ms)",
                1,
                2000,
                100,
                5,
                "Exponential decay time; initialized from circular loop history.",
            ),
            "green_ms": param(
                "Green decay (ms)", 1, 2000, 160, 5, "Independent green phosphor decay."
            ),
            "blue_ms": param("Blue decay (ms)", 1, 2000, 80, 5, "Independent blue phosphor decay."),
        },
    ),
    "curvature": ("CRT curvature", {"amount": AMOUNT}),
    "vignette": (
        "Vignette / bezel shadow",
        {
            "amount": AMOUNT,
            "bezel": param("Bezel shadow", 0, 1, 0.25, 0.01, "Darkens the outer screen border."),
        },
    ),
    "glare": ("Glass glare", {"amount": AMOUNT, "cycles": CYCLES}),
    "scanlines": (
        "Rolling scanlines",
        {
            "amount": AMOUNT,
            "pitch": param(
                "Pitch (px)",
                2,
                12,
                3,
                0.25,
                "Line spacing in final output pixels; independent of render supersampling.",
            ),
            "cycles": CYCLES,
            "jitter": param(
                "Refresh jitter",
                0,
                1,
                0,
                0.01,
                "Periodic variation in scanline travel speed; wraps smoothly across the loop.",
            ),
        },
    ),
    "mask": (
        "Phosphor mask",
        {
            "amount": AMOUNT,
            "pitch": param("Pitch (px)", 3, 12, 3, 1, "RGB triad spacing in final output pixels."),
            "dots": param(
                "Shadow mask", 0, 1, 0, 1, "0: aperture grille; 1: staggered dot shadow mask."
            ),
        },
    ),
}


def validate(settings):
    if not isinstance(settings, dict) or set(settings) - set(SPECS):
        raise ValueError("Unknown effect. Use an object of effect names and parameter objects.")
    result = {}
    for name, values in settings.items():
        specs = SPECS[name][1]
        if not isinstance(values, dict) or set(values) - set(specs):
            raise ValueError(f"Unknown parameters for {name}.")
        result[name] = {}
        for key, (_, low, high, default, step, _) in specs.items():
            value = values.get(key, default)
            if (
                not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{name}.{key} must be between {low} and {high}.")
            if step == 1 and int(value) != value:
                raise ValueError(f"{name}.{key} requires a whole number.")
            result[name][key] = value
    return result


def spatial_effects(image, settings, phase, index, seed, screen=False, leak_active=True):
    phase = phase % 1.0
    f = image.astype(np.float32) / 255
    h, w = f.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    nx, ny = (x + 0.5) / w * 2 - 1, (y + 0.5) / h * 2 - 1

    def remap(value, xx, yy):
        return cv2.remap(
            value,
            xx.astype(np.float32),
            yy.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    for name in SPECS:
        if name not in settings or name in ("echo", "persistence"):
            continue
        is_screen = name in ("leak", "curvature", "vignette", "glare", "scanlines", "mask")
        if is_screen != screen:
            continue
        p = settings[name]
        t = phase * 2 * np.pi * p.get("cycles", 1)
        amount = p.get("amount", 0)
        if name == "tearing":
            band = np.minimum(p["bands"] - 1, np.floor(y / h * p["bands"]))
            shift = np.sin(band * 1.713 + t) * (np.sin(band * 2.399) > 0.1)
            f = remap(f, x + shift * amount * w * 0.05, y)
        elif name == "rgb":
            shift = p["offset"] * (1 + p["drift"] * np.sin(t))
            f[:, :, 0] = remap(f[:, :, 0], x - shift, y)
            f[:, :, 2] = remap(f[:, :, 2], x + shift, y)
        elif name == "wobble":
            f = remap(f, x + amount * 0.02 * w * np.sin(ny * np.pi * p["waves"] + t), y)
        elif name == "composite":
            c = cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)
            if p["bleed"]:
                c[:, :, 1:] = cv2.GaussianBlur(c[:, :, 1:], (0, 1), p["bleed"])
            blur = cv2.GaussianBlur(c[:, :, 0], (0, 1), 1)
            detail = c[:, :, 0] - blur
            c[:, :, 0] += p["sharpness"] * detail
            c[:, :, 0] += p["ringing"] * (np.roll(detail, 2, axis=1) - np.roll(detail, -2, axis=1))
            c[:, :, 1] += (
                p["crawl"]
                * 0.08
                * np.sin(x * np.pi / 2 + y * np.pi + t)
                * np.clip(abs(detail) * 5, 0, 1)
            )
            f = cv2.cvtColor(c, cv2.COLOR_YCrCb2RGB)
        elif name == "static":
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            f += rng.normal(0, amount * 0.12, f.shape).astype(np.float32)
            rows = rng.random(h) < 0.02
            f[rows] += (rng.random((rows.sum(), 1, 1)).astype(np.float32) - 0.5) * p["dropouts"]
        elif name == "leak":
            if not leak_active:
                continue
            cx, cy = p["x"], p["y"]
            glow = np.exp(-((nx - cx) ** 2 + (ny - cy) ** 2) / (2 * p["spread"] ** 2))
            color = cv2.cvtColor(np.float32([[[p["hue"], 0.8, 1]]]), cv2.COLOR_HSV2RGB)[0, 0]
            f = 1 - (1 - f) * (1 - amount * glow[:, :, None] * color)
        elif name == "flicker":
            f *= 2 ** (amount * np.sin(t))
        elif name == "bloom":
            bright = np.maximum(f - 0.65, 0)
            glow = cv2.GaussianBlur(bright, (0, 0), p["radius"])
            f += amount * glow * np.array([1 + p["warmth"], 1, 1 - p["warmth"]])
        elif name == "curvature":
            from .curvature import curve

            f = curve(f, amount)
        elif name == "vignette":
            edge = np.minimum(1 - abs(nx), 1 - abs(ny))
            f *= (
                np.clip(1 - amount * 0.55 * (nx * nx + ny * ny), 0, 1)
                * (1 - p["bezel"] * np.exp(-edge * 35))
            )[:, :, None]
        elif name == "glare":
            glare = np.exp(-(((nx + ny * 0.4 - 0.6 - 0.1 * np.sin(t)) / 0.18) ** 2)) * 0.15 * amount
            f = 1 - (1 - f) * (1 - glare[:, :, None])
        elif name == "scanlines":
            t += p["jitter"] * np.sin(3 * t)
            # Pixel footprint integration attenuates high-frequency line modulation.
            wave = 0.5 + 0.5 * np.sinc(1 / p["pitch"]) * np.cos(2 * np.pi * y / p["pitch"] - t)
            f *= (1 - amount * wave)[:, :, None]
        elif name == "mask":
            stripe = np.floor(x / p["pitch"] * 3).astype(int) % 3
            m = np.eye(3, dtype=np.float32)[stripe]
            if p["dots"]:
                m *= (0.5 + 0.5 * np.cos(y * 2 * np.pi / p["pitch"] + (stripe % 2) * np.pi))[
                    :, :, None
                ]
            f *= 1 - amount * (1 - m)
    return np.rint(f.clip(0, 1) * 255).astype(np.uint8)


class TimelineSequence:
    """Disk-backed postprocess; one frame per actual playback time, bounded RAM."""

    timeline_expanded = True

    def __init__(
        self, base, settings, directory, progress=lambda _: None, cancelled=lambda: False, seed=42
    ):
        self.base = base
        self.settings = validate(settings)
        if base.motion.mode == "anaglyph" and self.settings:
            raise ValueError("Temporal effects are disabled for anaglyph eye-channel safety.")
        self.screen_settings = deepcopy(self.settings)
        if hasattr(base, "curvature_amount"):
            base.curvature_amount = self.settings.get("curvature", {}).get("amount", 0)
            self.screen_settings.pop("curvature", None)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.motion, self.anchor, self.crop = base.motion, base.anchor, base.crop
        self.positions = base.positions
        self.edits = base.edits
        self.metadata = deepcopy(base.metadata)
        self.metadata.update(
            effects=self.settings,
            effects_seed=seed,
            effect_grid="final output pixels",
            temporal_order="chronological; circular history",
        )
        order = list(base.indices)
        if self.motion.mode == "classic":
            holds = max(1, round(self.motion.hold_ms * self.motion.fps / 1000))
            order = [j for j in order for _ in range(holds)]
        n = len(order)
        # Limit periodic effects to the representable temporal frequency.
        for name, p in self.settings.items():
            if p.get("cycles", 0) > n // 2:
                raise ValueError(
                    f"{name}: lower cycles / loop to at most {n // 2}, or add more views/hold frames."
                )
        self.indices = list(range(n))
        self.durations_ms = [1000 / self.motion.fps] * n
        self.metadata["timeline_frames"] = n
        self.leak_frames = leak_schedule(n, self.settings.get("leak"))
        self.metadata["light_leak_frames"] = sorted(self.leak_frames)
        if "curvature" in self.settings:
            from .curvature import safe_scale

            self.metadata["curvature_safe_scale"] = safe_scale(self.settings["curvature"]["amount"])
            self.metadata["curvature_stage"] = "selected crop, safe inset, before output resize"
        for i, j in enumerate(order):
            if cancelled():
                raise InterruptedError("Cancelled")
            progress(f"Applying temporal effects {i + 1}/{n}")
            frame = base.frame(j)
            if "echo" in self.settings:
                p = self.settings["echo"]
                delay = p["delay"] * self.motion.fps / 1000
                sample = (i - delay) % n
                lo, fraction = int(np.floor(sample)), sample % 1
                echo = base.frame(order[lo]).astype(np.float32) * (1 - fraction)
                echo += base.frame(order[(lo + 1) % n]).astype(np.float32) * fraction
                matrix = np.float32([[1, 0, p["separation"]], [0, 1, 0]])
                echo = cv2.warpAffine(
                    echo,
                    matrix,
                    (frame.shape[1], frame.shape[0]),
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                frame = (
                    np.rint(frame * (1 - p["amount"]) + echo * p["amount"])
                    .clip(0, 255)
                    .astype(np.uint8)
                )
            np.save(
                self.directory / f"{i:04d}-signal.npy",
                spatial_effects(frame, self.settings, i / n, i, seed),
            )
        persistence = self.settings.get("persistence")
        state = None
        if persistence:
            tau = np.array([persistence[k] for k in ("red_ms", "green_ms", "blue_ms")])
            decay = np.exp(-1000 / self.motion.fps / tau).astype(np.float32)
            state = np.zeros_like(linear_rgb(np.load(self.directory / "0000-signal.npy")))
            for i in range(n):
                if cancelled():
                    raise InterruptedError("Cancelled")
                state = state * decay + linear_rgb(
                    np.load(self.directory / f"{i:04d}-signal.npy")
                ) * (1 - decay)
            state /= 1 - decay**n
        for i in range(n):
            if cancelled():
                raise InterruptedError("Cancelled")
            frame = np.load(self.directory / f"{i:04d}-signal.npy")
            if persistence:
                light = linear_rgb(frame)
                state = state * decay + light * (1 - decay)
                frame = srgb(light * (1 - persistence["amount"]) + state * persistence["amount"])
            frame = spatial_effects(
                frame,
                self.screen_settings,
                i / n,
                i,
                seed,
                screen=True,
                leak_active=i in self.leak_frames,
            )
            np.save(self.directory / f"{i:04d}.npy", frame)
            (self.directory / f"{i:04d}-signal.npy").unlink()

    def frame(self, index):
        return np.load(self.directory / f"{index:04d}.npy")

    def full_frame(self, index):
        return self.base.full_frame(index)

    def set_edits(self, edits):
        if edits != self.edits:
            raise ValueError(
                "Photo changes require rebuilding the effects preview; cached geometry is reused."
            )


def leak_schedule(count, settings):
    if not settings:
        return set()
    target = min(count, int(settings["frames"]))
    chosen = set()
    rng = np.random.default_rng(int(settings["seed"]))
    for start in rng.permutation(count):
        for offset in range(int(settings["burst"])):
            if len(chosen) == target:
                return chosen
            chosen.add((int(start) + offset) % count)
    return chosen
