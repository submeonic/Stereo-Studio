from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from .core import analyze, choose_anchor, read_pair
from .editing import PhotoEdits, ratio_crop
from .export import ffmpeg_available, write_animation, write_bundle
from .geometry import refine_analysis, render_grid
from .render import Motion, Sequence
from .temporal import TimelineSequence, validate
from .user_presets import SCHEMA, load_preset


def parser():
    p = argparse.ArgumentParser(description="Local stereo wigglegrams")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("gui", help="Open the desktop app")
    inspect = sub.add_parser("inspect", help="Inspect MPO or a stereo JPEG pair")
    inspect.add_argument("input")
    inspect.add_argument("--right")
    inspect.add_argument("--preview")
    render = sub.add_parser("render", help="Analyze and export a wigglegram")
    render.add_argument("input")
    render.add_argument("-o", "--output", required=True)
    render.add_argument("--right")
    render.add_argument("--swap", action="store_true")
    render.add_argument(
        "--width", type=int, default=960, help="Cached stereo geometry width; default 960"
    )
    render.add_argument(
        "--output-width", type=int, default=1280, help="Final width after crop; default 1280"
    )
    render.add_argument(
        "--supersample",
        type=int,
        choices=[1, 2],
        default=2,
        help="Render source images at this multiple of output width, then downsample",
    )
    render.add_argument(
        "--fusion", choices=["symmetric", "legacy", "left", "right"], default="symmetric"
    )
    render.add_argument("--no-edge-repair", action="store_true")
    render.add_argument("--refinement", choices=["raw", "guided"], default="guided")
    render.add_argument(
        "--effects-json", help="JSON object of independent effects and their parameters"
    )
    render.add_argument(
        "--edge-cleanup", action="store_true", help="Experimental mismatched-color blend rejection"
    )
    render.add_argument(
        "--profile", default="auto", help="auto, w3-wide, 3ds (nominal), or measured profile JSON"
    )
    render.add_argument("--backend", choices=["sgbm", "neural"], default="sgbm")
    render.add_argument("--mode", choices=["classic", "smooth", "anaglyph"], default="smooth")
    render.add_argument(
        "--anchor",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Pixel coordinates in aligned working-resolution left eye",
    )
    render.add_argument("--strength", type=float, default=0.65)
    render.add_argument("--views", type=int, default=25)
    render.add_argument("--fps", type=int, default=30)
    render.add_argument("--hold-ms", type=int, default=180)
    render.add_argument("--cycles", type=int, default=4)
    render.add_argument(
        "--linear", action="store_true", help="Use constant-speed instead of eased motion"
    )
    render.add_argument("--rotation", type=int, choices=[0, 90, 180, 270], default=0)
    render.add_argument("--antialias", choices=["off", "2x"], default="off")
    render.add_argument("--no-edge-cleanup", action="store_true")
    render.add_argument("--anaglyph-method", choices=["dubois", "color", "gray"], default="dubois")
    render.add_argument("--exposure", type=float, default=0, help="Exposure in stops, -4..4")
    render.add_argument("--temperature", type=float, default=0)
    render.add_argument("--tint", type=float, default=0)
    render.add_argument("--saturation", type=float, default=1)
    render.add_argument("--contrast", type=float, default=1)
    render.add_argument("--sharpness", type=float, default=0)
    render.add_argument("--auto-color", action="store_true")
    render.add_argument(
        "--crop",
        type=float,
        nargs=4,
        default=(0, 0, 1, 1),
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Normalized crop in the rotated common output area",
    )
    render.add_argument(
        "--aspect", type=float, help="Optional width/height ratio, e.g. 1 or 1.7777778"
    )
    render.add_argument("--assets", help="New directory for stereo asset bundle")
    render.add_argument("--overwrite", action="store_true")
    render.add_argument("--no-cache", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "gui":
            from .gui import main as gui_main

            return gui_main()
        if args.command == "inspect":
            left_image, r, info = read_pair(args.input, args.right)
            print(json.dumps(info, indent=2))
            if args.preview:
                import numpy as np
                from PIL import Image

                Image.fromarray(np.concatenate([left_image, r], axis=1)).save(args.preview)
            return 0
        motion = Motion(
            args.mode,
            args.strength,
            args.views,
            args.fps,
            args.hold_ms,
            args.cycles,
            not args.linear,
            args.rotation,
            antialias=args.antialias == "2x",
            edge_cleanup=args.edge_cleanup and not args.no_edge_cleanup,
            anaglyph_method=args.anaglyph_method,
            fusion=args.fusion,
            repair_edges=not args.no_edge_repair,
        )
        motion.validate()
        edits = PhotoEdits(
            exposure=args.exposure,
            temperature=args.temperature,
            tint=args.tint,
            saturation=args.saturation,
            contrast=args.contrast,
            sharpness=args.sharpness,
            auto_color=args.auto_color,
            crop=tuple(args.crop),
        )
        settings = json.loads(Path(args.effects_json).read_text()) if args.effects_json else {}
        if isinstance(settings, dict) and settings.get("schema") == SCHEMA:
            state = load_preset(args.effects_json, "effects")
            settings = (
                {}
                if state["bypass"]
                else {name: state["parameters"][name] for name in state["enabled"]}
            )
        settings = validate(settings)
        output_width = args.output_width or args.width
        if not 160 <= output_width <= 4096 or output_width % 2:
            raise ValueError("Output width must be an even number between 160 and 4096.")
        if args.supersample * output_width > 4096:
            raise ValueError(
                "Render width cannot exceed 4096; lower output width or supersampling."
            )
        if args.supersample > 1 and motion.antialias:
            raise ValueError("Use --supersample 2 or --antialias 2x, not both.")
        edits.validate()
        if args.aspect is not None and not 0.1 <= args.aspect <= 10:
            raise ValueError("Aspect ratio must be between 0.1 and 10.")
        if args.mode == "anaglyph" and settings:
            raise ValueError("Anaglyph requires temporal effects to be disabled.")
        output = Path(args.output)
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output}")
        if args.assets and Path(args.assets).exists():
            raise FileExistsError(f"Asset bundle exists: {args.assets}")
        allowed = (".jpg", ".jpeg") if args.mode == "anaglyph" else (".gif", ".mp4")
        if output.suffix.lower() not in allowed:
            raise ValueError("Anaglyph needs .jpg; motion needs .gif or .mp4.")
        if output.suffix.lower() == ".mp4":
            ffmpeg_available()

        def progress(message):
            print(message, file=sys.stderr, flush=True)

        backend = (
            "sparse" if args.mode in ("classic", "anaglyph") and not args.assets else args.backend
        )
        a = analyze(
            args.input,
            args.right,
            args.width,
            args.swap,
            args.profile,
            backend,
            cache_dir=None if args.no_cache else ".cache/wigglegram",
            progress=progress,
        )
        a = refine_analysis(
            a, args.refinement == "guided", cache_dir=None if args.no_cache else ".cache/wigglegram"
        )
        anchor = choose_anchor(a, args.anchor)
        a, anchor = render_grid(
            a,
            anchor,
            args.input,
            args.right,
            args.swap,
            args.profile,
            output_width * args.supersample,
        )
        progress(anchor.message)
        with tempfile.TemporaryDirectory(prefix="wigglegram-") as temp:
            seq = Sequence(
                a, anchor, motion, temp, progress, edits=edits, output_width=output_width
            )
            if args.aspect is not None:
                edits.crop = ratio_crop(edits.crop, seq.full_frame(0).shape, args.aspect)
                seq.set_edits(edits)
            seq.metadata["export_supersampling_requested"] = args.supersample
            if settings:
                seq = TimelineSequence(seq, settings, Path(temp) / "effects", progress)
            result = write_animation(seq, output, args.overwrite, progress)
            bundle = write_bundle(a, seq, args.assets) if args.assets else None
            report = {
                "output": str(result),
                "assets": str(bundle) if bundle else None,
                "anchor": asdict(anchor),
                "motion": asdict(motion),
                "render": seq.metadata,
                "analysis": a.metadata,
            }
            print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyboardInterrupt) as exc:
        print(f"wigglegram: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
