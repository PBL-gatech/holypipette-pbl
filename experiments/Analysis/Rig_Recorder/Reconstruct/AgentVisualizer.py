#!/usr/bin/env python3

import re
import sys
from pathlib import Path
from typing import Iterable, List

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    print("Pillow must be installed to run this script. Install it with `pip install pillow`.", file=sys.stderr)
    sys.exit(1)

# Hardcoded configuration
INPUT_DIR = Path(r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\AgentPath3")
OUTPUT_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\testing\AgentPath3\AgentPath.gif")
FPS = 30.0

def natural_sort_key(path: Path):
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", path.name)]

def collect_image_paths(directory: Path) -> List[Path]:
    supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in supported_exts),
        key=natural_sort_key,
    )

def load_frames(image_paths: Iterable[Path]):
    frames = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                frames.append(img.convert("RGBA"))
        except UnidentifiedImageError:
            print(f"Skipping unsupported image file: {path}", file=sys.stderr)
    return frames

def main():
    input_dir = INPUT_DIR.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if FPS <= 0:
        print("FPS must be a positive number.", file=sys.stderr)
        sys.exit(1)

    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        print(f"No supported images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    frames = load_frames(image_paths)
    if not frames:
        print("None of the images could be loaded successfully.", file=sys.stderr)
        sys.exit(1)

    output_path = OUTPUT_PATH.expanduser()
    output_path = output_path.with_suffix(".gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration_ms = max(1, int(round(1000.0 / FPS)))

    first_frame, *other_frames = frames
    first_frame.save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=other_frames,
        duration=duration_ms,
        loop=0,
        disposal=2,
    )

    print(
        f"Saved GIF to {output_path} with {len(frames)} frames at {FPS:.2f} fps "
        f"(frame duration {duration_ms} ms)."
    )

if __name__ == "__main__":
    main()
