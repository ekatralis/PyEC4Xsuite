#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def natural_key(s):
    """
    Natural sort key: splits strings into chunks of digits and non-digits
    so 'img2' < 'img10'.
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_images(folder):
    paths = [p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(paths, key=lambda p: natural_key(p.name))

def parse_size(size_str):
    if size_str is None:
        return None
    m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", size_str)
    if not m:
        raise argparse.ArgumentTypeError("Size must be like 1920x1080")
    return int(m.group(1)), int(m.group(2))

def fit_pad(img, target_w, target_h, pad_color=(0, 0, 0)):
    """
    Letterbox an image to target size, preserving aspect ratio:
    - scales it to fit within target,
    - pads equally to reach exact (target_w, target_h).
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create padded canvas
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = resized
    return canvas

def main():
    ap = argparse.ArgumentParser(description="Turn a folder of images into an MP4 (one frame per image).")
    ap.add_argument("folder", help="Path to folder containing images")
    ap.add_argument("-o", "--output", default="output.mp4", help="Output MP4 file (default: output.mp4)")
    ap.add_argument("--fps", type=float, default=24, help="Frames per second (default: 24)")
    ap.add_argument("--size", type=parse_size, default=None, help="Output size WxH (e.g. 1920x1080). Default: use first image size")
    ap.add_argument("--pad-color", default="0,0,0", help="Padding color as R,G,B (default: 0,0,0)")
    ap.add_argument("--reverse", action="store_true", help="Reverse frame order")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    images = list_images(folder)
    if not images:
        raise SystemExit(f"No images found in {folder} (supported: {', '.join(sorted(IMG_EXTS))})")

    if args.reverse:
        images = list(reversed(images))

    # Determine output size
    if args.size:
        out_w, out_h = args.size
    else:
        # Use first image’s size
        with Image.open(images[0]) as im:
            out_w, out_h = im.size

    # Parse pad color
    try:
        r, g, b = (int(x) for x in args.pad_color.split(","))
        pad_color = (b, g, r)  # OpenCV uses BGR
    except Exception:
        raise SystemExit("Invalid --pad-color. Use R,G,B (e.g., 0,0,0 or 255,255,255)")

    # Set up writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely supported; if issues, try 'avc1'
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit("Failed to open video writer. Try a different filename or codec.")

    total = 0
    for p in images:
        # Read with OpenCV (handles a lot of formats)
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: could not read {p}, skipping.")
            continue
        frame = fit_pad(img, out_w, out_h, pad_color=pad_color)
        writer.write(frame)
        total += 1

    writer.release()
    if total == 0:
        raise SystemExit("No frames written. Check that your images are readable.")
    print(f"Done: wrote {total} frames to {args.output} at {args.fps} fps ({out_w}x{out_h}).")

if __name__ == "__main__":
    main()
