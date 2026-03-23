#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS

def list_images(folder: Path):
    if not folder.exists(): 
        return {}
    return {p.name: p for p in folder.iterdir() if p.is_file() and is_image(p)}

def safe_open_image(p: Path) -> Image.Image:
    # Handle EXIF rotation; convert to RGB to avoid mode mismatches
    img = Image.open(p)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img

def fit_to_height(img: Image.Image, target_h: int) -> Image.Image:
    if target_h is None:
        return img
    w, h = img.size
    if h == target_h:
        return img
    new_w = int(round(w * (target_h / h)))
    return img.resize((new_w, target_h), Image.LANCZOS)

def make_label_bar(left_text: str, right_text: str, total_width: int, pad=10, sep=30, font=None):
    # Create a single-line label bar with left_text on the left and right_text on the right
    if font is None:
        try:
            # Try a common font if available; fallback to default
            font = ImageFont.truetype("DejaVuSans.ttf", 35)
        except Exception:
            font = ImageFont.load_default(size=35)

    # Measure text
    dummy = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(dummy)
    lt_w, lt_h = draw.textbbox((0,0), left_text, font=font)[2:]
    rt_w, rt_h = draw.textbbox((0,0), right_text, font=font)[2:]
    text_h = max(lt_h, rt_h)
    bar_h = text_h + 2*pad

    bar = Image.new("RGB", (total_width, bar_h), "white")
    d = ImageDraw.Draw(bar)
    # Left text
    d.text((pad, pad), left_text, fill="black", font=font)
    # Right text, right-aligned
    d.text((total_width - pad - rt_w, pad), right_text, fill="black", font=font)
    # Optional thin divider in the middle of texts if there's space (purely visual)
    return bar

def stitch_side_by_side(imgL: Image.Image, imgR: Image.Image, label_bar: Image.Image, separator_px=6, bg="white"):
    h = max(imgL.height, imgR.height)
    w = imgL.width + separator_px + imgR.width
    canvas = Image.new("RGB", (w, label_bar.height + h), bg)
    # paste label bar
    canvas.paste(label_bar, (0, 0))
    # vertically center images under the bar
    y0 = label_bar.height + (h - imgL.height)//2
    y1 = label_bar.height + (h - imgR.height)//2
    canvas.paste(imgL, (0, y0))
    # separator
    sep_x0 = imgL.width
    ImageDraw.Draw(canvas).rectangle([sep_x0, label_bar.height, sep_x0 + separator_px - 1, label_bar.height + h], fill=(220,220,220))
    canvas.paste(imgR, (imgL.width + separator_px, y1))
    return canvas

def main():
    ap = argparse.ArgumentParser(description="Create labeled side-by-side comparisons of images from two directory trees.")
    ap.add_argument("dir_left", type=Path, help="First (left) root directory")
    ap.add_argument("dir_right", type=Path, help="Second (right) root directory")
    ap.add_argument("out_dir", type=Path, help="Output root directory")
    ap.add_argument("--height", type=int, default=1200, help="Target height for both images (default: 600). Use 0 to keep original heights.")
    ap.add_argument("--separator", type=int, default=6, help="Separator width in pixels between images (default: 6)")
    ap.add_argument("--format", type=str, default="jpg", choices=["jpg","png","webp"], help="Output image format")
    ap.add_argument("--quality", type=int, default=90, help="JPEG/WebP quality (default: 90)")
    args = ap.parse_args()

    dirL, dirR, out_root = args.dir_left, args.dir_right, args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    left_name = dirL.name
    right_name = dirR.name
    target_h = None if args.height == 0 else args.height

    # Gather subfolders (immediate children only). You can change to rglob if you need deeper recursion.
    subfoldersL = {p.name for p in dirL.iterdir() if p.is_dir()} if dirL.exists() else set()
    subfoldersR = {p.name for p in dirR.iterdir() if p.is_dir()} if dirR.exists() else set()
    all_subfolders = sorted(subfoldersL | subfoldersR)

    total_pairs = 0
    skipped_missing_sub = []
    skipped_missing_images = []

    for sub in all_subfolders:
        subL = dirL / sub
        subR = dirR / sub
        if not subL.exists() or not subR.exists():
            skipped_missing_sub.append(sub)
            continue

        imgsL = list_images(subL)
        imgsR = list_images(subR)
        common_files = sorted(set(imgsL.keys()) & set(imgsR.keys()))
        if not common_files:
            skipped_missing_sub.append(sub + " (no matching filenames)")
            continue

        out_sub = out_root / sub
        out_sub.mkdir(parents=True, exist_ok=True)

        for fname in common_files:
            pL = imgsL[fname]
            pR = imgsR[fname]
            try:
                imL = safe_open_image(pL)
                imR = safe_open_image(pR)
            except Exception as e:
                skipped_missing_images.append(f"{sub}/{fname} (open error: {e})")
                continue

            # resize to same height if requested
            if target_h is not None:
                imL = fit_to_height(imL, target_h)
                imR = fit_to_height(imR, target_h)

            # Build labels
            left_label  = f"{left_name}/{sub}/{fname}"
            right_label = f"{right_name}/{sub}/{fname}"
            total_w = imL.width + args.separator + imR.width
            label_bar = make_label_bar(left_label, right_label, total_w)

            # Stitch
            combined = stitch_side_by_side(imL, imR, label_bar, separator_px=args.separator)

            # Save
            stem = Path(fname).stem
            out_path = out_sub / f"{stem}_compare.{args.format}"
            save_kwargs = {}
            if args.format.lower() in ("jpg", "jpeg", "webp"):
                save_kwargs["quality"] = args.quality
            combined.save(out_path, **save_kwargs)
            total_pairs += 1

    print(f"Done. Wrote {total_pairs} side-by-side images into: {out_root}")
    if skipped_missing_sub:
        print("\nSubfolders skipped (missing on one side or no matches):")
        for s in skipped_missing_sub:
            print(f"  - {s}")
    if skipped_missing_images:
        print("\nImages skipped due to errors:")
        for s in skipped_missing_images:
            print(f"  - {s}")

if __name__ == "__main__":
    main()
