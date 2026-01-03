#!/usr/bin/env python3
"""Convert images to 5-level grayscale."""

from pathlib import Path
from PIL import Image
import numpy as np

def to_5_level_grayscale(img_path: Path, out_dir: Path) -> None:
    """Convert an image to 5-level grayscale and save it."""
    img = Image.open(img_path).convert("L")  # Convert to grayscale
    arr = np.array(img, dtype=np.float32)

    # Quantize to 5 levels (0, 64, 128, 191, 255)
    # Map 0-255 to 0-4, then back to 0-255
    quantized = np.round(arr / 255 * 4) * (255 / 4)
    quantized = quantized.astype(np.uint8)

    result = Image.fromarray(quantized, mode="L")
    out_path = out_dir / img_path.name
    result.save(out_path)
    print(f"Saved: {out_path}")

def main():
    src_dir = Path(__file__).parent
    out_dir = src_dir / "out"
    out_dir.mkdir(exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

    for img_path in src_dir.iterdir():
        if img_path.suffix.lower() in extensions and img_path.is_file():
            to_5_level_grayscale(img_path, out_dir)

    print("Done!")

if __name__ == "__main__":
    main()
