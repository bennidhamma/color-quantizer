#!/usr/bin/env python3
"""Convert images to N-color palette using k-means++ in LAB color space."""

from pathlib import Path
from PIL import Image
import numpy as np

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to LAB color space."""
    # Normalize RGB to 0-1
    rgb_norm = rgb.astype(np.float32) / 255.0

    # Linearize (inverse sRGB companding)
    mask = rgb_norm > 0.04045
    rgb_linear = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

    # RGB to XYZ (sRGB D65)
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = rgb_linear @ matrix.T

    # XYZ to LAB (D65 white point)
    xyz_ref = np.array([0.95047, 1.0, 1.08883])
    xyz_norm = xyz / xyz_ref

    epsilon = 0.008856
    kappa = 903.3
    mask = xyz_norm > epsilon
    f = np.where(mask, xyz_norm ** (1/3), (kappa * xyz_norm + 16) / 116)

    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])

    return np.stack([L, a, b], axis=1)

def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB to RGB color space."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    # LAB to XYZ
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    epsilon = 0.008856
    kappa = 903.3

    xyz = np.zeros_like(lab)
    mask_x = fx ** 3 > epsilon
    mask_y = L > kappa * epsilon
    mask_z = fz ** 3 > epsilon

    xyz[:, 0] = np.where(mask_x, fx ** 3, (116 * fx - 16) / kappa)
    xyz[:, 1] = np.where(mask_y, fy ** 3, L / kappa)
    xyz[:, 2] = np.where(mask_z, fz ** 3, (116 * fz - 16) / kappa)

    # D65 white point
    xyz_ref = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz * xyz_ref

    # XYZ to linear RGB
    matrix = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    rgb_linear = xyz @ matrix.T

    # Clamp and apply sRGB companding
    rgb_linear = np.clip(rgb_linear, 0, 1)
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * (rgb_linear ** (1/2.4)) - 0.055, 12.92 * rgb_linear)

    return np.clip(rgb * 255, 0, 255).astype(np.uint8)

def kmeans_plusplus_init(pixels: np.ndarray, n_colors: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ initialization."""
    n_pixels = len(pixels)
    centroids = [pixels[rng.integers(n_pixels)]]

    for _ in range(1, n_colors):
        dists = np.min([np.sum((pixels - c) ** 2, axis=1) for c in centroids], axis=0)
        probs = dists / dists.sum()
        idx = rng.choice(n_pixels, p=probs)
        centroids.append(pixels[idx])

    return np.array(centroids, dtype=np.float32)

def kmeans_lab(pixels_lab: np.ndarray, n_colors: int, max_iter: int = 30) -> np.ndarray:
    """K-means clustering in LAB space."""
    rng = np.random.default_rng(42)
    centroids = kmeans_plusplus_init(pixels_lab, n_colors, rng)

    for _ in range(max_iter):
        distances = np.linalg.norm(pixels_lab[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            pixels_lab[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(n_colors)
        ])

        if np.allclose(centroids, new_centroids, atol=0.5):
            break
        centroids = new_centroids

    return centroids

def to_n_colors(img_path: Path, out_dir: Path, n_colors: int = 5) -> None:
    """Convert an image to N colors and save it."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    h, w, c = arr.shape
    pixels_rgb = arr.reshape(-1, 3)

    # Convert to LAB for clustering
    pixels_lab = rgb_to_lab(pixels_rgb)

    # Sample pixels for faster clustering
    if len(pixels_lab) > 50000:
        sample_idx = np.random.default_rng(42).choice(len(pixels_lab), 50000, replace=False)
        sample_lab = pixels_lab[sample_idx]
    else:
        sample_lab = pixels_lab

    # Cluster in LAB space
    palette_lab = kmeans_lab(sample_lab, n_colors)
    palette_rgb = lab_to_rgb(palette_lab)
    print(f"  Palette: {[tuple(c) for c in palette_rgb]}")

    # Map each pixel to nearest palette color (in LAB space)
    distances = np.linalg.norm(pixels_lab[:, None] - palette_lab[None, :], axis=2)
    labels = np.argmin(distances, axis=1)
    quantized = palette_rgb[labels].reshape(h, w, c)

    result = Image.fromarray(quantized, mode="RGB")
    out_path = out_dir / img_path.name
    result.save(out_path)
    print(f"  Saved: {out_path}")

def main():
    import sys
    n_colors = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    src_dir = Path(__file__).parent
    out_dir = src_dir / f"out_color_{n_colors}"
    out_dir.mkdir(exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

    print(f"Converting images to {n_colors} colors (LAB space clustering)...")
    for img_path in sorted(src_dir.iterdir()):
        if img_path.suffix.lower() in extensions and img_path.is_file():
            print(f"Processing: {img_path.name}")
            to_n_colors(img_path, out_dir, n_colors)

    print("Done!")

if __name__ == "__main__":
    main()
