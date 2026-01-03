#!/usr/bin/env python3
"""Color quantizer GUI - select photo, preview options, and save."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
import threading

# Color conversion functions
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb_norm = rgb.astype(np.float32) / 255.0
    mask = rgb_norm > 0.04045
    rgb_linear = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = rgb_linear @ matrix.T
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
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    epsilon = 0.008856
    kappa = 903.3
    xyz = np.zeros_like(lab)
    xyz[:, 0] = np.where(fx ** 3 > epsilon, fx ** 3, (116 * fx - 16) / kappa)
    xyz[:, 1] = np.where(L > kappa * epsilon, fy ** 3, L / kappa)
    xyz[:, 2] = np.where(fz ** 3 > epsilon, fz ** 3, (116 * fz - 16) / kappa)
    xyz = xyz * np.array([0.95047, 1.0, 1.08883])
    matrix = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    rgb_linear = np.clip(xyz @ matrix.T, 0, 1)
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * (rgb_linear ** (1/2.4)) - 0.055, 12.92 * rgb_linear)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)

def kmeans_plusplus_init(pixels: np.ndarray, n_colors: int, rng: np.random.Generator) -> np.ndarray:
    n_pixels = len(pixels)
    centroids = [pixels[rng.integers(n_pixels)]]
    for _ in range(1, n_colors):
        dists = np.min([np.sum((pixels - c) ** 2, axis=1) for c in centroids], axis=0)
        probs = dists / dists.sum()
        idx = rng.choice(n_pixels, p=probs)
        centroids.append(pixels[idx])
    return np.array(centroids, dtype=np.float32)

def kmeans_lab(pixels_lab: np.ndarray, n_colors: int, max_iter: int = 20) -> np.ndarray:
    rng = np.random.default_rng(42)
    centroids = kmeans_plusplus_init(pixels_lab, n_colors, rng)
    for _ in range(max_iter):
        distances = np.linalg.norm(pixels_lab[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            pixels_lab[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(n_colors)
        ])
        if np.allclose(centroids, new_centroids, atol=1.0):
            break
        centroids = new_centroids
    return centroids

def quantize_image(img: Image.Image, n_colors: int, grayscale: bool = False, fast: bool = False) -> Image.Image:
    """Quantize image to n colors."""
    if grayscale:
        img_gray = img.convert("L")
        arr = np.array(img_gray)
        levels = n_colors
        lut = [int(round(i / 255 * (levels - 1)) * 255 / (levels - 1)) for i in range(256)]
        result = img_gray.point(lut)
        return result.convert("RGB")

    # Resize for faster processing if fast mode
    work_img = img.convert("RGB")
    if fast and max(work_img.size) > 800:
        work_img.thumbnail((800, 800), Image.Resampling.NEAREST)

    arr = np.array(work_img)
    h, w, c = arr.shape
    pixels_rgb = arr.reshape(-1, 3)
    pixels_lab = rgb_to_lab(pixels_rgb)

    # Sample for clustering - smaller sample for speed
    sample_size = 10000 if fast else 30000
    if len(pixels_lab) > sample_size:
        sample_idx = np.random.default_rng(42).choice(len(pixels_lab), sample_size, replace=False)
        sample_lab = pixels_lab[sample_idx]
    else:
        sample_lab = pixels_lab

    palette_lab = kmeans_lab(sample_lab, n_colors)
    palette_rgb = lab_to_rgb(palette_lab)

    # For final output, process original size
    if fast:
        # Apply to working image only
        distances = np.linalg.norm(pixels_lab[:, None] - palette_lab[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        quantized = palette_rgb[labels].reshape(h, w, c)
        return Image.fromarray(quantized, mode="RGB"), palette_rgb
    else:
        # Apply to full resolution
        full_arr = np.array(img.convert("RGB"))
        fh, fw, fc = full_arr.shape
        full_pixels = full_arr.reshape(-1, 3)
        full_lab = rgb_to_lab(full_pixels)
        distances = np.linalg.norm(full_lab[:, None] - palette_lab[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        quantized = palette_rgb[labels].reshape(fh, fw, fc)
        return Image.fromarray(quantized, mode="RGB"), palette_rgb


class ColorQuantizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Quantizer")
        self.root.geometry("1000x700")

        self.original_image = None
        self.preview_image = None
        self.current_result = None
        self.current_palette = None
        self.image_path = None
        self.processing = False

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top controls
        controls = ttk.Frame(main_frame)
        controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(controls, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=5)

        ttk.Label(controls, text="Colors:").pack(side=tk.LEFT, padx=(20, 5))
        self.n_colors = tk.IntVar(value=5)
        self.color_slider = ttk.Scale(controls, from_=2, to=16, variable=self.n_colors,
                                       orient=tk.HORIZONTAL, length=150, command=self.on_slider_change)
        self.color_slider.pack(side=tk.LEFT, padx=5)
        self.color_label = ttk.Label(controls, text="5")
        self.color_label.pack(side=tk.LEFT, padx=5)

        self.grayscale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Grayscale", variable=self.grayscale_var,
                        command=self.update_preview).pack(side=tk.LEFT, padx=20)

        ttk.Button(controls, text="Preview", command=self.update_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Save Full Res", command=self.save_image).pack(side=tk.LEFT, padx=5)

        # Image display area
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)

        orig_frame = ttk.LabelFrame(image_frame, text="Original", padding=5)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.orig_canvas = tk.Canvas(orig_frame, bg="#333")
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)

        result_frame = ttk.LabelFrame(image_frame, text="Result", padding=5)
        result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.result_canvas = tk.Canvas(result_frame, bg="#333")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

        # Palette display
        self.palette_frame = ttk.Frame(main_frame)
        self.palette_frame.pack(fill=tk.X, pady=(10, 0))
        self.palette_canvas = tk.Canvas(self.palette_frame, height=30, bg="#222")
        self.palette_canvas.pack(fill=tk.X)

        # Status bar
        self.status = ttk.Label(main_frame, text="Open an image to get started")
        self.status.pack(fill=tk.X, pady=(10, 0))

    def on_slider_change(self, event=None):
        self.color_label.config(text=str(int(self.n_colors.get())))

    def open_image(self):
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.webp"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.image_path = Path(path)
            self.original_image = Image.open(path).convert("RGB")
            self.display_original()
            self.update_preview()
            self.status.config(text=f"Loaded: {self.image_path.name}")

    def display_original(self):
        if self.original_image is None:
            return
        self.root.update_idletasks()
        canvas_w = self.orig_canvas.winfo_width()
        canvas_h = self.orig_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 400, 400

        img = self.original_image.copy()
        img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        self.orig_photo = ImageTk.PhotoImage(img)
        self.orig_canvas.delete("all")
        self.orig_canvas.create_image(canvas_w//2, canvas_h//2, image=self.orig_photo)

    def update_preview(self):
        if self.original_image is None or self.processing:
            return

        n = int(self.n_colors.get())
        grayscale = self.grayscale_var.get()
        self.processing = True
        self.status.config(text=f"Processing preview...")
        self.root.update()

        def process():
            if grayscale:
                result = quantize_image(self.original_image, n, grayscale=True)
                palette = None
            else:
                result, palette = quantize_image(self.original_image, n, grayscale=False, fast=True)
            self.root.after(0, lambda: self.finish_preview(result, palette, n, grayscale))

        threading.Thread(target=process, daemon=True).start()

    def finish_preview(self, result, palette, n, grayscale):
        self.current_result = result
        self.current_palette = palette
        self.display_result()
        self.display_palette(palette, grayscale)
        self.processing = False
        self.status.config(text=f"Preview - {n} {'gray levels' if grayscale else 'colors'}")

    def display_result(self):
        if self.current_result is None:
            return
        self.root.update_idletasks()
        canvas_w = self.result_canvas.winfo_width()
        canvas_h = self.result_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 400, 400

        img = self.current_result.copy()
        img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        self.result_photo = ImageTk.PhotoImage(img)
        self.result_canvas.delete("all")
        self.result_canvas.create_image(canvas_w//2, canvas_h//2, image=self.result_photo)

    def display_palette(self, palette, grayscale):
        self.palette_canvas.delete("all")
        if grayscale or palette is None:
            self.palette_canvas.create_text(10, 15, text="Grayscale mode", anchor=tk.W, fill="white")
            return

        w = self.palette_canvas.winfo_width()
        n = len(palette)
        box_w = min(50, w // n)
        for i, (r, g, b) in enumerate(palette):
            color = f"#{r:02x}{g:02x}{b:02x}"
            x0 = i * box_w
            self.palette_canvas.create_rectangle(x0, 0, x0 + box_w - 2, 28, fill=color, outline="")
            # Color hex below
            self.palette_canvas.create_text(x0 + box_w//2, 15, text=color, fill="white" if (r+g+b)/3 < 128 else "black", font=("monospace", 7))

    def save_image(self):
        if self.original_image is None:
            messagebox.showwarning("No image", "No image loaded")
            return

        n = int(self.n_colors.get())
        grayscale = self.grayscale_var.get()
        suffix = f"_gray{n}" if grayscale else f"_color{n}"
        default_name = f"{self.image_path.stem}{suffix}.png" if self.image_path else "output.png"

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")]
        )
        if not path:
            return

        self.status.config(text="Saving full resolution...")
        self.root.update()

        def save():
            if grayscale:
                result = quantize_image(self.original_image, n, grayscale=True)
            else:
                result, _ = quantize_image(self.original_image, n, grayscale=False, fast=False)
            result.save(path)
            self.root.after(0, lambda: self.status.config(text=f"Saved: {path}"))

        threading.Thread(target=save, daemon=True).start()


def main():
    root = tk.Tk()
    app = ColorQuantizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
