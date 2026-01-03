# Color Quantizer

Reduce images to a limited color palette using k-means clustering in LAB color space.

## Features

- **N-level grayscale**: Convert images to 2-16 gray levels
- **N-color quantization**: Extract dominant colors using k-means++ in perceptually uniform LAB space
- **GUI app**: Interactive preview with adjustable color count
- **Batch scripts**: Process entire folders from command line

## Setup

```bash
./setup.sh
source venv/bin/activate
```

On Ubuntu, you may also need:
```bash
sudo apt-get install python3-tk python3-pil.imagetk
```

## Usage

### GUI App
```bash
python color_quantizer_ui.py
```

### Command Line - Grayscale
```bash
python convert_5level_gray.py
```
Converts all images in current directory to 5-level grayscale, outputs to `out/`.

### Command Line - Color
```bash
python convert_n_colors.py 5      # 5 colors -> out_color_5/
python convert_n_colors.py 8      # 8 colors -> out_color_8/
```

## How It Works

1. Converts image to LAB color space (perceptually uniform)
2. Uses k-means++ clustering to find N representative colors
3. Maps each pixel to nearest palette color
4. Converts back to RGB for output

LAB clustering preserves distinct hues better than RGB clustering, which tends to produce muddy browns.
