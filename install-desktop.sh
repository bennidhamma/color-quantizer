#!/bin/bash
# Install desktop shortcut for Color Quantizer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Update .desktop file with correct path
sed "s|Exec=.*|Exec=$SCRIPT_DIR/run.sh|" "$SCRIPT_DIR/color-quantizer.desktop" > ~/.local/share/applications/color-quantizer.desktop

echo "Desktop shortcut installed! Search for 'Color Quantizer' in your app menu."
