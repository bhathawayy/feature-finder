# FeatureFinder

A Python GUI application for computer vision feature detection. FeatureFinder provides an intuitive interface to detect and analyze blobs, rectangular shapes, and cross-hairs in images using OpenCV.

## Features

- **Blob Detection**: Identify circular/blob-like features with configurable size and circularity parameters
- **Rectangle Detection**: Find rectangular shapes and structures
- **Cross-hair Detection**: Locate cross-hair patterns and intersections
- **Real-time Processing**: Interactive parameter adjustment with live preview
- **Multiple Image Formats**: Support for common image formats
- **Export Results**: Save processed images and detection data

## Requirements

- Python 3.12+
- See `requirements.txt` for full dependency list

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bhathawayy/feature-finder.git
cd feature-finder
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python -m feature_finder.app_main
```
--or--
```bash
from feature_finder import app_main

app_main.launch_gui()
```

1. Load an image using the file dialog
2. Select detection method (Blob, Rectangle, or Cross-hair)
3. Adjust parameters in real-time
4. View results in the preview window
5. Export processed images or detection data

## Project Structure

```
src/feature_finder/
├── app_main.py              # Main application entry point
├── detection_methods.py     # Core detection algorithms
├── processing_support.py    # Image processing utilities
├── interface/              # GUI components
│   ├── ui_form.py          # Generated UI code
│   └── form.ui             # Qt Designer UI file
└── resources/              # Application resources
```

## Development

To modify the UI:
1. Edit `interface/form.ui` in Qt Designer
2. Click "Build" or regenerate Python UI file:
```bash
pyside6-uic interface/form.ui -o interface/ui_form.py
```

## License

This project is licensed under the MIT License.

## Author

Brooke Hathaway
