# OCR Model Setup

This Python script, `ocr_model_setup.py`, automates the process of setting up an Optical Character Recognition (OCR) model for inference. The script creates a directory for the OCR model and runs two additional Python scripts: `install_models.py` and `inference_post_ocr.py`.

### Prerequisites

Before running this script, ensure that you have the following:

1. Python 3.x installed on your system.
2. The `install_models.py` and `inference_post_ocr.py` scripts available in your project directory.

### Usage

To use this script, follow these steps:

1. Update the paths in the script to match the locations of your `install_models.py` and `inference_post_ocr.py` scripts.
2. Update the path for the OCR model directory as needed.
3. Run the `ocr_model_setup.py` script.

### How It Works

The script performs the following actions:

1. Imports the `os` module to interact with the file system.
2. Creates a directory for the OCR model using `os.makedirs()`. The `exist_ok=True` parameter ensures that the directory is created only if it does not already exist.
3. Runs the `install_models.py` script using `os.system()`. This script installs the necessary OCR models for the project.
4. Runs the `inference_post_ocr.py` script using `os.system()`. This script performs OCR inference on the input data.

By automating these steps, the `ocr_model_setup.py` script simplifies the process of setting up an OCR model for inference, making it easier to integrate OCR functionality into your projects.