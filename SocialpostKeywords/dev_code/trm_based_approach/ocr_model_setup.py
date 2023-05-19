
#!/usr/bin/env python
# coding: utf-8

import os

# Create a directory for the OCR model
os.makedirs('/path/to/your/OCR_postSupport/model/', exist_ok=True)

# Run the install_models.py script
# Replace the path with the correct path to your install_models.py script
os.system('python /path/to/your/OCR_postSupport/api/v0.1/app/api/install_models.py')

# Run the inference_post_ocr.py script
# Replace the path with the correct path to your inference_post_ocr.py script
os.system('python /path/to/your/OCR_postSupport/api/v0.1/app/api/inference_post_ocr.py')
