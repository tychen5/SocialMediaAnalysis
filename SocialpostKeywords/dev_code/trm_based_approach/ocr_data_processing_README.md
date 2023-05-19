
# OCR Data Processing

This Python script, `ocr_data_processing.py`, is designed to process Optical Character Recognition (OCR) data and perform grammar checks and data augmentation on the extracted text. The script also includes functions to create datasets and perform topic modeling using two different models.

## Features

1. Processes OCR data and extracts text from the data.
2. Performs grammar checks on the extracted text.
3. Augments the text data by creating variations of the original text.
4. Creates datasets for further processing and analysis.
5. Performs topic modeling using two different models.

## Classes

1. `GetData2`: A class that inherits from `torch.utils.data.Dataset` and is used to create a dataset from the input data.
2. `GetFulldocData`: A class that inherits from `torch.utils.data.Dataset` and is used to create a dataset from the full document data.
3. `GetLabelsData`: A class that inherits from `torch.utils.data.Dataset` and is used to create a dataset from the label data.

## Functions

1. `function1`: A remote function that performs topic modeling using the first model.
2. `function2`: A remote function that performs topic modeling using the second model.

## Usage

1. Replace the placeholders `'path/to/your/grammar_chk_point'` and `'path/to/your/grammar_pass_df_path'` with the actual paths to your files.
2. Replace the placeholders `'model_checkpoint1_path'` and `'model_checkpoint2_path'` with the actual paths to your model checkpoint files.
3. Run the script using `python ocr_data_processing.py`.

## Dependencies

- torch
- pandas
- pickle
- tqdm
- ray

## Note

Please ensure that you have the necessary dependencies installed and the appropriate paths to your files before running the script.