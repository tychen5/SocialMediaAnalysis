# Initialize Text Classification Models

This Python script initializes two sets of models for text classification and zero-shot classification. The models are loaded using the transformers library and are either loaded onto the CPU or GPU depending on the device specified. The models are then pickled and saved to disk for later use.

### Features

- Reads parameters from a JSON file
- Sets the appropriate directories and device to use
- Initializes two sets of models:
  - Text classification models for grammar checking
  - Zero-shot classification models for topic classification
- Pickles and saves the models to disk

### Dependencies

- `transformers`
- `pickle`
- `os`
- `json`
- `gc`
- `torch`

### Usage

1. Ensure that the `parameter.json` file is in the same directory as the script.
2. Set the desired parameters in the `parameter.json` file:
   - `"models_dir"`: The directory where the pickled models will be saved
   - `"use_device"`: The device to use for loading the models (1 for GPU, -1 for CPU)
3. Run the script to initialize the models and save them to disk.

### Functions

- `init_stage1_clf(use_device)`: Initializes the first set of models for text classification (grammar checking)
  - `use_device`: The device to use for loading the models (1 for GPU, -1 for CPU)
  - Returns: Two grammar checking models

- `init_stage2_clf(use_device)`: Initializes the second set of models for zero-shot classification (topic classification)
  - `use_device`: The device to use for loading the models (1 for GPU, -1 for CPU)
  - Returns: Two topic classification models

### Models

The script initializes the following models:

- Text classification models for grammar checking:
  - `yevheniimaslov/deberta-v3-large-cola`
  - `cointegrated/roberta-large-cola-krishna2020`

- Zero-shot classification models for topic classification:
  - `facebook/bart-large-mnli`
  - `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`