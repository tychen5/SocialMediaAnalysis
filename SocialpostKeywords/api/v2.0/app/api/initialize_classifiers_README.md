# Initialize Classifiers

This Python script, `initialize_classifiers.py`, is designed to initialize and save three zero-shot-classification pipelines using different pre-trained models from the Hugging Face Transformers library. The script sets up the classifiers, initializes them, and saves them as a pickle file for later use.

### Dependencies

- `os`
- `json`
- `pickle`
- `transformers`

### Usage

1. Ensure that the required dependencies are installed.
2. Set the appropriate model names and paths in the `parameter.json` file.
3. Run the `initialize_classifiers.py` script.

### How it works

The script performs the following steps:

1. Sets the current directory and parameter file path.
2. Loads parameters from the `parameter.json` file.
3. Sets the models path.
4. Defines the `init_clfs()` function, which initializes the zero-shot-classification pipelines with different models.
5. Calls the `init_clfs()` function to initialize the classifiers.
6. Saves the initialized classifiers using pickle.

### Functions

- `init_clfs()`: This function initializes the zero-shot-classification pipelines with different models and returns a tuple containing the initialized classifiers.

### Parameters

The script uses the following parameters from the `parameter.json` file:

- `models_path`: The path where the initialized classifiers will be saved as a pickle file.
- `model1_gpu_device`: The GPU device index for the first classifier.
- `model2_gpu_device`: The GPU device index for the second classifier.
- `model3_gpu_device`: The GPU device index for the third classifier.

### Models

The script initializes the following pre-trained models for zero-shot-classification:

1. `facebook/bart-large-mnli`
2. `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
3. `joeddav/xlm-roberta-large-xnli`

**Note**: Replace the model names and paths with dummy examples if needed.