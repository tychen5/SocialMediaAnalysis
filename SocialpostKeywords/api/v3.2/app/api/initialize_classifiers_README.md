# Initialize Classifiers

This Python script, `initialize_classifiers.py`, initializes three zero-shot classification models using the Hugging Face Transformers library and saves them as a pickle file. The script is designed to be easily customizable, allowing users to modify the models and parameters used for classification.

### Dependencies

- Python 3.6 or higher
- Hugging Face Transformers library

### Usage

1. Ensure that the `parameter.json` file is located in the same directory as the `initialize_classifiers.py` script. Replace any sensitive information in the `parameter.json` file with dummy or fake examples.

2. Run the `initialize_classifiers.py` script:

```bash
python initialize_classifiers.py
```

This will initialize the classifiers and save them as a pickle file in the specified models path.

### Code Overview

The script performs the following steps:

1. Import necessary libraries and modules.
2. Get the current directory and parameter file path.
3. Load parameters from the `parameter.json` file.
4. Define the `init_clfs()` function, which initializes three zero-shot classification models using the Hugging Face Transformers library.
5. Call the `init_clfs()` function to initialize the classifiers.
6. Save the classifiers as a pickle file in the specified models path.

### Customization

To use different models or modify the parameters, update the `parameter.json` file with the desired values. The script will automatically load the new parameters and use them to initialize the classifiers.

### Models Used

The script initializes the following zero-shot classification models:

1. `facebook/bart-large-mnli`
2. `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
3. `joeddav/xlm-roberta-large-xnli`

These models are chosen for their performance in zero-shot classification tasks. However, you can replace them with any other compatible models from the Hugging Face Model Hub.