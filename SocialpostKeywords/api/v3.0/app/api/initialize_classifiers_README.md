# Initialize Classifiers

This Python script, `initialize_classifiers.py`, is designed to initialize and save three zero-shot classification models from the Hugging Face Transformers library. The script loads parameters from a JSON file, initializes the classifiers, and saves them using the pickle library.

### Dependencies

- `os`
- `json`
- `pickle`
- `transformers`

### Usage

1. Ensure that the `parameter.json` file is located in the same directory as the script and contains the necessary parameters. Replace any sensitive information with dummy or fake examples as needed. The JSON file should have the following structure:

```json
{
  "models_path": "path/to/save/models.pkl",
  "model1_gpu_device": 0,
  "model2_gpu_device": 1,
  "model3_gpu_device": 2
}
```

2. Run the `initialize_classifiers.py` script:

```bash
python initialize_classifiers.py
```

3. The script will initialize the classifiers and save them as a tuple in a pickle file at the specified `models_path`.

### Functions

- `init_clfs()`: Initializes three zero-shot classification pipelines with different models from the Hugging Face Transformers library and returns them as a tuple.

### Models

The script initializes the following models:

1. `facebook/bart-large-mnli`
2. `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
3. `joeddav/xlm-roberta-large-xnli`

These models are used for zero-shot classification tasks and are loaded using the Hugging Face Transformers library.