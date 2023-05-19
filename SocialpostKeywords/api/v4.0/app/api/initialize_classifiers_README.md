Suggested .py filename: `initialize_classifiers.py`

# README.md

## Initialize Classifiers

This Python script, `initialize_classifiers.py`, is designed to set up and save three different zero-shot classification models using the Hugging Face Transformers library. The script downloads the necessary NLTK data, initializes the classifiers, and saves them using the pickle library for later use.

### Dependencies

- Python 3.6+
- NLTK
- Hugging Face Transformers

### Usage

1. Ensure that the `parameter.json` file is located in the same directory as the script and contains the necessary information, such as the models path and GPU device numbers for each model.

Example `parameter.json` file:

```json
{
  "models_path": "path/to/save/models.pkl",
  "model1_gpu_device": 0,
  "model2_gpu_device": 1,
  "model3_gpu_device": 2
}
```

2. Run the script:

```bash
python initialize_classifiers.py
```

This will download the necessary NLTK data, initialize the classifiers, and save them to the specified path in the `parameter.json` file.

### Models

The script initializes three zero-shot classification models using the Hugging Face Transformers library:

1. `facebook/bart-large-mnli`
2. `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
3. `joeddav/xlm-roberta-large-xnli`

These models are saved as a tuple using the pickle library for easy loading and usage in other scripts.

### Functions

- `init_clfs()`: Initializes the classifiers using the transformers pipeline and returns a tuple containing the three classifiers.

### Additional Notes

- The script uses the `os.cpu_count()` function to determine the number of workers for each model's pipeline.
- The GPU device numbers for each model can be specified in the `parameter.json` file.