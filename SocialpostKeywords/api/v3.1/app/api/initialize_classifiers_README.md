# Initialize Classifiers

This Python script, `initialize_classifiers.py`, initializes three zero-shot-classification pipelines using the Hugging Face Transformers library and saves them as a pickle file. The classifiers are based on different pre-trained models, which are specified in a `parameter.json` file.

### Dependencies

- `os`
- `json`
- `pickle`
- `transformers`

### Usage

1. Ensure that the `parameter.json` file is located in the same directory as the `initialize_classifiers.py` script.
2. Replace the sensitive information in the `parameter.json` file with dummy or fake examples.
3. Run the `initialize_classifiers.py` script.

### `parameter.json` Structure

The `parameter.json` file should contain the following keys:

- `models_path`: The path where the classifiers will be saved as a pickle file.
- `model1_gpu_device`: The GPU device index for the first classifier.
- `model2_gpu_device`: The GPU device index for the second classifier.
- `model3_gpu_device`: The GPU device index for the third classifier.

Example:

```json
{
  "models_path": "path/to/save/classifiers.pkl",
  "model1_gpu_device": 0,
  "model2_gpu_device": 1,
  "model3_gpu_device": 2
}
```

### Function: `init_clfs()`

This function initializes three zero-shot-classification pipelines using the Hugging Face Transformers library. It returns a tuple containing the three classifiers.

#### Returns

- `tuple`: A tuple containing three zero-shot-classification pipelines.

### Saving Classifiers

After initializing the classifiers, they are saved as a pickle file in the specified `models_path` from the `parameter.json` file.