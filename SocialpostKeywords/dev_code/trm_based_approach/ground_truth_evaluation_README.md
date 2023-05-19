# Ground Truth Evaluation

This Python script, `ground_truth_evaluation.py`, is designed to evaluate topic classification performance by comparing ground truth data with predicted tags. It retrieves data from a MongoDB database, processes it, and writes the evaluation results back to the database. The script also includes utility functions for data cleaning and conversion.

### Dependencies

- `requests`
- `pandas`
- `numpy`
- `IPython`
- `sklearn`
- `pymongo`
- `datetime`
- `html`

### Functions

1. `convert_tags(ori_str)`: Converts the original string of tags into a list of tags.
2. `clean_url(url_str)`: Cleans the given URL string by removing unnecessary characters and formatting it properly.
3. `mongo_to_df(filter_day=7)`: Retrieves data from MongoDB and converts it into a pandas DataFrame. By default, it filters data from the past 7 days.
4. `convert_tag(ori_li)`: Converts the original list of tags into a list of binary-encoded tags.
5. `calc_postlen(title, selftext)`: Calculates the length of a post by combining the lengths of the title and selftext.

### Main Script

The main script performs the following steps:

1. Reads the verification and tag files from the specified file paths.
2. Retrieves data from MongoDB and converts it into a pandas DataFrame.
3. Cleans and processes the data, including converting tags and calculating post lengths.
4. Calculates evaluation metrics such as precision, recall, F1-score, and accuracy.
5. Writes the evaluation results back to the MongoDB database.

### Usage

To use the script, replace the placeholders (e.g., 'http://your_database_url/api/v1/update/ground_truth', '/path/to/your/verify_file.xlsx', etc.) with your actual information. Then, simply run the script using a Python interpreter.

```bash
python ground_truth_evaluation.py
```

This script will then evaluate the topic classification performance and update the ground truth data in the MongoDB database.