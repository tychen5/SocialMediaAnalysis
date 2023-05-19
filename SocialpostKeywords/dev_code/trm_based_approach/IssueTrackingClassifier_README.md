IssueTrackingClassifier.py

# README.md

## Issue Tracking Classifier

This Python script, `IssueTrackingClassifier.py`, is designed to classify and update issue tracking data based on keywords and applications. The script processes a dataset containing issue names and associated keywords, and then classifies the issues based on their application and keyword frequency. The output is a pickled file containing the updated issue tracking data and a DataFrame with the classified keywords.

### Functions

The script contains two main functions:

1. `classify_function(element)`: Classifies the given element and updates the `application_clf_di` dictionary.
2. `get_kw_name(kw_li)`: Gets the keyword names from the given list and returns a sorted set of new names.

### Workflow

1. The `classify_function` is applied to all issue names using the `map` function.
2. A `knee_dict` is created to store the issue names and their associated keyword lists.
3. The script iterates through the `application_clf_di` dictionary and calculates the knee points for each application using the `KneeLocator` class from the `kneed` library.
4. Based on the knee points, the script updates the `knee_dict` with the appropriate keyword lists.
5. The updated data is pickled and saved to a specified file.
6. The `get_kw_name` function is applied to the original DataFrame to generate a new column with the classified keyword names.
7. The updated DataFrame is saved as a pickled file.

### Dependencies

- Python 3.x
- pandas
- kneed
- pickle

### Usage

1. Replace the dummy paths in the script with your actual file paths.
2. Ensure that the required dependencies are installed.
3. Run the `IssueTrackingClassifier.py` script.

### Output

The script generates two output files:

1. A pickled file containing the updated issue tracking data, including the `taken_keywords`, `kw_cat_di`, `all_kw_df`, `application_clf_di`, `version_dict`, and `knee_dict`.
2. A pickled DataFrame with the classified keywords added as a new column.