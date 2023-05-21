# MTL EDA and Preprocessing

This Python script, `mtl_eda_preprocessing.py`, is designed to perform exploratory data analysis (EDA) and preprocessing on a dataset containing protect survey data. The dataset is assumed to be in CSV format and contains information about various issue types and their associated comments.

### Dependencies

- pandas
- numpy

### Usage

1. Ensure that the required dependencies are installed.
2. Modify the path to the CSV file containing the protect survey data in the script.
3. Run the script using a Python interpreter.

### Functionality

The script performs the following tasks:

1. Reads the protect survey data from a CSV file using pandas.
2. Prints the column names of the dataset.
3. Prints the count of each issue type in the dataset.
4. Displays the protect survey data in a tabular format.
5. Prints the unique issue types present in the dataset.
6. Displays the comments associated with the 'Installation' issue type.

### Example

Given a CSV file with the following data:

```
issue_type,comment
Installation,The installation process was confusing.
Maintenance,The maintenance instructions were unclear.
Installation,The installation took longer than expected.
```

The script will output:

```
Column names: Index(['issue_type', 'comment'], dtype='object')
Issue type count: Installation    2
Maintenance      1
Name: issue_type, dtype: int64
Unique issue types: ['Installation' 'Maintenance']
Comments for 'Installation' issue type: 0    The installation process was confusing.
2       The installation took longer than expected.
Name: comment, dtype: object
```

This script is useful for gaining insights into the protect survey data and understanding the distribution of issue types. It can be further extended to perform additional EDA and preprocessing tasks as needed.