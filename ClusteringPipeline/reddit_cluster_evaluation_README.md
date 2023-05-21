# Reddit Cluster Evaluation

This Python script, `reddit_cluster_evaluation.py`, is designed to evaluate the performance of clustering algorithms applied to Reddit data. The script calculates various statistical measures, such as mean, standard deviation, chi-square, and KL divergence, to assess the quality of the clustering results.

### Dependencies

The script requires the following Python libraries:

- os
- pickle
- numpy
- pandas
- scipy

### Functions

The script contains the following functions:

- `calc_statistics(li)`: Calculates the mean and standard deviation of a list of numbers.
- `calc_dist(mean, std, cluster_size_li)`: Calculates the 2-standard deviation and 3-standard deviation values for a list of cluster sizes.
- `my_dist(mean, std, cluster_size_li)`: Calculates a custom distance metric based on the mean and standard deviation of a list of cluster sizes.
- `clean(path)`: Cleans the input path by removing unnecessary parts.

### Usage

1. Replace the `dir_path` and `csv_path` variables with the appropriate paths to your data and CSV files.
2. Run the script using `python reddit_cluster_evaluation.py`.

### Output

The script generates two CSV files:

1. `evaluation_statistiscs.csv`: Contains various statistical measures for each clustering result, such as mean, standard deviation, chi-square, KL divergence, and custom distance metric.
2. `Leo_select_parameters.csv`: Contains the selected clustering parameters based on the custom distance metric.

Additionally, the script prints the optimal parameters for DBSCAN and HDBSCAN clustering algorithms.

### Note

Please ensure that you replace the `dir_path` and `csv_path` variables with your own paths before running the script.