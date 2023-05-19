# Cluster Evaluation for Reddit Data

This Python script, `cluster_evaluation_reddit.py`, evaluates the clustering results of Reddit data by calculating various statistical measures such as Chi-Square, KL-Divergence, and Cosine Distance. The script also computes the similarity matrix between two tensors using PyTorch and TensorFlow.

### Dependencies

- os
- pickle
- numpy
- pandas
- scipy
- torch
- numba
- tensorflow
- faiss
- cuml

### Usage

1. Update the `dir_path` and `csv_path` variables with the appropriate data and results paths.
2. Run the script using `python cluster_evaluation_reddit.py`.

### Functionality

The script performs the following tasks:

1. Iterates through the pickle files containing clustering results.
2. For each pickle file, it calculates the Chi-Square and KL-Divergence values.
3. Appends the results to lists for further processing.
4. Creates a DataFrame for storing the statistics.
5. Calculates the mean and standard deviation of the cluster sizes.
6. Calculates the distance based on the mean and standard deviation.
7. Saves the statistics to a CSV file.
8. Defines functions for computing the similarity matrix and cosine distances using PyTorch and TensorFlow.
9. Provides example usage of the `sim_matrix` and `compute_cosine_distances` functions.

### Functions

- `sim_matrix(a, b, eps=1e-6)`: Computes the similarity matrix between two tensors `a` and `b` using PyTorch.
- `compute_cosine_distances(a, b)`: Computes the cosine distances between two tensors `a` and `b` using TensorFlow.
- `calc_statistics(li)`: Calculates the mean and standard deviation of a list of values.
- `calc_dist(mean, std, cluster_size_li)`: Calculates the distance based on the mean and standard deviation of cluster sizes.

### Output

The script generates a CSV file named `evaluation_statistiscs.csv` containing the calculated statistics for each clustering result.