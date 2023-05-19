# Reddit Cluster Analysis

This Python script, `reddit_cluster_analysis.py`, performs clustering analysis on Reddit comments using various clustering algorithms and evaluation metrics. The script imports necessary libraries, loads data, preprocesses it, and applies clustering algorithms to identify patterns and relationships among the comments. The results are then evaluated using various metrics and saved to output files.

### Libraries

The script uses the following libraries:

- os
- pickle
- functools
- operator
- warnings
- random
- time
- gc
- requests
- json
- numpy
- pandas
- cupy
- matplotlib
- torch
- collections
- scipy
- sklearn
- kneed
- cuml
- effcossim
- tqdm
- DBCV
- hdbscan
- timeout_decorator

### Data Loading and Preprocessing

The script requires the following data files:

- `params.pkl`: Contains the parameters for the clustering algorithms.
- `training_data.pkl`: Contains the training data for the clustering algorithms.
- `precompute_duplicate_path.pkl`: Contains precomputed duplicate data.

The script also requires the user to provide their own paths for the data files and their own credentials for the UIAim library.

### Clustering Algorithms and Evaluation Metrics

The script uses the following clustering algorithms:

- HDBSCAN
- DBSCAN

The script calculates various clustering evaluation scores for the given vectors and labels using the `calc_scores` function. The evaluation metrics used include:

- Silhouette Score
- Davies-Bouldin Score
- Calinski-Harabasz Score
- HDBSCAN Score

### Functions

The script contains several functions that perform various tasks such as:

- `calc_scores`: Calculates various clustering evaluation scores for the given vectors and labels.
- `refine_statistics`: Recalculates statistics by grouping by sentence due to remapping and adds centroid sentence to the statistics.
- `combine_with_duplicate`: Combines vectors, labels, and tid_cid_li with their duplicates.
- `append_filtered_commsent`: Appends filtered comments and sentences to the sentence table.

### Main Loop

The main loop of the script performs the following tasks:

1. Loads the used parameters list from the `params.pkl` file.
2. Selects random parameters for the clustering algorithms and evaluation metrics.
3. Sets the hyperparameters for the UIAim library.
4. Performs operations with the selected parameters.
5. Saves and updates the used parameters list.
6. Clears memory and output.

### Output

The script saves the updated topic and cluster tables as CSV files in the specified output directory.

## Usage

To use the `reddit_cluster_analysis.py` script, ensure that you have all the required libraries installed and the necessary data files in the specified paths. Update the paths and credentials in the script as needed. Then, simply run the script in your Python environment.