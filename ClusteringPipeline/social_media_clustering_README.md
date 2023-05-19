# Social Media Clustering

This Python script, `social_media_clustering.py`, is designed to perform clustering on social media data, specifically Reddit comments. The script preprocesses the data, calculates various clustering evaluation scores, and saves the results in a structured format. It also provides visualization remapping, sentiment mapping, and additional statistics for the clustered data.

### Dependencies

The script requires the following libraries:

- os
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
- pickle
- cupy
- matplotlib
- timeout_decorator
- torch
- hdbscan
- scipy
- sklearn
- cuml
- collections
- DBCV
- kneed
- tqdm
- aitools

### Preprocessing and Setup

The script starts by loading the labeled data and dropping unnecessary columns. It then initializes the UIAim object for tracking the experiment.

### Clustering Evaluation Scores

The `calc_scores` function calculates various clustering evaluation scores for the given vectors and labels. These scores include Silhouette score, Davies-Bouldin score, Calinski-Harabasz score, HDBSCAN scores, noise score, count penalty, and group penalty.

### Saving Results

The `save_results` function saves the clustering results in a structured format, including comment tables, cluster tables, and topic statistics.

### Helper Functions

Several helper functions are included in the script to assist with data processing and analysis:

- `convert_str`: Converts a list of numbers to a formatted string.
- `remapping_sentence_table`: Provides visualization remapping for the sentence table.
- `map_sentiment`: Maps sentiment scores and types to the data.
- `add_centroid_dist`: Adds centroid distance information to the data.
- `refine_statistics`: Recalculates statistics based on refined data.
- `combine_with_duplicate`: Combines vectors, labels, and tid_cid_li with their duplicates.
- `append_filtered_commsent`: Appends filtered comments and sentences to the sentence table.

### Usage

To use the script, simply run it with the appropriate input data and adjust the file paths as needed. The script will perform clustering on the input data and save the results in the specified output directories.