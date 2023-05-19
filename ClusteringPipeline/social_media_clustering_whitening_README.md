# Social Media Clustering

This Python script is designed to perform clustering on social media data, specifically Reddit comments. The script preprocesses the data, applies clustering algorithms (HDBSCAN and DBSCAN), and evaluates the clustering results using various metrics. The output includes CSV files containing the clustering results and statistics.

### Key Features

- Preprocessing of social media data (Reddit comments)
- Clustering using HDBSCAN and DBSCAN algorithms
- Evaluation of clustering results using various metrics
- Saving results in CSV format
- Handling duplicate data
- Appending filtered comments and sentences to the sentence table

### Dependencies

- cupy
- numpy
- pandas
- pickle
- gc
- requests
- json
- cuml
- sklearn
- matplotlib
- collections
- DBCV
- scipy
- timeout_decorator
- hdbscan
- torch
- tqdm
- kneed
- effcossim
- aitools

### Usage

1. Set the appropriate paths and credentials for the following variables:

   - `social_media_name`
   - `param_path`
   - `train_path`
   - `csv_output_dir`
   - `precompute_duplicate_path`

2. Load the labeled DataFrame and remove unnecessary columns.

3. Perform preprocessing steps, including normalization and dimensionality reduction.

4. Apply clustering algorithms (HDBSCAN and DBSCAN) and calculate evaluation scores.

5. Save the results in CSV format.

6. Refine the statistics by grouping by sentence due to remapping and adding centroid sentences to the statistics.

7. Combine vectors, labels, and tid_cid_li with their duplicates.

8. Append filtered comments and sentences to the sentence table.

### Functions

- `calc_scores(vectors, labels)`: Calculate various clustering evaluation scores for the given vectors and labels.
- `save_results(topic_di, output_dir_path, pickle_path)`: Save the clustering results and statistics in CSV format.
- `convert_str(li)`: Convert a list of numbers to a formatted string.
- `refine_statistics(sentence_table, cluster_table_path, topic_table_path)`: Recalculate statistics by grouping by sentence due to remapping and add centroid sentences to the statistics.
- `combine_with_duplicate(vectors, labels, tid_cid_li, topic_oridf, topic_dup_di, no_duplicate_idx)`: Combine vectors, labels, and tid_cid_li with their duplicates.
- `append_filtered_commsent(topic_statistic_df, df_full_comments, sentence_table, sentence_table_path)`: Append filtered comments and sentences to the sentence table.