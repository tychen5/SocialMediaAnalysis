# Reddit Cluster Analysis

This Python script, `reddit_cluster_analysis.py`, performs clustering analysis on Reddit comments and sentences. The main goal of this script is to identify and analyze clusters of similar comments and sentences, providing insights into the topics and sentiments discussed on Reddit.

### Dependencies

The script requires the following libraries:

- os
- numpy
- pandas
- pickle
- sklearn
- IPython

### Input Data

The input data for this script should be in the form of a DataFrame containing Reddit comments and sentences. The DataFrame should have the following columns:

- App Store
- App Name
- version
- published_at
- language
- comment_id
- rating
- ori_comment
- translation
- sentiment_overalltype
- sentiment_overallscore

### Functions

The script contains several functions that perform various tasks, such as:

- `save_results()`: Saves the results of the clustering analysis, including comment-topic mapping, topic-cluster mapping, and topic statistics.
- `convert_str()`: Converts a list of values into a string.
- `remapping_sentence_table()`: Remaps the sentence table for visualization purposes.
- `map_sentiment()`: Maps sentiment information to the comments and sentences.
- `add_centroid_dist()`: Adds centroid distance information to the sentence table.
- `refine_statistics()`: Refines statistics due to remapping.
- `combine_with_duplicate()`: Combines vectors, labels, and tid_cid_li with duplicates.
- `append_filtered_commsent()`: Appends filtered comments and sentences to the sentence table.

### Execution

The script iterates through different combinations of parameters for clustering analysis. For each combination, it performs the following steps:

1. Search for the best metric weights for big and small clusters.
2. Evaluate the small clusters.
3. Save the results of the clustering analysis.
4. Refine the statistics and append filtered comments and sentences to the sentence table.

### Output

The output of the script includes:

- A sentence table containing information about each sentence and its corresponding cluster.
- A cluster table containing information about each cluster and its corresponding topic.
- A topic table containing statistics about each topic.

These tables are saved as CSV files in the specified output directory. Additionally, a pickle file containing the comment-topic mapping, topic-cluster mapping, and topic statistics is saved for further analysis.