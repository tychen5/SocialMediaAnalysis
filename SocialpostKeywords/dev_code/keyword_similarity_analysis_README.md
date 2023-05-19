# Keyword Similarity Analysis

This Python script, `keyword_similarity_analysis.py`, is designed to analyze and merge similar keywords based on their similarity scores. The script utilizes the FuzzyWuzzy library to calculate similarity scores between keywords and the KneeLocator library to identify optimal thresholds for merging keywords.

## Features

- Calculate similarity scores between keywords using FuzzyWuzzy library
- Identify optimal thresholds for merging keywords using KneeLocator library
- Merge similar keywords based on their similarity scores and thresholds
- Calculate document frequency and term frequency scores for merged keywords
- Output merged keywords, document frequency, and keyword popularity scores in a pickle file

## Dependencies

- Python 3.6+
- numpy
- itertools
- fuzzywuzzy
- kneed (KneeLocator)
- pickle

## Functions

- `df_tf_score(kw_li, doc_li)`: Calculate document frequency and term frequency scores for a list of keywords and documents.
- `calc_similar(x.keyword, x.similarity, use_kw_df)`: Calculate similarity scores between keywords.
- `valid_kw(x.similar_scoreli, x.similarity, x.name, sim_thr, combined_idx_li)`: Validate keywords based on similarity scores and thresholds.
- `find_common_str(x.keyword, x.keyword_li)`: Find common strings between keywords.
- `compare_list_str_sim(leftli, rightli)`: Compare similarity scores between two lists of strings.

## Usage

1. Ensure all dependencies are installed.
2. Modify the input data and parameters as needed.
3. Run the `keyword_similarity_analysis.py` script.

## Output

The script will output a pickle file containing the merged keywords, document frequency, and keyword popularity scores for each keyword group. The output file will be saved in the `../output/` directory with the format `year_week.pkl`.

## Note

Please ensure that the input data is properly formatted and cleaned before running the script. The script assumes that the input data is in the correct format and does not include any error handling for improperly formatted data.