# Output Result Formatter

This Python script, `OutputResultFormatter.py`, is designed to process and format output results based on input topics and keyphrases. It also includes a function to fetch data from a MongoDB database and convert it to a DataFrame.

### Functions

The script contains two main functions:

1. `get_output_result()`
2. `mongo_to_df()`

#### `get_output_result()`

This function processes and formats the output result based on the input topics and keyphrases. It takes the following input parameters:

- `title_topic`: A list of topics from the title.
- `long_post_topic`: A list of topics from a long post.
- `long_post_keyphrase`: A list of keyphrases from a long post.
- `short_post_topic`: A list of topics from a short post.
- `short_post_keyphrase`: A list of keyphrases from a short post.
- `topic2cat_di`: A dictionary mapping topics to categories.
- `latest_topics`: A list of the latest topics.
- `kw_map_di`: A dictionary mapping keywords to topics.

The function first determines the type of post (long or short) and copies the corresponding topics and keyphrases. It then adds title topics to the list if they are not already present. Finally, it processes and formats the output names and returns the cleaned output names.

#### `mongo_to_df()`

This function fetches data from a MongoDB database and converts it to a DataFrame. It takes an optional input parameter:

- `filter_day`: The number of days to filter the data (default is 7).

The function connects to the MongoDB database using the provided credentials and fetches data from the `posts_today` and `posts_today_community` collections. It then converts the fetched data into DataFrames and concatenates them. The resulting DataFrame is sorted by the `created_utc` column in descending order and returned.

### Usage

To use this script, import the functions and call them with the required parameters. Make sure to replace the placeholders (e.g., 'your_host', 'your_username', etc.) with your actual values.

```python
from OutputResultFormatter import get_output_result, mongo_to_df

# Call the functions with the required parameters
output_result = get_output_result(title_topic, long_post_topic, long_post_keyphrase, short_post_topic, short_post_keyphrase, topic2cat_di, latest_topics, kw_map_di)
df = mongo_to_df(filter_day=7)
```

### Dependencies

This script requires the following Python libraries:

- `pymongo`
- `pandas`

Make sure to install these libraries before running the script.