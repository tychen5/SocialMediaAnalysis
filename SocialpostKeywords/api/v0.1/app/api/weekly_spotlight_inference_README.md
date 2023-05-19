# Weekly Spotlight Inference

This Python script, `weekly_spotlight_inference.py`, is designed to load weekly spotlight inference results from pickle files. The script takes a list of year and week numbers as input and returns a list of dictionaries containing the inference results for each week.

### Dependencies

- Python 3.x
- os
- pickle

### Usage

To use this script, simply import the `weekly_spotlight_inference_inhouse` function and provide a list of strings containing year and week numbers, e.g., `['2022_8', '2022_9']`. The function will return a list of dictionaries containing the inference results for each week.

```python
from weekly_spotlight_inference import weekly_spotlight_inference_inhouse

user_input = ['2022_29']  # Example input
results = weekly_spotlight_inference_inhouse(user_input)
print(results)
```

### Function

#### `weekly_spotlight_inference_inhouse(year_week_list)`

- **Input**: `year_week_list` (list) - A list of strings containing year and week numbers, e.g., `['2022_8', '2022_9']`.
- **Output**: `final_result` (list) - A list of dictionaries containing the inference results for each week.

### Error Handling

If a specified year and week combination has not been inferred yet, the script will print an error message and continue processing the remaining weeks in the input list.

### Demo

A demo is included in the script, which can be run by executing the script directly:

```bash
python weekly_spotlight_inference.py
```

This demo will print the keys of the first dictionary in the list of results for the example input `['2022_29']`.