#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Read the protect survey data from a CSV file
protect_survey_df = pd.read_csv("/path/to/protect_survey_data.csv")

# Print the column names and the count of issue types
print("Column names:", protect_survey_df.columns)
print("Issue type count:", protect_survey_df['issue_type'].value_counts())

# Display the protect survey data
protect_survey_df

# Print the unique issue types and display the comments for the 'Installation' issue type
print("Unique issue types:", protect_survey_df['issue_type'].unique())
print("Comments for 'Installation' issue type:", protect_survey_df[protect_survey_df['issue_type']=='Installation']['comment'].head(60))