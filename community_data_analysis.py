#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
import numpy as np
import requests
import json

# Define API endpoint and parameters
url = 'https://api.example.com/data'
params = {'token': 'enter_your_token', 'date': '2021-01-01'}

# Make API request and store response
response = requests.get(url, params=params)
data = json.loads(response.text)

# Convert response to pandas dataframe
df = pd.DataFrame(data)

# Clean and preprocess data
df = df.dropna()
df['date'] = pd.to_datetime(df['date'])
df['value'] = df['value'].astype(float)

# Calculate summary statistics
mean_value = np.mean(df['value'])
max_value = np.max(df['value'])
min_value = np.min(df['value'])

# Print summary statistics
print('Mean value: {}'.format(mean_value))
print('Max value: {}'.format(max_value))
print('Min value: {}'.format(min_value))