# Read output.txt in pandas 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

df = pd.read_csv('reject_vocab.txt', sep=' ', header=None)
df.columns = ['dummy', 'x', 'y']

print(df)

# I want to count the number of occurences of each value in the 'x' column
# and store the result in a new column 'count'
df['count'] = df['x'].map(df['x'].value_counts())

# sort the dataframe by the 'count' column
df = df.sort_values(by='count', ascending=False)
print(df)

# print the top 100 unique values with highest count
top_values = df['x'].unique()[:100]

for value in top_values:
    count = df[df['x'] == value]['count'].values[0]
    print(f'Value: {value}, Count: {count}')