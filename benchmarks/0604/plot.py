import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('benchmark_results.csv')

# Define the label conditions
conditions = [
    (df['draft_size'] == 0) & (df['chunk_prefill']
                               == False) & (df['collocate'] == False),
    (df['draft_size'] == 0) & (df['chunk_prefill']
                               == True) & (df['collocate'] == False),
    (df['draft_size'] == 4) & (df['chunk_prefill']
                               == False) & (df['collocate'] == False),
    (df['draft_size'] == 4) & (df['chunk_prefill']
                               == False) & (df['collocate'] == True),
    (df['draft_size'] == 4) & (df['chunk_prefill']
                               == True) & (df['collocate'] == False),
    (df['draft_size'] == 4) & (df['chunk_prefill']
                               == True) & (df['collocate'] == True),
    (df['draft_size'] == 7) & (df['chunk_prefill']
                               == False) & (df['collocate'] == False),
    (df['draft_size'] == 7) & (df['chunk_prefill']
                               == False) & (df['collocate'] == True),
    (df['draft_size'] == 7) & (df['chunk_prefill']
                               == True) & (df['collocate'] == False),
    (df['draft_size'] == 7) & (df['chunk_prefill']
                               == True) & (df['collocate'] == True)
]

# Define the labels
labels = [
    'baseline',
    'baseline_cp',
    'draft_4',
    'draft_4_collocate',
    'draft_4_cp',
    'draft_4_collocate_cp',
    'draft_7',
    'draft_7_collocate',
    'draft_7_cp',
    'draft_7_collocate_cp',
]

# Apply the labels
df['label'] = 'unlabeled'
for condition, label in zip(conditions, labels):
    df.loc[condition, 'label'] = label

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# Plot data without chunk prefill
for label in df['label'].unique():
    if 'cp' not in label:
        subset = df[df['label'] == label]
        ax1.plot(subset['request_rate'],
                 subset['avg_per_token_latency'], marker='o', label=label)

ax1.set_title('Avg Per Token Latency vs Request Rate (Without Chunk Prefill)')
ax1.set_xlabel('Arrival Rate (reqs/s)')
ax1.set_ylabel('Avg Per Token Latency (s/token)')
ax1.legend()
ax1.grid(True)

# Plot data with chunk prefill
for label in df['label'].unique():
    if 'cp' in label:
        subset = df[df['label'] == label]
        ax2.plot(subset['request_rate'], subset['avg_per_token_latency'],
                 marker='o', label=label.replace('_cp', ''))

ax2.set_title('Avg Per Token Latency vs Request Rate (With Chunk Prefill)')
ax2.set_xlabel('Arrival Rate (reqs/s)')
ax2.legend()
ax2.grid(True)

plt.savefig('result.png')
plt.close()
