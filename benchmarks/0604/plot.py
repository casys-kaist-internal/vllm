import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('benchmark_results.csv')

# Define the request rate threshold
max_request_rate = 128  # Change this value as needed

# Define the label conditions
conditions = [
    (df['draft_size'] == 0) & (df['chunk_prefill']
                               == False) & (df['colocate'] == False),
    (df['draft_size'] == 0) & (df['chunk_prefill']
                               == True) & (df['colocate'] == False),
    (df['draft_size'] == 4) & (df['chunk_prefill']
                               == False) & (df['colocate'] == False),
    (df['draft_size'] == 4) & (df['chunk_prefill']
                               == False) & (df['colocate'] == True),
    (df['draft_size'] == 4) & (df['chunk_prefill']
                               == True) & (df['colocate'] == False),
    (df['draft_size'] == 4) & (df['chunk_prefill']
                               == True) & (df['colocate'] == True),
    (df['draft_size'] == 7) & (df['chunk_prefill']
                               == False) & (df['colocate'] == False),
    (df['draft_size'] == 7) & (df['chunk_prefill']
                               == False) & (df['colocate'] == True),
    (df['draft_size'] == 7) & (df['chunk_prefill']
                               == True) & (df['colocate'] == False),
    (df['draft_size'] == 7) & (df['chunk_prefill']
                               == True) & (df['colocate'] == True)
]

# Define the labels
labels = [
    'baseline',
    'baseline_cp',
    'draft_4',
    'draft_4_colocate',
    'draft_4_cp',
    'draft_4_colocate_cp',
    'draft_7',
    'draft_7_colocate',
    'draft_7_cp',
    'draft_7_colocate_cp',
]

# Apply the labels
df['label'] = 'unlabeled'
for condition, label in zip(conditions, labels):
    df.loc[condition, 'label'] = label

# Filter data based on the request rate threshold
df = df[df['request_rate'] < max_request_rate]

# Loop through unique datasets
for dataset in df['dataset'].unique():
    subset_df = df[df['dataset'] == dataset]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    # Plot data without chunk prefill
    for label in subset_df['label'].unique():
        if 'cp' not in label:
            subset = subset_df[subset_df['label'] == label]
            ax1.plot(subset['request_rate'],
                     subset['avg_per_token_latency'], marker='o', label=label)

    ax1.set_title(
        f'Avg Per Token Latency vs Request Rate (Without Chunk Prefill) - {dataset}')
    ax1.set_xlabel('Arrival Rate (reqs/s)')
    ax1.set_ylabel('Avg Per Token Latency (s/token)')
    ax1.legend()
    ax1.grid(True)

    # Plot data with chunk prefill
    for label in subset_df['label'].unique():
        if 'cp' in label:
            subset = subset_df[subset_df['label'] == label]
            ax2.plot(subset['request_rate'], subset['avg_per_token_latency'],
                     marker='o', label=label.replace('_cp', ''))

    ax2.set_title(
        f'Avg Per Token Latency vs Request Rate (With Chunk Prefill) - {dataset}')
    ax2.set_xlabel('Arrival Rate (reqs/s)')
    ax2.legend()
    ax2.grid(True)

    # Save the plot for this dataset
    plt.savefig(f'benchmark_results_{dataset}.png')
    plt.close(fig)
