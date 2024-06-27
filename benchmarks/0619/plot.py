import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('benchmark_results.csv')

# Define lists for subplot division and labeling
# Controls how subplots are divided
subplot_division_features = ['chunk_prefill', 'colocate', 'output_len']
# Controls labeling within subplots
labeling_features = ['draft_size', 'target_attention']

# Generate a composite key for subplot division and labeling
df['subplot_key'] = df.apply(lambda row: ' | '.join(
    [f"{col}={row[col]}" for col in subplot_division_features]), axis=1)
df['label_key'] = df.apply(lambda row: ' | '.join(
    [f"{col}={row[col]}" for col in labeling_features]), axis=1)

# Define the metrics to plot (x-axis and y-axis variables)
metric_x = 'throughput'
metric_y = 'latency'

# Prepare custom color mappings
unique_draft_sizes = df['draft_size'].unique()
# Define a set of colors for each draft_size
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# Ensure there are enough colors, repeat the list if necessary
if len(unique_draft_sizes) > len(custom_colors):
    custom_colors = custom_colors * \
        (len(unique_draft_sizes) // len(custom_colors) + 1)
color_mapping = {draft: color for draft, color in zip(
    sorted(unique_draft_sizes), custom_colors)}

# Line styles for target_attention
line_styles = {'True': '--', 'False': '-'}

# Function to plot data for each unique subplot division


def plot_data(df, x, y, ax):
    # Sort configurations by draft_size numerically and target_attention alphabetically
    unique_configs = sorted(df['label_key'].unique(), key=lambda x: (
        int(x.split('|')[0].split('=')[1]), x.split('|')[1].split('=')[1]))

    for config in unique_configs:
        draft_size = config.split('|')[0].split('=')[1]
        target_att = config.split('|')[1].split('=')[1]
        subset = df[df['label_key'] == config]
        color = color_mapping[int(draft_size)]  # Get color for draft_size
        # Get line style for target_attention
        line_style = line_styles[target_att]
        ax.plot(subset[x], subset[y], marker='o', linestyle=line_style,
                color=color, label=f"Draft {draft_size}, TA {target_att}")
    ax.set_xlabel(x.replace('_', ' ').title())
    ax.set_ylabel(y.replace('_', ' ').title())
    ax.legend()
    ax.grid(True)


# Loop through unique datasets and dynamically generate plots based on subplot keys
for dataset in df['dataset'].unique():
    subset_df = df[df['dataset'] == dataset]
    unique_subplot_keys = subset_df['subplot_key'].unique()

    # Create a subplot for each unique subplot key
    n_cols = len(unique_subplot_keys)
    fig, axes = plt.subplots(1, n_cols, figsize=(10 * n_cols, 8), sharey=True)

    # If only one subplot, make sure 'axes' is iterable
    if n_cols == 1:
        axes = [axes]

    # Plot data for each unique subplot division
    for ax, key in zip(axes, unique_subplot_keys):
        ax.set_title(
            f"{metric_y.replace('_', ' ').title()} vs {metric_x.replace('_', ' ').title()} ({key})")
        config_df = subset_df[subset_df['subplot_key'] == key]
        plot_data(config_df, metric_x, metric_y, ax)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'benchmark_results_{dataset}.png')
    plt.close(fig)
