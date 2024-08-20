import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Load the data from the CSV file
df = pd.read_csv('figure_1_0813.csv')

# Define features used for subplot division and labeling
subplot_division_features = ['target_model',
                             'draft_model', 'temperature', 'budget_seq']
labeling_features = ['draft_size', 'colocate',
                     'target_attention', 'drop_threshold', 'prefill_schedule_mode']
labeling_features_with_rate = labeling_features + ['request_rate']

# Draw only budget_seq = 128
# df = df[df['budget_seq'] == 128]

# Generate a composite key for subplot division and labeling
df['subplot_key'] = df.apply(lambda row: ' | '.join(
    [f"{col}={row[col]}" for col in subplot_division_features]), axis=1)
df['label_key'] = df.apply(lambda row: '|'.join(
    [f"{col}={row[col]}" for col in labeling_features]), axis=1)
df['label_key_with_rate'] = df.apply(lambda row: '|'.join(
    [f"{col}={row[col]}" for col in labeling_features_with_rate]), axis=1)

# Define the metrics to plot (x-axis and y-axis variables)
metric_x = 'request_throughput'
metric_y = 'token_latency'

# Prepare custom color mappings
unique_draft_sizes = df['draft_size'].unique()
# Define a set of colors for each draft_size
custom_colors = ['#4E79A7', '#F28E2B', '#E15759',  '#59A14F', '#EDC948',
                 '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']

# Ensure there are enough colors, repeat the list if necessary
if len(unique_draft_sizes) > len(custom_colors):
    custom_colors = custom_colors * \
        (len(unique_draft_sizes) // len(custom_colors) + 1)
color_mapping = {draft: color for draft, color in zip(
    sorted(unique_draft_sizes), custom_colors)}

# Define marker styles based on request rates
marker_styles = {'1': 'o', '2': '^', '4': 's', '8': 'D', '12': 'P'}

# Function to plot data for each subplot


def plot_data(df, x, y, ax):
    # Generate unique configs including request_rate for line plot
    unique_configs_with_rate = sorted(df['label_key'].unique(), key=lambda x: (
        int(x.split('|')[0].split('=')[1]), x.split('|')[1].split('=')[1]))

    # Plot lines based on unique config with request_rate
    for config_with_rate in unique_configs_with_rate:
        draft_size = int(config_with_rate.split('|')[0].split('=')[1])
        subset_with_rate = df[df['label_key'] == config_with_rate]
        color = color_mapping[draft_size]  # Get color for draft_size
        ax.plot(subset_with_rate[x], subset_with_rate[y],
                color=color, linestyle='-', linewidth=2, zorder=1)

    # Plot points (scatter) based on unique config without request_rate
    unique_configs = sorted(df['label_key_with_rate'].unique(), key=lambda x: (
        int(x.split('|')[0].split('=')[1]), x.split('|')[1].split('=')[1]))

    # Plot scatter after lines to ensure they appear on top
    for config in unique_configs:
        draft_size = int(config.split('|')[0].split('=')[1])
        subset = df[df['label_key_with_rate'] == config]
        color = color_mapping[draft_size]  # Get color for draft_size
        for req_rate in subset['request_rate'].unique():
            marker = marker_styles[str(req_rate)]
            subset_req_rate = subset[subset['request_rate'] == req_rate]
            ax.scatter(subset_req_rate[x], subset_req_rate[y],
                       color=color, marker=marker, s=100, label=None, zorder=2)

    ax.set_xlabel("Request Throughput (req/s)", fontsize=14)
    ax.set_ylabel("Latency (s/token)", fontsize=14)
    ax.grid(True, linewidth=1)

    # Cut at 0.275 for y axis
    # ax.set_ylim(top=0)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # # Manually add annotations for request rates at specific positions
    # ax.text(450, 0.03, "1 req/s", fontsize=12, ha='center')
    # ax.text(850, 0.03, "2 req/s", fontsize=12, ha='center')
    # ax.text(1550, 0.05, "4 req/s", fontsize=12, ha='center')
    # ax.text(1900, 0.20, "8 req/s", fontsize=12, ha='center')


def create_legend(ax):
    # Manually create legend handles with only colors (no markers)
    legend_elements = []
    for draft_size, color in color_mapping.items():
        if draft_size == 0:
            label = "AR Decode"
        else:
            label = f"Spec Decode {draft_size}"
        legend_elements.append(
            Line2D([0], [0], color=color, lw=4, label=label))

    # Add the legend to the axis
    ax.legend(handles=legend_elements, fontsize=14)


# Loop through unique datasets and dynamically generate plots based on subplot keys
for dataset in df['dataset'].unique():
    subset_df = df[df['dataset'] == dataset]
    unique_subplot_keys = subset_df['subplot_key'].unique()

    # Create a subplot for each unique subplot key
    n_cols = len(unique_subplot_keys)
    fig, axes = plt.subplots(1, n_cols, figsize=(
        7 * n_cols, 4), sharey=True, sharex=True)

    # If only one subplot, make sure 'axes' is iterable
    if n_cols == 1:
        axes = [axes]

    # Plot data for each unique subplot division
    for ax, key in zip(axes, unique_subplot_keys):
        config_df = subset_df[subset_df['subplot_key'] == key]
        plot_data(config_df, metric_x, metric_y, ax)
        create_legend(ax)  # Add the legend with only line colors

    # Start x axis from 0
    for ax in axes:
        ax.set_xlim(left=0)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(
        f'plot/figure_1.png', bbox_inches='tight')
    plt.close(fig)
