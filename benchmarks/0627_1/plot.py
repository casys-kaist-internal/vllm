import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('benchmark_results.csv')

# Define lists for subplot division and labeling
# Controls how subplots are divided
subplot_division_features = ['chunk_prefill', 'output_len']
# Controls labeling within subplots
labeling_features = ['draft_size', 'target_attention', 'colocate']

# Generate a composite key for subplot division and labeling
df['subplot_key'] = df.apply(lambda row: ' | '.join(
    [f"{col}={row[col]}" for col in subplot_division_features]), axis=1)
df['label_key'] = df.apply(lambda row: '|'.join(
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

line_styles = {'True': '--', 'False': '-'}
marker_styles = {'True': 'o', 'False': 'x'}


def plot_data(df, x, y, ax):
    # Sort configurations by draft_size numerically and target_attention alphabetically
    unique_configs = sorted(df['label_key'].unique(), key=lambda x: (
        int(x.split('|')[0].split('=')[1]), x.split('|')[1].split('=')[1]))

    for config in unique_configs:
        draft_size = config.split('|')[0].split('=')[1]
        target_att = config.split('|')[1].split('=')[1]
        colocate = config.split('|')[2].split('=')[1]

        subset = df[df['label_key'] == config]
        color = color_mapping[int(draft_size)]  # Get color for draft_size
        # Get line style for target_attention
        line_style = line_styles[colocate]
        marker_style = marker_styles[target_att]

        # Plot points
        ax.plot(subset[x], subset[y], marker=marker_style,
                linestyle=line_style, color=color)

        # Annotate each point with the request rate
        for i, point in subset.iterrows():
            ax.annotate(f"{point['request_rate']}",
                        (point[x], point[y]),
                        textcoords="offset points",  # how to position the text
                        xytext=(10, 0),  # distance from text to points (x,y)
                        ha='center',  # horizontal alignment can be left, right or center
                        fontsize=10)

    ax.set_xlabel("Throughput (tokens/s)")
    ax.set_ylabel("Latency (s/token)")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(left=0)  # Modify this to set the minimum x value


# Loop through unique datasets and dynamically generate plots based on subplot keys
for dataset in df['dataset'].unique():
    subset_df = df[df['dataset'] == dataset]
    unique_subplot_keys = subset_df['subplot_key'].unique()

    # Create a subplot for each unique subplot key
    n_cols = len(unique_subplot_keys)
    fig, axes = plt.subplots(1, n_cols, figsize=(
        10 * n_cols, 8), sharey=True, sharex=True)

    # If only one subplot, make sure 'axes' is iterable
    if n_cols == 1:
        axes = [axes]

    # Plot data for each unique subplot division
    for ax, key in zip(axes, unique_subplot_keys):
        ax.set_title(
            f"{metric_y.replace('_', ' ').title()} vs {metric_x.replace('_', ' ').title()} ({key})")
        config_df = subset_df[subset_df['subplot_key'] == key]
        plot_data(config_df, metric_x, metric_y, ax)

    # Custom legend
    # Show available draft sizes and their corresponding colors
    draft_size_legend = [plt.Line2D([0], [0], color=color_mapping[draft], label=f'Draft {draft}')
                         for draft in sorted(unique_draft_sizes)]
    # Show line styles for target_attention
    line_style_legend = [plt.Line2D([0], [0], color='black', linestyle=style, label=f'colocate {val}')
                         for val, style in line_styles.items()]
    # Show marker styles for chunk_prefill
    marker_style_legend = [plt.Line2D([0], [0], color='black', marker=style, label=f'Target Atten {val}')
                           for val, style in marker_styles.items()]

    # Show the legends
    fig.legend(handles=draft_size_legend + line_style_legend + marker_style_legend,
               loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3)

    # Adjust layout and save the figure
    # Adjust the rectangle in which to fit the subplots
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(f'benchmark_results_{dataset}.png', bbox_inches='tight')
    plt.close(fig)
