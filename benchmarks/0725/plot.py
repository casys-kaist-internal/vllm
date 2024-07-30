import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('1_benchmark_results_0730.csv')

# If prefill_schedule_mode is chunked_prefill, set budget to 512
df.loc[df['prefill_schedule_mode'] == 'chunked_prefill', 'budget_token'] = 512

# Draw only temperature 0
# df = df[df['temperature'] == 0.5]

# Define lists for subplot division and labeling
# Controls how subplots are divided
# subplot_division_features = ['budget_token', 'budget_seq', 'temperature']
subplot_division_features = ['temperature']

# Controls labeling within subplots
labeling_features = ['draft_size', 'colocate',
                     'target_attention', 'drop_threshold', 'prefill_schedule_mode']

# Generate a composite key for subplot division and labeling
df['subplot_key'] = df.apply(lambda row: ' | '.join(
    [f"{col}={row[col]}" for col in subplot_division_features]), axis=1)
df['label_key'] = df.apply(lambda row: '|'.join(
    [f"{col}={row[col]}" for col in labeling_features]), axis=1)

# Define the metrics to plot (x-axis and y-axis variables)
metric_x = 'request_throughput'
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
# line_styles = {'0.0': '--', '0.25': '-', '0.5': '-.', '0.75': ':'}
# marker_styles = {'True': 'o', 'False': 'x'}
# marker_styles = {'0': 'o', '0.25': 'x', '0.5': 's', '0.75': 'D'}
marker_styles = {'0': 'o', '0.1': 'x', '0.25': 's', '0.5': 'D', '0.75': 'P'}


def plot_data(df, x, y, ax):
    # Sort configurations by draft_size numerically and target_attention alphabetically
    unique_configs = sorted(df['label_key'].unique(), key=lambda x: (
        int(x.split('|')[0].split('=')[1]), x.split('|')[1].split('=')[1]))

    for config in unique_configs:
        print(config)
        draft_size = config.split('|')[0].split('=')[1]
        colocate = config.split('|')[1].split('=')[1]
        target_attention = config.split('|')[2].split('=')[1]
        drop_threshold = config.split('|')[3].split('=')[1]
        prefill_schedule_mode = config.split('|')[4].split('=')[1]

        subset = df[df['label_key'] == config]
        # color = color_mapping[int(draft_size)]  # Get color for draft_size
        # Get line style for target_attention
        line_style = line_styles[colocate]
        marker_style = marker_styles[drop_threshold]

        # color with prefill_schedule_mode which is one of 3 ['full_prefill', 'chunked_prefill', 'chunked_prefill_demote_draft']
        if prefill_schedule_mode == 'full_prefill':
            color = 'blue'
        elif prefill_schedule_mode == 'chunked_prefill':
            color = 'red'
        elif prefill_schedule_mode == 'chunked_prefill_demote_draft':
            color = 'green'
        elif prefill_schedule_mode == '1_process':
            color = 'orange'
        elif prefill_schedule_mode == '2_process':
            color = 'purple'

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

    ax.set_xlabel(x.replace('_', ' ').title())
    ax.set_ylabel(y.replace('_', ' ').title())
    ax.legend()
    ax.grid(True)
    # ax.set_xlim(left=0)  # Modify this to set the minimum x value


# Loop through unique datasets and dynamically generate plots based on subplot keys
for dataset in df['dataset'].unique():
    subset_df = df[df['dataset'] == dataset]
    unique_subplot_keys = subset_df['subplot_key'].unique()

    # Create a subplot for each unique subplot key
    n_cols = len(unique_subplot_keys)
    fig, axes = plt.subplots(1, n_cols, figsize=(
        10 * n_cols, 6), sharey=True, sharex=True)

    # If only one subplot, make sure 'axes' is iterable
    if n_cols == 1:
        axes = [axes]

    # Plot data for each unique subplot division
    for ax, key in zip(axes, unique_subplot_keys):
        ax.set_title(
            f"({key})")
        config_df = subset_df[subset_df['subplot_key'] == key]
        plot_data(config_df, metric_x, metric_y, ax)

    # Custom legend
    # Show available draft sizes and their corresponding colors
    draft_size_legend = [plt.Line2D([0], [0], color=color_mapping[draft], label=f'Draft {draft}')
                         for draft in sorted(unique_draft_sizes)]
    # Show line styles for target_attention
    line_style_legend = [plt.Line2D([0], [0], color='black', linestyle=style, label=f'Colocate {val}')
                         for val, style in line_styles.items()]
    # Show marker styles for chunk_prefill
    marker_style_legend = [plt.Line2D([0], [0], color='black', marker=style, label=f'Drop threshold {val}')
                           for val, style in marker_styles.items()]

    # prefill_schedule_legend = [plt.Line2D([0], [0], color=color, label=prefill_schedule_mode)
    #                            for prefill_schedule_mode, color in zip(['full_prefill', 'chunked_prefill', 'chunked_prefill_demote_draft'], ['blue', 'red', 'green'])]

    prefill_schedule_legend = [plt.Line2D([0], [0], color=color, label=prefill_schedule_mode)
                               for prefill_schedule_mode, color in zip(['1_process', '2_process'], ['orange', 'purple'])]

    # Show the legends
    # fig.legend(handles=draft_size_legend + line_style_legend,
    #            loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    # fig.legend(handles=draft_size_legend + line_style_legend + marker_style_legend,
    #            loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    # fig.legend(handles=draft_size_legend, loc='upper center',
    #            bbox_to_anchor=(0.5, 1.1), ncol=3)
    fig.legend(handles=prefill_schedule_legend, loc='upper center',
               bbox_to_anchor=(0.5, 1.1), ncol=3)

    # start x axis from 0
    for ax in axes:
        ax.set_xlim(left=0)

    # grid step 0.5
    for ax in axes:
        ax.set_xticks(np.arange(0, 12, 0.5))
        ax.set_yticks(np.arange(0, 0.5, 0.1))

    # Adjust layout and save the figure
    # Adjust the rectangle in which to fit the subplots
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(f'benchmark_results_{dataset}.png', bbox_inches='tight')
    plt.close(fig)

for metric in ['p50_ttft', 'p99_ttft', 'p50_tpot', 'p99_tpot', 'p50_tpt', 'p99_tpt']:
    # Loop through unique datasets and plot p50_ttft, p99_ttft, p50_tpot, p99_tpot, p50_tpt, p99_tpt
    for dataset in df['dataset'].unique():
        subset_df = df[df['dataset'] == dataset]
        unique_subplot_keys = subset_df['subplot_key'].unique()

        # Create a subplot for each unique subplot key
        n_cols = len(unique_subplot_keys)
        fig, axes = plt.subplots(1, n_cols, figsize=(
            8 * n_cols, 8), sharey=True, sharex=True)

        # If only one subplot, make sure 'axes' is iterable
        if n_cols == 1:
            axes = [axes]

        # Plot data for each unique subplot division
        for ax, key in zip(axes, unique_subplot_keys):
            ax.set_title(
                f"({key})")
            config_df = subset_df[subset_df['subplot_key'] == key]
            plot_data(config_df, 'request_rate', metric, ax)

        # Custom legend
        # Show available draft sizes and their corresponding colors
        draft_size_legend = [plt.Line2D([0], [0], color=color_mapping[draft], label=f'Draft {draft}')
                             for draft in sorted(unique_draft_sizes)]
        # Show line styles for target_attention
        line_style_legend = [plt.Line2D([0], [0], color='black', linestyle=style, label=f'colocate {val}')
                             for val, style in line_styles.items()]
        # Show marker styles for chunk_prefill
        marker_style_legend = [plt.Line2D([0], [0], color='black', marker=style, label=f'Attention {val}')
                               for val, style in marker_styles.items()]

        # Show the legends
        fig.legend(handles=draft_size_legend + line_style_legend + marker_style_legend,
                   loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

        # start x axis from 0
        for ax in axes:
            ax.set_xlim(left=0)

        # Adjust layout and save the figure
        # Adjust the rectangle in which to fit the subplots
        # plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(
            f'benchmark_results_{metric}_{dataset}.png', bbox_inches='tight')
