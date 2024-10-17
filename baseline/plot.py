import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV data into a pandas DataFrame
# file_name = "baseline_10_06_NVIDIAA100-PCIE-40GB.csv"
# file_name = "baseline_10_06_NVIDIARTXA6000.csv"
# file_name = "chunked_prefill_NVIDIAA100-PCIE-40GB.csv"
file_name = "baseline_10_15_NVIDIAGeForceRTX3090.csv"
# file_name = "baseline_10_15_NVIDIAA100-PCIE-40GB.csv"
data = pd.read_csv(
    file_name,
    engine='python'  # Use the Python engine because C engine doesn't support 'on_bad_lines'
)
# Strip whitespace from all column names
data.columns = data.columns.str.strip()

# gpu_name,target_model,draft_model,dataset,temperature,request_rate,draft_size,prefill_schedule_mode,budget_token,budget_seq,colocate,consolidated_attention,drop_threshold,p50_ttft,p99_ttft,p50_tpot,p99_tpot,p50_token_latency,p99_token_latency,token_throughput,request_throughput,token_latency,preempt_flag

# Convert appropriate columns to numeric and boolean types
numeric_cols = [
    'temperature', 'request_rate', 'draft_size', 'p50_ttft', 'p99_ttft', 'p50_tpot', 'p99_tpot',
    'p50_token_latency', 'p99_token_latency', 'token_throughput', 'request_throughput', 'token_latency'
]

data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

boolean_cols = ['preempt_flag']
data[boolean_cols] = data[boolean_cols].astype(bool)

# Remove rows with missing data in the key columns we're using
data = data.dropna(subset=['token_throughput', 'token_latency', 'request_rate', 'draft_size'])

# Ensure Draft Size is an integer (after handling any NaN values)
data['draft_size'] = data['draft_size'].astype(int)

# Only include temperatures 0, 0.5, Random (-1)
temperatures = [0, 0.25, 0.5, 0.75, -1]
data = data[data['temperature'].isin(temperatures)]

# Map temperatures for labels
temperature_labels = {0: '0', 0.25:'0.25', 0.5: '0.5', 0.75: '0.75', -1: 'Random'}

# Define unique draft sizes and datasets
draft_sizes = sorted(data['draft_size'].unique())
datasets = data['dataset'].unique()

# Get the list of GPUs
gpus = data['gpu_name'].unique()

# budget_seq = 128
data = data[data['budget_seq'] == 64]

# Custom blueish and reddish color palette for the plot
blue_palette = ['#E74C3C', '#D7E2F9', '#88BCFF', '#3864B9', '#1B345F']

# Loop over each GPU and generate plots for all datasets and temperatures
for gpu in gpus:
    num_rows = len(datasets)
    num_cols = len(temperatures)
    
    # Create a figure with datasets as rows and temperatures as columns
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 2*num_rows), sharey=True, sharex=True)
    
    # Adjust axes to be a 2D array even if there's only one subplot
    if num_rows == 1:
        axes = [axes]

    if num_cols == 1:
        axes = [[ax] for ax in axes]

    # Determine global x and y limits
    x_min = data['token_throughput'].min()
    x_max = data['token_throughput'].max()
    y_min = data['token_latency'].min()
    y_max = data['token_latency'].max()

    # Set consistent x and y limits
    x_limits = (0, x_max + 200)
    y_limits = (10**(-2.2), 1.8)  # A100
    y_limits = (1e-2, 2.8)  # RTX A6000

    # Collect handles and labels for the legend
    handles_dict = {}

    # Define line styles or colors for each draft size
    draft_size_colors = {}
    for idx, draft_size in enumerate(sorted(draft_sizes)):
        draft_size_colors[draft_size] = blue_palette[idx % len(blue_palette)]

    # Loop over datasets (rows) and temperatures (columns) to fill subplots
    for row, dataset in enumerate(datasets):
        for col, temp in enumerate(temperatures):
            ax = axes[row, col] if num_rows > 1 else axes[col]
            gpu_dataset_data = data[(data['gpu_name'] == gpu) & (data['dataset'] == dataset)]
            temp_data = gpu_dataset_data[gpu_dataset_data['temperature'] == temp]
            if temp_data.empty:
                continue  # Skip if no data for this combination

            # Track which request rates have already been annotated
            annotated_req_rates = set()

            # Plot for each draft size
            for draft_size in draft_sizes:
                subset = temp_data[temp_data['draft_size'] == draft_size]
                if draft_size == 0:
                    # For draft size 0, since temperature doesn't affect performance, get data from any temperature
                    subset = gpu_dataset_data[gpu_dataset_data['draft_size'] == draft_size]
                    subset = subset.drop_duplicates(subset=['request_rate', 'token_throughput', 'token_latency'])
                    label = "AR"
                else:
                    subset = temp_data[temp_data['draft_size'] == draft_size]
                    label = f"Spec {draft_size}"

                if not subset.empty:
                    # Plot the main colored line
                    line_colored, = ax.plot(subset['token_throughput'], subset['token_latency'],
                                            label=label, marker='o', markersize=6, markerfacecolor=draft_size_colors[draft_size],
                                            markeredgewidth=0.5, markeredgecolor='black', color=draft_size_colors[draft_size],
                                            linestyle='-', linewidth=1.5)

                    # Collect handles and labels for the legend
                    if draft_size not in handles_dict:
                        handles_dict[draft_size] = line_colored

            # Set titles and labels
            temp_label = temperature_labels.get(temp, str(temp))
            if row == 0:
                ax.set_title(f"Temp: {temp_label}", fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_yscale('log')
            ax.tick_params(axis='both', labelsize=12)

    # Add a common legend
    handles = list(handles_dict.values())
    labels = [f'Spec {draft_size}' if draft_size != 0 else "AR" for draft_size in handles_dict.keys()]
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.02), fontsize=16, frameon=False, columnspacing=1.3)

    # Add a single shared label for the x-axis and y-axis
    fig.text(0.5, 0.03, 'Token Throughput (tokens/s)', ha='center', fontsize=20)
    fig.text(0.03, 0.5, 'Token Latency (s/token)', va='center', rotation='vertical', fontsize=20)

    # Add dataset names on the right side, as y-axis labels
    for row, dataset in enumerate(datasets):
        ax_right = axes[row, -1] if num_cols > 1 else axes[row]
        ax_right.annotate(dataset, xy=(1.05, 0.5), xycoords='axes fraction', ha='left', va='center', fontsize=16, rotation=90)

    # Adjust layout with no spaces between subplots
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.subplots_adjust(wspace=0, hspace=0)  # No space between subplots

    # Save the figure
    filename = f"main_{gpu}_datasets_temperatures.pdf".replace('/', '_').replace(' ', '_')
    plt.savefig(filename, bbox_inches='tight', format='pdf')
    plt.show()
