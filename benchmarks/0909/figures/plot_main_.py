import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV data into pandas DataFrames
baseline_file_name = "baseline_10_06_NVIDIARTXA6000.csv"
baseline_file_name = "baseline_10_06_NVIDIARTXA6000.csv"

ours_file_name = "ours.csv"

# Load baseline data
data_baseline = pd.read_csv(
    baseline_file_name,
    engine='python'  # Use the Python engine because C engine doesn't support 'on_bad_lines'
)
data_baseline.columns = data_baseline.columns.str.strip()

# Load CoSpec data
data_ours = pd.read_csv(
    ours_file_name,
    engine='python'
)
data_ours.columns = data_ours.columns.str.strip()

# Rename columns in data_ours to match baseline data naming conventions
data_ours.rename(columns={
    'gpu_name': 'GPU Name',
    'target_model': 'Target Model',
    'draft_model': 'Draft Model',
    'dataset': 'Dataset',
    'temperature': 'Temperature',
    'request_rate': 'Request Rate',
    'draft_size': 'Draft Size',
    'p50_ttft': 'P50 TTFT (s)',
    'p99_ttft': 'P99 TTFT (s)',
    'p50_tpot': 'P50 TPOT (s/token)',
    'p99_tpot': 'P99 TPOT (s/token)',
    'p50_token_latency': 'P50 Token Latency (s/token)',
    'p99_token_latency': 'P99 Token Latency (s/token)',
    'token_throughput': 'Token Throughput (tokens/s)',
    'request_throughput': 'Request Throughput (reqs/s)',
    'token_latency': 'Token Latency (s/token)',
    'preempt_flag': 'Preempt Flag'
}, inplace=True)

# Standardize GPU names
standardize_gpu_name = lambda x: x.replace('_', '').strip() if isinstance(x, str) else x
data_baseline['GPU Name'] = data_baseline['GPU Name'].apply(standardize_gpu_name)
data_ours['GPU Name'] = data_ours['GPU Name'].apply(standardize_gpu_name)

# Ensure that numeric columns exist in the data before conversion
numeric_cols = [
    'Temperature', 'Request Rate', 'Draft Size', 'P50 TTFT (s)', 'P99 TTFT (s)', 'P50 TPOT (s/token)', 'P99 TPOT (s/token)',
    'P50 Token Latency (s/token)', 'P99 Token Latency (s/token)', 'Token Throughput (tokens/s)',
    'Request Throughput (reqs/s)', 'Token Latency (s/token)'
]

# Convert appropriate columns to numeric and boolean types for both datasets
for col in numeric_cols:
    if col in data_baseline.columns:
        data_baseline[col] = pd.to_numeric(data_baseline[col], errors='coerce')
    if col in data_ours.columns:
        data_ours[col] = pd.to_numeric(data_ours[col], errors='coerce')

boolean_cols = ['Preempt Flag']
for col in boolean_cols:
    if col in data_baseline.columns:
        data_baseline[col] = data_baseline[col].astype(bool)
    if col in data_ours.columns:
        data_ours[col] = data_ours[col].astype(bool)

# Remove rows with missing data in the key columns we're using (only if those columns exist)
required_cols = ['Token Throughput (tokens/s)', 'Token Latency (s/token)', 'Request Rate', 'Draft Size']
data_baseline = data_baseline.dropna(subset=[col for col in required_cols if col in data_baseline.columns])
data_ours = data_ours.dropna(subset=[col for col in required_cols if col in data_ours.columns])

# Ensure Draft Size is an integer (after handling any NaN values)
if 'Draft Size' in data_baseline.columns:
    data_baseline['Draft Size'] = data_baseline['Draft Size'].astype(int)
if 'Draft Size' in data_ours.columns:
    data_ours['Draft Size'] = data_ours['Draft Size'].astype(int)

# Define unique draft sizes and datasets
draft_sizes = sorted(data_baseline['Draft Size'].unique()) if 'Draft Size' in data_baseline.columns else []
datasets = data_baseline['Dataset'].unique() if 'Dataset' in data_baseline.columns else []

# Get the list of GPUs
gpus = data_baseline['GPU Name'].unique() if 'GPU Name' in data_baseline.columns else []

# Custom blueish and reddish color palette for the plot
blue_palette = ['#E74C3C', '#D7E2F9', '#88BCFF', '#3864B9', '#1B345F']
green_color = '#228B22'  # Green color for CoSpec

# Update temperature list to include 0, 0.25, 0.5, 0.75, and random (-1)
temperatures = [0, 0.25, 0.5, 0.75, -1]

# Map temperatures for labels
temperature_labels = {0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', -1: 'Random'}

# Define the number of rows and columns based on the number of temperatures and datasets
num_rows = len(temperatures)  # 5 rows (for each temperature)
num_cols = len(datasets)      # 3 columns (for each dataset)


# Loop over each GPU and generate plots for all datasets and temperatures
for gpu in gpus:
    num_rows = len(datasets)
    num_cols = len(temperatures)
    
    # Create a figure with datasets as rows and temperatures as columns
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 2.5*num_rows), sharey=False, sharex=True)
    
    # Adjust axes to be a 2D array even if there's only one subplot
    if num_rows == 1:
        axes = [axes]

    # Determine global x and y limits
    x_min = min(data_baseline['Token Throughput (tokens/s)'].min(), data_ours['Token Throughput (tokens/s)'].min())
    x_max = max(data_baseline['Token Throughput (tokens/s)'].max(), data_ours['Token Throughput (tokens/s)'].max())
    y_min = min(data_baseline['Token Latency (s/token)'].min(), data_ours['Token Latency (s/token)'].min())
    y_max = max(data_baseline['Token Latency (s/token)'].max(), data_ours['Token Latency (s/token)'].max())

    # Set consistent x and y limits
    x_limits = (0, x_max + 200)
    y_limits = [(10**(-2.1), 0.8) , (10**(-2.1), 0.93) , (10**(-2), 2.3)]

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
            gpu_dataset_data_baseline = data_baseline[(data_baseline['GPU Name'] == gpu) & (data_baseline['Dataset'] == dataset)]
            gpu_dataset_data_ours = data_ours[(data_ours['GPU Name'] == gpu) & (data_ours['Dataset'] == dataset)]
            temp_data_baseline = gpu_dataset_data_baseline[gpu_dataset_data_baseline['Temperature'] == temp]
            
            temp_data_ours = gpu_dataset_data_ours[gpu_dataset_data_ours['Temperature'] == temp]

            # Define line styles, colors, and markers for each draft size (Speculative Labels)
            draft_size_colors = {}
            draft_size_markers = {1: 'D', 3: 'D', 5: 'D', 7: 'D'}  # Different markers for Speculative sizes
            for idx, draft_size in enumerate(sorted(draft_sizes)):
                draft_size_colors[draft_size] = blue_palette[idx % len(blue_palette)]  # Colors for Speculative
                if draft_size not in draft_size_markers:
                    draft_size_markers[draft_size] = 'o'  # Default marker style for other sizes
                    
            # Plot baseline data with black edges for lines (dual line approach)
            for draft_size in draft_sizes:
                # Skip AR (draft_size == 0) for now, to plot it last
                if draft_size == 0:
                    continue  # We will plot AR after speculative lines
                
                subset = temp_data_baseline[temp_data_baseline['Draft Size'] == draft_size]
                
                label = f"Spec {draft_size}"
                if not subset.empty:
                    # First, plot a thicker black line as the "edge"
                    ax.plot(subset['Token Throughput (tokens/s)'], subset['Token Latency (s/token)'],
                            label=None, linestyle='-', linewidth=2, color='black', alpha=0.7)

                    # Then, overlay the actual colored line on top of the black edge
                    ax.plot(subset['Token Throughput (tokens/s)'], subset['Token Latency (s/token)'],
                            label=label, marker=draft_size_markers[draft_size], markersize=5,
                            markerfacecolor=draft_size_colors[draft_size], markeredgewidth=0.5, markeredgecolor='black', 
                            color=draft_size_colors[draft_size], linestyle='-', linewidth=1.5)

                    # Collect handles and labels for the legend
                    if draft_size not in handles_dict:
                        handles_dict[draft_size] = ax.plot([], [], label=label, color=draft_size_colors[draft_size], 
                                                        marker=draft_size_markers[draft_size], linestyle='-', 
                                                        linewidth=3, markeredgewidth=1, markeredgecolor='black', markersize=10)[0]

            # Now, plot AutoRegressive (AR) on top
            subset_ar = gpu_dataset_data_baseline[gpu_dataset_data_baseline['Draft Size'] == 0]
            if not subset_ar.empty:
                label_ar = "AR"
                
                # First, plot the black edge for AR
                ax.plot(subset_ar['Token Throughput (tokens/s)'], subset_ar['Token Latency (s/token)'],
                        label=None, linestyle='-', linewidth=2, color='black', alpha=0.7)

                # Then, overlay the colored line for AR
                ax.plot(subset_ar['Token Throughput (tokens/s)'], subset_ar['Token Latency (s/token)'],
                        label=label_ar, marker='o', markersize=5,
                        markerfacecolor='red', markeredgewidth=0.5, markeredgecolor='black', 
                        color='red', linestyle='-', linewidth=1.5)

                # Add AR to the legend if not already added
                if 0 not in handles_dict:
                    handles_dict[0] = ax.plot([], [], label=label_ar, color='red', linestyle='-', marker='o', linewidth=3, markeredgewidth=1, markeredgecolor='black', markersize=10)[0]


            # Plot CoSpec data with black edges for lines (dual line approach)
            if not temp_data_ours.empty:
                # Plot the black edge first
                ax.plot(temp_data_ours['Token Throughput (tokens/s)'], temp_data_ours['Token Latency (s/token)'],
                        label=None, linestyle='-', linewidth=2, color='black', alpha=0.7)

                # Overlay the green CoSpec line on top
                ax.plot(temp_data_ours['Token Throughput (tokens/s)'], temp_data_ours['Token Latency (s/token)'],
                        label='CoSpec', marker='s', markersize=5, markerfacecolor=green_color, markeredgewidth=0.5, markeredgecolor='black',
                        color=green_color, linestyle='-', linewidth=1.5)

                # Add CoSpec to the legend if not already added
                if 'CoSpec' not in handles_dict:
                    handles_dict['CoSpec'] = ax.plot([], [], label='CoSpec', color=green_color, linestyle='-', 
                                                    marker='s', linewidth=3, markeredgewidth=1, markeredgecolor='black', markersize=10)[0]

            # Set titles and labels
            temp_label = temperature_labels.get(temp, str(temp))
            if row == 0:
                ax.set_title(f"Temp: {temp_label}", fontsize=16)  # Column headers as dataset names
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(x_limits)
            y_limit = y_limits[row]
            print("y_limit", y_limit)
            ax.set_ylim(y_limit)
            ax.set_yscale('log')
            ax.tick_params(axis='both', labelsize=12)

            # only first column should have y tick labels
            if col != 0:
                ax.set_yticklabels([])

    # Add a common legend
    handles = []
    labels = []

    # Ensure AR is drawn last by plotting AR after all speculative lines
    if 0 in handles_dict:
        handles.append(handles_dict[0])  # AR handle
        labels.append("AR")  # AR label

    # Add Speculative sizes in order (1, 3, 5, 7)
    for draft_size in [1, 3, 5, 7]:
        if draft_size in handles_dict:
            handles.append(handles_dict[draft_size])  # Spec handle
            labels.append(f'Spec {draft_size}')  # Spec label

    # Add CoSpec at the end
    if 'CoSpec' in handles_dict:
        handles.append(handles_dict['CoSpec'])  # CoSpec handle
        labels.append('CoSpec')  # CoSpec label

    # Create the legend
    fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.02), fontsize=18, frameon=False, columnspacing=1.3)

    # Add a single shared label for the x-axis and y-axis
    fig.text(0.5, 0.03, 'Token Throughput (tokens/s)', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Token Latency (s/token)', va='center', rotation='vertical', fontsize=20)

    # Add dataset names on the right side, as y-axis labels
    dataset_rename_map = {
        'finance': 'Finance',
        'sharegpt': 'ShareGPT',
        'gsm8k': 'GSM8K'
    }

    # Add dataset names on the right side, as y-axis labels
    for row, dataset in enumerate(datasets):
        dataset_label = dataset_rename_map.get(dataset.lower(), dataset)  # Rename datasets if present in the map
        ax_right = axes[row, -1] if num_cols > 1 else axes[row]
        ax_right.annotate(dataset_label, xy=(1.05, 0.5), xycoords='axes fraction', ha='left', va='center', fontsize=16, rotation=90)

    # Adjust layout with no spaces between subplots
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # No space between subplots

    # Save the figure
    filename = f"main_{gpu}_datasets_temperatures.pdf".replace('/', '_').replace(' ', '_')
    plt.savefig(filename, bbox_inches='tight', format='pdf')
    plt.show()