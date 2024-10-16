import pandas as pd
import matplotlib.pyplot as plt

# Load baseline data
data_baseline = pd.read_csv("full_prefill/baseline.csv")
data_ours = pd.read_csv("full_prefill/ours.csv")

# Standardize GPU names
standardize_gpu_name = lambda x: x.replace('_', '').strip() if isinstance(x, str) else x
data_baseline['gpu_name'] = data_baseline['gpu_name'].apply(standardize_gpu_name)
data_ours['gpu_name'] = data_ours['gpu_name'].apply(standardize_gpu_name)

# Create pairs of dataset and GPU from the data itself
pairs_baseline = data_baseline[['gpu_name', 'dataset']].drop_duplicates().values.tolist()
pairs_ours = data_ours[['gpu_name', 'dataset']].drop_duplicates().values.tolist()

# Ensure that the pairs match between both datasets
pairs = [pair for pair in pairs_baseline if pair in pairs_ours]

# Define unique draft sizes, temperatures, and labels
draft_sizes = [1, 3, 5, 7]  # Only these draft sizes
temperatures = [0, 0.25, 0.5, 0.75, -1]
temperature_labels = {0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', -1: 'Random'}

# Define colors and markers
red_color = '#E74C3C'  # Red color for AR
blue_palette = ['#D7E2F9', '#88BCFF', '#3864B9', '#1B345F']  # Blueish colors for Speculative
green_color = '#228B22'  # Green color for CoSpec

# Set figure size based on rows and columns
fig, axes = plt.subplots(len(pairs), 5, figsize=(20, 12), sharey=False, sharex=False)

# Define y-limits for each row
# y_limits = [(10**(-2.1), 0.8), (10**(-2.1), 0.93), (10**(-2), 2.3)]
# x_limits = (0, max(data_baseline['token_throughput'].max(), data_ours['token_throughput'].max()) + 200)

# Predefine handles and labels for the legend
legend_handles = []
legend_labels = []

# Loop over dataset-GPU pairs and temperatures to plot the data
for row, (gpu, dataset) in enumerate(pairs):
    gpu_dataset_data_baseline = data_baseline[(data_baseline['gpu_name'] == gpu) & (data_baseline['dataset'] == dataset)]
    gpu_dataset_data_ours = data_ours[(data_ours['gpu_name'] == gpu) & (data_ours['dataset'] == dataset)]

    print(gpu_dataset_data_baseline)

    for col, temp in enumerate(temperatures):
        ax = axes[row, col]
        temp_data_baseline = gpu_dataset_data_baseline[gpu_dataset_data_baseline['temperature'] == temp]
        temp_data_ours = gpu_dataset_data_ours[gpu_dataset_data_ours['temperature'] == temp]

        # Plot baseline speculative data (with colored lines)
        for idx, draft_size in enumerate(draft_sizes):
            subset = temp_data_baseline[temp_data_baseline['draft_size'] == draft_size]
            if not subset.empty:
                line, = ax.plot(subset['token_throughput'], subset['token_latency'],
                                marker='D', markersize=5, color=blue_palette[idx],
                                linestyle='-', label=f"Spec {draft_size}", )
                if row == 0 and col == 0 and draft_size not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(f"Spec {draft_size}")

        # Plot baseline AR data (red)
        subset_ar = gpu_dataset_data_baseline[gpu_dataset_data_baseline['draft_size'] == 0]

        if not subset_ar.empty:

            line, = ax.plot(subset_ar['token_throughput'], subset_ar['token_latency'],
                            marker='o', markersize=5, color=red_color, linestyle='-', label='AR')
            if row == 0 and col == 0 and 'AR' not in legend_labels:
                legend_handles.append(line)
                legend_labels.append('AR')

        # Plot CoSpec data (green)
        if not temp_data_ours.empty:
            line, = ax.plot(temp_data_ours['token_throughput'], temp_data_ours['token_latency'],
                            marker='s', markersize=5, color=green_color, linestyle='-', label='CoSpec')
            if row == 0 and col == 0 and 'CoSpec' not in legend_labels:
                legend_handles.append(line)
                legend_labels.append('CoSpec')

        # Set limits, grid, and labels
        # ax.set_xlim(x_limits)
        # ax.set_ylim(y_limits[row])
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f"Temp: {temperature_labels[temp]}", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        if col != 0:
            ax.set_yticklabels([])

    # Annotate dataset and GPU name on the right side
    axes[row, -1].annotate(f'{dataset} / {gpu}', xy=(1.05, 0.5), xycoords='axes fraction',
                           ha='left', va='center', fontsize=16, rotation=90)

# Add x and y labels
fig.text(0.5, 0.03, 'Token Throughput (tokens/s)', ha='center', fontsize=18)
fig.text(0.04, 0.5, 'Token Latency (s/token)', va='center', rotation='vertical', fontsize=18)

# Add a common legend
fig.legend(legend_handles, legend_labels, loc='upper center', ncol=6, fontsize=14, frameon=False, columnspacing=1.3)

# Adjust layout and save as PNG
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig("full_prefill_e2e.png", format='png', bbox_inches='tight')
plt.show()
