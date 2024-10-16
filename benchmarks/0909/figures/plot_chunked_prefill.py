import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the CSV file for main data
file_path_main = 'chunked_prefill_A6000_ours.csv'
data_main = pd.read_csv(file_path_main)

# Load the CSV file for draft size 0 data
file_path_draft0 = 'chunked_prefill_A6000_draft0.csv'
data_draft0 = pd.read_csv(file_path_draft0)

# Rename the columns of draft size 0 data to match the main dataset
data_draft0 = data_draft0.rename(columns={
    'Result': 'result',
    'GPU Name': 'gpu_name',
    'Target Model': 'target_model',
    'Draft Model': 'draft_model',
    'Dataset': 'dataset',
    'Temperature': 'temperature',
    'Request Rate': 'request_rate',
    'Draft Size': 'draft_size',
    'Request Throughput (reqs/s)': 'request_throughput',
    'Token Throughput (tokens/s)': 'token_throughput',
    'Token Latency (s/token)': 'token_latency',
    'P50 TTFT (s)': 'p50_ttft',
    'P99 TTFT (s)': 'p99_ttft',
    'P50 TPOT (s/token)': 'p50_tpot',
    'P99 TPOT (s/token)': 'p99_tpot',
    'P50 Token Latency (s/token)': 'p50_token_latency',
    'P99 Token Latency (s/token)': 'p99_token_latency',
    'Preempt Flag': 'preempt_flag'
})

# Append draft size 0 data to the main data
data_draft0['draft_size'] = 0  # Ensure the draft size is set to 0 for this data
data = pd.concat([data_main, data_draft0])

autoregressive_data = data[data['draft_size'] == 0]
max_autoregressive_throughput = autoregressive_data['request_throughput'].max()

# Define the temperatures you want to plot
# selected_temperatures = [0, 0.25, 0.5, 0.75]
selected_temperatures = [0, 0.5, -1]

# Define the configuration order and labels
config_order = ['Autoregressive', 'Spec 1', 'Spec 3', 'Spec 5', 'Spec 7', 'CoSpec']

# Set up the figure with 10 subplots in a 2-row format
fig, axs = plt.subplots(2, len(selected_temperatures), figsize=(len(selected_temperatures) * 3.5, 5), sharey='row')  # Adjust figure size for compact yet readable design, share y-axis for each row
fig.tight_layout(pad=3.0)

# Custom color palette
colors = ['#d62728', '#D7E2F9', '#88BCFF', '#3864B9', '#1B345F', '#228B22']  # Red for Autoregressive, blueish for Spec, green for CoSpec

# Loop through the selected temperatures and create bar plots for max throughput and P99 TPOT
for idx, temperature in enumerate(selected_temperatures):
    row1, col1 = 0, idx  # Determine subplot location for max throughput
    row2, col2 = 1, idx  # Determine subplot location for P99 TPOT
    ax1 = axs[row1, col1]  # Access the subplot for max throughput
    ax2 = axs[row2, col2]  # Access the subplot for P99 TPOT
    
    # Filter data for the specific temperature
    subset = data[data['temperature'] == temperature]
    
    # Initialize max throughput and corresponding P99 TPOT per configuration
    max_throughput_per_config = []
    p99_tpot_per_config = []

    # Add Autoregressive (same for all temperatures)
    autoregressive_throughput = max_autoregressive_throughput
    max_throughput_per_config.append(autoregressive_throughput)
    p99_tpot_per_config.append(autoregressive_data['p99_tpot'].max())
    
    # Add Spec 1, 3, 5, 7 configurations
    for draft_size in [1, 3, 5, 7]:
        filtered_data = subset[(subset['draft_size'] == draft_size) & (subset['colocate'] == False)]
        if not filtered_data.empty:
            max_throughput = filtered_data['request_throughput'].max()
            max_throughput_per_config.append(max_throughput)
            p99_tpot_per_config.append(filtered_data[filtered_data['request_throughput'] == max_throughput]['p99_tpot'].values[0])
        else:
            max_throughput_per_config.append(0)
            p99_tpot_per_config.append(0)
    
    # Add CoSpec (draft size 7, colocation enabled, drop_threshold 0.3, consolidated_attention True)
    cospec_data = subset[(subset['draft_size'] == 7) & 
                         (subset['colocate'] == True) & 
                         (subset['drop_threshold'] == 0.3) & 
                         (subset['consolidated_attention'] == True)]
    if not cospec_data.empty:
        max_throughput = cospec_data['request_throughput'].max()
        max_throughput_per_config.append(max_throughput)
        p99_tpot_per_config.append(cospec_data[cospec_data['request_throughput'] == max_throughput]['p99_tpot'].values[0])
    else:
        max_throughput_per_config.append(0)
        p99_tpot_per_config.append(0)
    
    # Create bar graph for max throughput with color-coding and wider bars
    bars1 = ax1.bar(config_order, max_throughput_per_config, color=colors, edgecolor='black')
    
    # Set labels and title for max throughput subplot
    temperature_label = temperature if temperature >= 0 else 'Random'
    ax1.set_title(f'Temp: {temperature_label}', fontsize=16)
    ax1.set_ylim(0, max(max_throughput_per_config) * 1.33)  # Add padding to the Y-axis
    ax1.grid(True, axis='y', linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove x ticks
    if col1 == 0:
        ax1.set_ylabel('Capacity (reqs/s)', fontsize=14, labelpad=20)
    ax1.yaxis.set_label_coords(-0.16, 0.5)
    
    # Add data labels above the bars with speedup annotations for max throughput
    for i, v in enumerate(max_throughput_per_config):
        if v > 0:
            speedup = v / autoregressive_throughput if autoregressive_throughput > 0 else 0
            ax1.text(i, v + (max(max_throughput_per_config) * 0.04), 
                     f'{speedup:.2f}x', ha='center', fontsize=12, color='black')
    
    # Create bar graph for P99 TPOT with color-coding and wider bars
    bars2 = ax2.bar(config_order, p99_tpot_per_config, color=colors, edgecolor='black')
    
    # Set labels and title for P99 TPOT subplot
    ax2.set_ylim(0, max(p99_tpot_per_config) * 1.15)  # Add padding to the Y-axis
    ax2.grid(True, axis='y', linestyle='--', linewidth=0.5)
    if col2 == 0:
        ax2.set_ylabel('P99 TPOT (s/token)', fontsize=14, labelpad=20)
    ax2.yaxis.set_label_coords(-0.16, 0.5)

    # remove x tickx
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    # Add data labels above the bars with speedup annotations for P99 TPOT
    for i, v in enumerate(p99_tpot_per_config):
        if v > 0:
            speedup = autoregressive_data['p99_tpot'].max() / v if v > 0 else 0
            ax2.text(i, v + (max(p99_tpot_per_config) * 0.04), 
                     f'{speedup:.2f}x', ha='center', fontsize=12, color='black')

# Adjust the layout for a two-column paper format
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Add a global legend
from matplotlib.lines import Line2D
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
labels = ['AR', 'Spec 1', 'Spec 3', 'Spec 5', 'Spec 7', 'CoSpec']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=6, frameon=False, fontsize=15)

# Save the figure with high resolution suitable for publication
plt.savefig('chunked_prefill_e2e.pdf',  bbox_inches='tight', format='pdf', dpi=300)
plt.show()