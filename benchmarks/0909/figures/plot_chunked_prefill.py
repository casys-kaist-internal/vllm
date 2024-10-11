import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File names for baseline and "ours"
baseline_file_name = "chunked_prefill_NVIDIAA100-PCIE-40GB.csv"
ours_file_name = "ours_chunked_prefill.csv"

# Read the baseline and "ours" CSV data into pandas DataFrames
baseline_data = pd.read_csv(baseline_file_name, engine='python')
ours_data = pd.read_csv(ours_file_name, engine='python')

# Strip whitespace from all column names and values in both datasets
baseline_data.columns = baseline_data.columns.str.strip()
ours_data.columns = ours_data.columns.str.strip()

# Strip leading and trailing whitespace from all string columns
baseline_data = baseline_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
ours_data = ours_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Rename columns for consistency between baseline and ours
rename_dict = {
    'GPU Name': 'gpu_name', 
    'Target Model': 'target_model', 
    'Draft Model': 'draft_model',
    'Dataset': 'dataset', 
    'Temperature': 'temperature', 
    'Request Rate': 'request_rate',
    'Draft Size': 'draft_size',
    'Request Throughput (reqs/s)': 'request_throughput',
    'P50 TTFT (s)': 'p50_ttft', 
    'P99 TPOT (s/token)': 'p99_tpot',
    'Budget Seq Len': 'budget_seq'
}

baseline_data = baseline_data.rename(columns=rename_dict)
ours_data = ours_data.rename(columns=rename_dict)

# Filter ours_data for budget_seq == 128 if 'budget_seq' exists
if 'budget_seq' in ours_data.columns:
    ours_data = ours_data[ours_data['budget_seq'] == 256]

# Convert appropriate columns to numeric for both datasets
numeric_cols = ['temperature', 'request_rate', 'draft_size', 'p50_ttft', 'p99_tpot', 'request_throughput']
baseline_data[numeric_cols] = baseline_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
ours_data[numeric_cols] = ours_data[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Remove rows with missing data in the key columns we're using
baseline_data = baseline_data.dropna(subset=['request_throughput', 'request_rate'])
ours_data = ours_data.dropna(subset=['request_throughput', 'request_rate'])

# Add a 'method' column to both datasets
baseline_data['method'] = 'Baseline'
ours_data['method'] = 'Ours'

# Duplicate baseline data for all temperatures (since it's constant across temperatures)
all_temperatures = ours_data['temperature'].unique()
baseline_data_all_temps = pd.concat([baseline_data.assign(temperature=temp) for temp in all_temperatures], ignore_index=True)

# Combine the two datasets
combined_data = pd.concat([baseline_data_all_temps, ours_data], ignore_index=True)

# Find the maximum request throughput for each dataset, temperature, and method
max_throughput = combined_data.groupby(['dataset', 'temperature', 'method'])['request_throughput'].max().reset_index()

# Calculate percentage improvement
improvement_df = max_throughput.pivot(index=['dataset', 'temperature'], columns='method', values='request_throughput').reset_index()
improvement_df['improvement'] = ((improvement_df['Ours'] - improvement_df['Baseline']) / improvement_df['Baseline']) * 100

# Plotting with Matplotlib

# Define unique datasets and temperatures
datasets = max_throughput['dataset'].unique()
temperatures = sorted(max_throughput['temperature'].unique())

# Determine subplot grid size
num_datasets = len(datasets)
num_temperatures = len(temperatures)
fig, axes = plt.subplots(num_datasets, num_temperatures, figsize=(6*num_temperatures, 5*num_datasets), 
                         squeeze=False)

# Define colors for methods
colors = {'Baseline': '#1f77b4', 'Ours': '#ff7f0e'}

# Iterate through each dataset and temperature to plot
for i, dataset in enumerate(datasets):
    for j, temp in enumerate(temperatures):
        ax = axes[i][j]
        subset = max_throughput[(max_throughput['dataset'] == dataset) & (max_throughput['temperature'] == temp)]
        
        methods = subset['method']
        throughput = subset['request_throughput']
        
        bars = ax.bar(methods, throughput, color=[colors[m] for m in methods], width=0.6)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        # Set titles and labels
        if i == 0:
            ax.set_title(f'Temperature: {temp}', fontsize=14, fontweight='bold')
        if j == 0:
            ax.set_ylabel(f'Dataset: {dataset}\nMax Throughput\n(reqs/s)', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
        
        # Set consistent y-axis limits
        ax.set_ylim(0, max_throughput['request_throughput'].max() * 1.1)
        
        # Annotate improvement
        improvement = improvement_df[(improvement_df['dataset'] == dataset) & (improvement_df['temperature'] == temp)]['improvement'].values
        if len(improvement) > 0:
            improvement = improvement[0]
            ax.text(0.5, 0.95, f'Improvement: {improvement:.1f}%', 
                    transform=ax.transAxes, ha='center', va='center', 
                    fontsize=12, color='green', fontweight='bold')

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Add a main title
plt.suptitle('Max Request Throughput Comparison by Dataset and Temperature', fontsize=20, fontweight='bold')

# Create a custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors['Baseline'], label='Baseline'),
                   Patch(facecolor=colors['Ours'], label='Ours')]
plt.legend(handles=legend_elements, loc='upper right', fontsize=14, title='Method', title_fontsize=16)

# Save the figure
plt.savefig('max_request_throughput_with_improvement_tighter.png', bbox_inches='tight', format='png')

# Show the plot
plt.show()
