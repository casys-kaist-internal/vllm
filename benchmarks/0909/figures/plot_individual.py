import pandas as pd
import matplotlib.pyplot as plt
import sys

# -----------------------------
# 1. Load and Process Main Data
# -----------------------------

# Load the main CSV data with error handling for bad lines
file_path = "ours_individual.csv"  # Replace with the actual file path
try:
    df = pd.read_csv(file_path, on_bad_lines='warn')  # For pandas >=1.3.0
except FileNotFoundError:
    print(f"File not found: {file_path}. Please check the file path.")
    sys.exit(1)
except pd.errors.EmptyDataError:
    print(f"No data: {file_path} is empty.")
    sys.exit(1)
except pd.errors.ParserError:
    print(f"Parsing error: Check the CSV format in {file_path}.")
    sys.exit(1)

# Define required columns
required_columns = ['colocate', 'drop_threshold', 'consolidated_attention',
                    'token_throughput', 'token_latency', 'temperature', 'request_rate']

# Drop rows with missing values in required columns
df = df.dropna(subset=required_columns)

# Ensure correct data types for boolean columns
bool_columns = ['colocate', 'consolidated_attention', 'preempt_flag']
for col in bool_columns:
    if col in df.columns:
        df[col] = df[col].astype(bool)
    else:
        print(f"Warning: Column '{col}' not found in the dataset.")

# Filter the DataFrame for a specific request rate
request_rate = 8  # Modify as needed
df_filtered = df[df['request_rate'] == request_rate].copy()  # Use .copy() to avoid SettingWithCopyWarning

# Check if df_filtered is empty
if df_filtered.empty:
    print(f"No data found for request_rate = {request_rate}. Please check the request_rate value.")
    sys.exit(1)

# Define the technique order with cumulative application, including 'Autoregressive'
technique_order = [
    'B',
    'C',
    'C+SV',
    'C+SV+CA',
    'Autoregressive'  # Add 'Autoregressive' to the technique order
]

# Define the technique determination function
def determine_technique(row):
    if not row['colocate'] and row['drop_threshold'] == 0 and not row['consolidated_attention']:
        return 'B'
    if row['colocate'] and row['drop_threshold'] == 0 and not row['consolidated_attention']:
        return 'C'
    if not row['colocate'] and row['drop_threshold'] > 0 and not row['consolidated_attention']:
        return 'SV'
    if not row['colocate'] and row['drop_threshold'] ==  0 and row['consolidated_attention']:
        return 'CA'
    if row['colocate'] and row['drop_threshold'] > 0 and not row['consolidated_attention']:
        return 'C+SV'
    if row['colocate'] and  row['drop_threshold'] == 0 and row['consolidated_attention']:
        return 'C+CA'
    if not row['colocate'] and  row['drop_threshold'] > 0 and row['consolidated_attention']:
        return 'SV+CA'
    if row['colocate'] and  row['drop_threshold'] > 0 and row['consolidated_attention']:
        return 'C+SV+CA'
    # Handle unexpected combinations
    return 'Unknown'

# Apply the determine_technique function
df_filtered['Technique'] = df_filtered.apply(determine_technique, axis=1)

# Ensure 'Technique' is a categorical type with the defined extended order
df_filtered['Technique'] = pd.Categorical(df_filtered['Technique'],
                                         categories=technique_order,
                                         ordered=True)

# Verify Technique labels and identify any mismatches
generated_techniques = df_filtered['Technique'].unique()
missing_techniques = set(generated_techniques) - set(technique_order)

if missing_techniques:
    print(f"Warning: The following Technique labels are not in technique_order and will be treated as NaN: {missing_techniques}")

print("Technique Distribution:")
print(df_filtered['Technique'].value_counts())

# Drop rows with NaN in 'Technique' after categorization
df_filtered = df_filtered.dropna(subset=['Technique'])

# Ensure that Baseline ('B') exists for each temperature
baseline_df = df_filtered[df_filtered['Technique'] == 'B'][['temperature', 'token_throughput', 'token_latency']]
baseline_df = baseline_df.rename(columns={
    'token_throughput': 'baseline_throughput',
    'token_latency': 'baseline_latency'
})

if baseline_df.empty:
    print("Error: No baseline ('B') entries found in the dataset for the specified request rate.")
    sys.exit(1)

# Merge baseline metrics back to the filtered dataframe based on temperature
df_merged = pd.merge(df_filtered, baseline_df, on='temperature', how='left', suffixes=('', '_baseline'))

# Check for any missing baseline values after merge
if df_merged['baseline_throughput'].isnull().any() or df_merged['baseline_latency'].isnull().any():
    print("Error: Some Temperature entries do not have corresponding Baseline ('B') metrics.")
    problematic_rows = df_merged[df_merged['baseline_throughput'].isnull() | df_merged['baseline_latency'].isnull()]
    print(problematic_rows[['temperature', 'Technique']])
    sys.exit(1)

# Compute speedup metrics
df_merged['speedup_throughput'] = df_merged['token_throughput'] / df_merged['baseline_throughput']
df_merged['speedup_latency'] = df_merged['baseline_latency'] / df_merged['token_latency']

# Optionally, handle cases where baseline_throughput or baseline_latency is zero to avoid division by zero
df_merged.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
df_merged = df_merged.dropna(subset=['speedup_throughput', 'speedup_latency'])

# -----------------------------
# 2. Load and Process Autoregressive Data
# -----------------------------

# Read the autoregressive CSV file
autoreg_file_path = "baseline_10_06_NVIDIARTXA6000.csv"  # Replace with actual path

try:
    df_autoreg = pd.read_csv(autoreg_file_path, on_bad_lines='warn')
except FileNotFoundError:
    print(f"Autoregressive file not found: {autoreg_file_path}. Please check the file path.")
    sys.exit(1)
except pd.errors.EmptyDataError:
    print(f"No data: {autoreg_file_path} is empty.")
    sys.exit(1)
except pd.errors.ParserError:
    print(f"Parsing error: Check the CSV format in {autoreg_file_path}.")
    sys.exit(1)

# Define required columns for autoregressive data
required_columns_autoreg = ['Temperature', 'Request Rate', 'Dataset', 'Draft Size',
                            'Token Throughput (tokens/s)', 'Token Latency (s/token)']

# Drop rows with missing values in required columns
df_autoreg = df_autoreg.dropna(subset=required_columns_autoreg)

# Filter for Request Rate == 8 and Dataset == 'sharegpt'
df_autoreg = df_autoreg[(df_autoreg['Request Rate'] == request_rate) & (df_autoreg['Dataset'] == 'sharegpt')]

# Rename columns to match the main dataframe
df_autoreg.rename(columns={
    'Temperature': 'temperature',
    'Request Rate': 'request_rate',
    'Token Throughput (tokens/s)': 'token_throughput_autoreg',
    'Token Latency (s/token)': 'token_latency_autoreg'
}, inplace=True)

# Ensure correct data types for boolean columns, if present
bool_columns_autoreg = ['Preempt Flag']  # Adjust based on actual boolean columns
for col in bool_columns_autoreg:
    if col in df_autoreg.columns:
        df_autoreg[col] = df_autoreg[col].astype(bool)
    else:
        print(f"Warning: Column '{col}' not found in the autoregressive dataset.")

# Filter for Draft Size == 0 (autoregressive)
df_autoreg_filtered = df_autoreg[df_autoreg['Draft Size'] == 0].copy()

# Check if df_autoreg_filtered is empty
if df_autoreg_filtered.empty:
    print("No autoregressive data found with Draft Size == 0.")
    sys.exit(1)

# Assign Technique label for autoregressive data
df_autoreg_filtered['Technique'] = 'Autoregressive'

# Ensure 'Technique' is categorical with the extended order
df_autoreg_filtered['Technique'] = pd.Categorical(df_autoreg_filtered['Technique'],
                                                 categories=technique_order,
                                                 ordered=True)

# Replicate autoregressive data for temperatures 0.5 and -1
if not df_autoreg_filtered.empty:
    # Extract values for temperature 0
    temp_zero = df_autoreg_filtered[df_autoreg_filtered['temperature'] == 0]
    if not temp_zero.empty:
        token_throughput_autoreg_0 = temp_zero['token_throughput_autoreg'].values[0]
        token_latency_autoreg_0 = temp_zero['token_latency_autoreg'].values[0]
        
        # Create replicated rows for temperatures 0.5 and -1
        replicated_autoreg = pd.DataFrame({
            'temperature': [0.5, -1],
            'token_throughput_autoreg': [token_throughput_autoreg_0, token_throughput_autoreg_0],
            'token_latency_autoreg': [token_latency_autoreg_0, token_latency_autoreg_0],
            'Technique': ['Autoregressive', 'Autoregressive']
        })
        
        # Combine with the original autoreg data
        df_autoreg_final = pd.concat([df_autoreg_filtered, replicated_autoreg], ignore_index=True)
    else:
        print("Error: No autoregressive speedup data found for temperature 0.")
        sys.exit(1)
else:
    print("Error: Autoregressive data is empty after filtering.")
    sys.exit(1)

# -----------------------------
# 3. Plotting
# -----------------------------

# Define the desired order for temperatures: 0, 0.5, -1
desired_temperatures = [0, 0.5, -1]

# Create a mapping from temperature to label
temperature_labels = {0: '0', 0.5: '0.5', -1: 'Random'}

# Define the color palette
colors = ['#D7E2F9', '#88BCFF', '#3864B9', '#1B345F', 'red']  # More distinguishable bluish palette

color_map = {
    'B': colors[0],
    'C': colors[1],
    'C+SV': colors[2],
    'C+SV+CA': colors[3],
    'Autoregressive': colors[4]
}

# Set global font size
plt.rcParams.update({'font.size': 12})  # Adjust font size slightly smaller for compact look

# Create subplots: 2 rows (Throughput and Latency) x 3 columns (Temperatures)
fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex='col', sharey='row')  # Adjust figsize for two-column paper
# Loop through each temperature and plot
for col, temp in enumerate(desired_temperatures):
    # Filter main data for the current temperature
    df_temp = df_merged[df_merged['temperature'] == temp]
    
    # Sort df_temp by 'Technique' to ensure consistent ordering in plots
    df_temp = df_temp.sort_values('Technique')
    
    # Check for multiple configurations per Technique and Temperature
    duplicates = df_temp.duplicated(subset=['Technique'], keep=False)
    if duplicates.any():
        print(f"Warning: Multiple configurations found for Temperature {temp} with the same Technique. Aggregating by mean.")
        # Aggregate by taking the mean to ensure unique Technique entries
        df_temp = df_temp.groupby('Technique').agg({
            'token_throughput': 'mean',
            'token_latency': 'mean',
            'speedup_throughput': 'mean',
            'speedup_latency': 'mean'
        }).reset_index()
    
    # Convert 'Technique' to string to ensure compatibility with matplotlib
    techniques = df_temp['Technique'].astype(str)
    
    # Assign colors based on Technique
    bar_colors = df_temp['Technique'].map(color_map)
    
    # -------------------
    # Plot Throughput Actual Values
    # -------------------
    ax_throughput = axes[0, col]
    bars_throughput = ax_throughput.bar(techniques, df_temp['token_throughput'], color=bar_colors, edgecolor='black', label='Throughput')
    if col == 0:
        ax_throughput.set_ylabel('Token Throughput (tokens/s)' , fontsize=14)

    ax_throughput.set_xticks([])  # Remove x-tick labels
    
    # Annotate speedup throughput values
    for idx, bar in enumerate(bars_throughput):
        speedup = df_temp['speedup_throughput'].iloc[idx]
        ax_throughput.annotate(f'{speedup:.2f}\u00D7',
                               xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                               xytext=(0, 1),  # 5 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=14, color='black')
        
    # y limit for throughput
    ax_throughput.set_ylim(1500, 3900)

    # Plot autoregressive speedup as a red dotted line (absolute value)
    autoreg_row = df_autoreg_final[df_autoreg_final['temperature'] == temp]
    if not autoreg_row.empty:
        autoreg_throughput = autoreg_row['token_throughput_autoreg'].values[0]
        ax_throughput.axhline(autoreg_throughput, color='red', linestyle='--', linewidth=2, label='Autoregressive')

    # Add temperature label above the current column of subplots
    ax_throughput.text(0.5, 1.06, f"Temp: {temperature_labels[temp]}", 
                       ha='center', va='center', transform=ax_throughput.transAxes, fontsize=16)
    
    # -------------------
    # Plot Latency Actual Values
    # -------------------
    ax_latency = axes[1, col]
    bars_latency = ax_latency.bar(techniques, df_temp['token_latency'], color=bar_colors, edgecolor='black', label='Latency')
    if col == 0:
        ax_latency.set_ylabel('Token Latency (s/token)', fontsize=14)
    ax_latency.set_xticks([])  # Remove x-tick labels

    # y limit for latency
    ax_latency.set_ylim(0, 1.7)
    
    # Annotate speedup latency values
    for idx, bar in enumerate(bars_latency):
        speedup = df_temp['speedup_latency'].iloc[idx]
        ax_latency.annotate(f'{speedup:.2f}\u00D7',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 1),  # 5 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=14, color='black')
    
    # Plot latency speedup as a red dotted line (absolute value)
    if not autoreg_row.empty:
        autoreg_latency = autoreg_row['token_latency_autoreg'].values[0]
        ax_latency.axhline(autoreg_latency, color='red', linestyle='--', linewidth=2, label='Autoregressive')

from matplotlib.lines import Line2D
from matplotlib.lines import Line2D

# Add legend outside the plot
# Create legend for bars (except Autoregressive)
handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[tech], edgecolor='black') for tech in technique_order if tech != 'Autoregressive']

# Add Autoregressive as a red line in the legend
handles.append(Line2D([0], [0], color='red', linestyle='--', linewidth=2))

# Updated labels showing the cumulativeness
labels = [
    'Spec 7',                           # B
    'C',                         # C
    'C + SV',             # C + SV
    'C + SV + CA',           # C + SV + CA
    'AR'                      # Autoregressive (red line)
]

# Place the legend
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=len(labels), frameon=False, fontsize=14)

# grid
for ax in axes.flat:
    ax.grid(axis='y', linestyle='--', linewidth=0.5)

# # Align y-axis labels for throughput and latency
# for ax in axes[:, 0]:  # Loop through the first column of axes (for both rows)
#     ax.yaxis.set_label_coords(-0.25, 0.5)  # Adjust the coordinates to align

axes[0, 0].yaxis.set_label_coords(-0.25, 0.6) 
axes[1, 0].yaxis.set_label_coords(-0.25, 0.48)

# Space between subplots
plt.subplots_adjust(hspace=0.05, wspace=0.05)

# Save the figure with high resolution suitable for publication
plt.savefig('speedup_comparison_by_temperature_with_autoregressive.pdf', bbox_inches='tight', format='pdf')

# Display the plot
plt.show()
