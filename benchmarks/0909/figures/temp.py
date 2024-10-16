import pandas as pd

# Read the CSV file
df = pd.read_csv('baseline_10_06_NVIDIARTXA6000.csv')

# Filter rows where dataset is 'shareGPT'
filtered_df = df[df['Dataset'].str.lower() == 'sharegpt']

# Create a dictionary that maps the original columns to the new ones
column_mapping = {
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
}

# remove the 'Result' column
filtered_df = filtered_df.drop(columns=['Result'])

# Assign default values for the new columns that are not in the original dataset
filtered_df['prefill_schedule_mode'] = "full_prefill"  # or appropriate default
filtered_df['budget_token'] = 4095           # or appropriate default
filtered_df['budget_seq'] = 256             # or appropriate default
filtered_df['colocate'] = False               # or appropriate default
filtered_df['consolidated_attention'] = False # or appropriate default
filtered_df['drop_threshold'] = 0         # or appropriate default

# Rename the columns using the mapping
filtered_df = filtered_df.rename(columns=column_mapping)

# Save the updated DataFrame to a new CSV file
filtered_df.to_csv('filtered_sharegpt.csv', index=False)