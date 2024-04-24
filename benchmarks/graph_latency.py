import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Directory paths
directories = [
    "/home/sjchoi/workspace/vllm/benchmarks/output_latency_3090",
    "/home/sjchoi/workspace/vllm/benchmarks/output_latency_A100",
    "/home/sjchoi/workspace/vllm/benchmarks/output_latency_A6000"
]


def categorize_model(model_name):
    if 'opt' in model_name:
        return 'opt'
    elif 'pythia' in model_name:
        return 'pythia'
    elif 'bloom' in model_name:
        return 'bloom'
    elif 'llama' in model_name or 'Llama' in model_name:
        return 'llama'
    else:
        return 'other'

# Read CSV files from directories
dfs = []
for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            df.columns = ['dummy', 'batch_size', 'context_length', 'latency']
            model_name = filename.split('.csv')[0]
            gpu_name = directory.split('_')[-1]
            df['model_name'] = model_name
            df['gpu_name'] = gpu_name
            dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Filter by batch_size and context_length and calculate mean latency
latency_summary = []
for model_name in combined_df['model_name'].unique():
    for gpu_name in combined_df['gpu_name'].unique():
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            for context_length in [64, 128, 256, 512, 1024]:
                filtered = combined_df[(combined_df['model_name'] == model_name) &
                                       (combined_df['gpu_name'] == gpu_name) &
                                       (combined_df['batch_size'] == batch_size) &
                                       (combined_df['context_length'] >= context_length - 10) &
                                       (combined_df['context_length'] <= context_length + 10)]
                if not filtered.empty:
                    q1 = filtered['latency'].quantile(0.25)
                    q3 = filtered['latency'].quantile(0.75)
                    iqr = q3 - q1
                    filtered = filtered[(filtered['latency'] > (q1 - 1.5 * iqr)) & (filtered['latency'] < (q3 + 1.5 * iqr))]
                mean_latency = filtered['latency'].mean()
                latency_summary.append({
                    'model_name': model_name,
                    'gpu_name': gpu_name,
                    'batch_size': batch_size,
                    'context_length': context_length,
                    'mean_latency': mean_latency
                })

# save summary to csv
summary_df = pd.DataFrame(latency_summary)
summary_df.to_csv('latency_summary.csv')

target_draft_models = [
    ('llama-2-7b-chat-hf', 'Llama-68M-Chat-v1'), 
    ('bloom-7b1', 'bloomz-560m'),
    ('opt-13b', 'opt-350m'),
    ('opt-13b', 'opt-125m'),
    ('opt-6.7b', 'opt-350m'),
    ('opt-6.7b', 'opt-125m'),
    ('pythia-12b', 'pythia-410m'),
    ('pythia-12b', 'pythia-160m'),
    ('pythia-12b', 'pythia-70m'),
    ('pythia-12b', 'pythia-31m'),
    ('pythia-12b', 'pythia-14m'),
    ('pythia-6.9b', 'pythia-410m'),
    ('pythia-6.9b', 'pythia-160m'),
    ('pythia-6.9b', 'pythia-70m'),
    ('pythia-6.9b', 'pythia-31m'),
    ('pythia-6.9b', 'pythia-14m')
]
   
# Find the ratio of latency for target model and draft model for each batch size, context_length, gpu 
for target_model, draft_model in target_draft_models:
    for gpu_name in summary_df['gpu_name'].unique():
        for batch_size in summary_df['batch_size'].unique():
            for context_length in summary_df['context_length'].unique():
                target_latency = summary_df[(summary_df['model_name'] == target_model) & 
                                            (summary_df['gpu_name'] == gpu_name) &
                                            (summary_df['batch_size'] == batch_size) &
                                            (summary_df['context_length'] == context_length)]['mean_latency'].values[0]
                draft_latency = summary_df[(summary_df['model_name'] == draft_model) & 
                                            (summary_df['gpu_name'] == gpu_name) &
                                            (summary_df['batch_size'] == batch_size) &
                                            (summary_df['context_length'] == context_length)]['mean_latency'].values[0]
                ratio =  draft_latency / target_latency
                print(f'{target_model}, {draft_model}, {gpu_name}, {batch_size}, {context_length}, {ratio:.2f}')



# # Convert summary to DataFrame
# summary_df = pd.DataFrame(latency_summary)

# summary_df['model_group'] = summary_df['model_name'].apply(categorize_model)
# # Get unique GPUs, context lengths, and model groups
# unique_gpus = summary_df['gpu_name'].unique()
# unique_context_lengths = summary_df['context_length'].unique()
# unique_groups = summary_df['model_group'].unique()

# # Create subplots for each context length, GPU, and model group
# for model_group in unique_groups:
#     filtered_group = summary_df[summary_df['model_group'] == model_group]
#     fig, axs = plt.subplots(nrows=len(unique_context_lengths), ncols=len(unique_gpus), figsize=(30, 20), squeeze=False)

#     for row_idx, context_length in enumerate(unique_context_lengths):
#         for col_idx, gpu_name in enumerate(unique_gpus):
#             ax = axs[row_idx, col_idx]
#             subset = filtered_group[(filtered_group['context_length'] == context_length) & (filtered_group['gpu_name'] == gpu_name)]
            
#             # Plotting each model within the group as a separate bar
#             for model_name in subset['model_name'].unique():
#                 model_subset = subset[subset['model_name'] == model_name]
#                 ax.bar(model_subset['batch_size'].astype(str), model_subset['mean_latency'], label=model_name)

#             ax.set_title(f'{model_group} - GPU: {gpu_name}, Context Length: {context_length}')
#             ax.set_xlabel('Batch Size')
#             ax.set_ylabel('Mean Latency')
#             ax.set_xticks(model_subset['batch_size'].astype(str))  # Setting x-ticks explicitly
#             ax.legend()

#     plt.tight_layout()
#     plt.show()
#     plt.savefig(f'{model_group}_latency.png')