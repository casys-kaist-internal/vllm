import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of model configurations
target_draft_models = [['llama-2-7b-chat-hf', 'Llama-68M-Chat-v1'], 
                       ['bloom-7b1', 'bloomz-560m'],
                       ['opt-6.7b', 'opt-125m'], 
                       ['opt-6.7b', 'opt-350m'],
                       ['opt-13b', 'opt-125m'], 
                       ['opt-13b', 'opt-350m'], 
                       ['pythia-12b', 'pythia-14m'], 
                       ['pythia-12b', 'pythia-31m'], 
                       ['pythia-12b', 'pythia-70m'], 
                       ['pythia-12b', 'pythia-160m'], 
                       ['pythia-12b', 'pythia-410m'], 
                       ['pythia-6.9b', 'pythia-14m'], 
                       ['pythia-6.9b', 'pythia-31m'], 
                       ['pythia-6.9b', 'pythia-70m'], 
                       ['pythia-6.9b', 'pythia-160m'], 
                       ['pythia-6.9b', 'pythia-410m']]
datasets = ["apps", "alpaca", "sharegpt", "gsm8k"]

models_datasets = []
for target_draft_model in target_draft_models:
    for dataset in datasets:
        models_datasets.append((target_draft_model[0], target_draft_model[1], dataset))

# temperatures = ['0.0', '0.5', '1.0']
temperatures = ['0.5']

beta_dfs = {}
prompt_len_med = {}
# Loop through all combinations
for j, temperature in enumerate(temperatures):
    for i, (target_model, draft_model, dataset) in enumerate(models_datasets):
        # Construct the file name and load data
        name = f"{target_model}_{draft_model}_{dataset}_{temperature}"
        index = f"_{i}_{j}"
        print(name+index)
        try:
            beta_df = pd.read_csv(f"{name}/beta_list.csv", names=range(2048), dtype=str, na_values=[' None'])
            prompt_len_df = pd.read_csv(f"{name}/prompt_len.csv", names=['length'])
            prompt_len_median = int(prompt_len_df['length'].median().item())
            # Apply shift for each row based on prompt length
            for idx in range(len(beta_df)):
                shift_amount = prompt_len_df.at[idx, 'length']
                beta_df.iloc[idx] = beta_df.iloc[idx].shift(shift_amount, fill_value=np.nan)
        except:
            # data with just 0 
            beta_df = pd.DataFrame(np.zeros((10, 2048)))
            prompt_len_median = 0
        
        beta_dfs[name+index] = beta_df.astype(float)
        prompt_len_med[name+index] = prompt_len_median


# Create a figure for plotting
fig, axs = plt.subplots(nrows=len(target_draft_models), ncols=len(datasets), figsize=(8*len(datasets), 5*len(target_draft_models)))
n = 4 
# Loop through all combinations
for name, beta_df in beta_dfs.items():
    target_model = name.split('_')[0]
    draft_model = name.split('_')[1]
    dataset = name.split('_')[2]
    temperature = name.split('_')[3]
    # i = int(name.split('_')[4])
    # j = int(name.split('_')[5])
    # find the index from the target_draft_models list which has same target model and draft model  
    i = [idx for idx, val in enumerate(target_draft_models) if val[0] == target_model and val[1] == draft_model][0]
    j = datasets.index(dataset)

    row_averages = beta_df.mean(axis=1)
    sorted_indices = np.argsort(row_averages)
    n_rows = len(beta_df)
    edges = np.linspace(0, n_rows, n+1, dtype=int)  # Quartiles
    quartile_means = [beta_df.iloc[sorted_indices[edges[k]:edges[k+1]]].mean() for k in range(n)]

    for k, quartile_mean in enumerate(quartile_means):
        # Plot from index 100
        axs[i, j].plot(quartile_mean[prompt_len_med[name]:], label=f'Q{k+1}')
        # axs[i, j].plot(quartile_mean, label=f'Q{k+1}')

    axs[i, j].set_title(f'{target_model}, {draft_model}, {dataset}, Temp: {temperature}')
    axs[i, j].set_ylim(0, 1.1)
    # axs[i, j].legend()
    axs[i, j].set_xlabel('Inference Context Length')
    axs[i, j].set_ylabel('Average Beta Value')

# Main title 
# fig.suptitle('Average Beta Value for Different Models and Datasets for Temperature 0.5', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('beta_plot_0_5.png')
