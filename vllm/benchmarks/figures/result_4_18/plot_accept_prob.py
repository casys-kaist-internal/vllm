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

temperatures = ['0.0', '0.5', '1.0']

prob_dfs = {}
prompt_len_med = {}
# Loop through all combinations
for j, temperature in enumerate(temperatures):
    for i, (target_model, draft_model, dataset) in enumerate(models_datasets):
        # Construct the file name and load data
        name = f"{target_model}_{draft_model}_{dataset}_{temperature}"
        index = f"_{i}_{j}"
        print(name+index)
        try:
            prob_df = pd.read_csv(f"{name}/accept_probs.csv", names=range(2048), dtype=str, na_values=[' None'])
            prompt_len_df = pd.read_csv(f"{name}/prompt_len.csv", names=['length'])
            prompt_len_median = int(prompt_len_df['length'].median().item())
            # Apply shift for each row based on prompt length
            for idx in range(len(prob_df)):
                shift_amount = prompt_len_df.at[idx, 'length']
                prob_df.iloc[idx] = prob_df.iloc[idx].shift(shift_amount, fill_value=np.nan)
        except:
            # data with just 0 
            prob_df = pd.DataFrame(np.zeros((10, 2048)))
            prompt_len_median = 0
        
        prob_dfs[name+index] = prob_df.astype(float)
        # change value greater than 1 to 1
        prob_dfs[name+index] = prob_dfs[name+index].clip(0, 1)
        prompt_len_med[name+index] = prompt_len_median


# Create a figure for plotting
fig, axs = plt.subplots(nrows=len(models_datasets), ncols=len(temperatures), figsize=(8*len(temperatures), 5*len(models_datasets)))
n = 4 
# Loop through all combinations
for name, beta_df in prob_dfs.items():
    target_model = name.split('_')[0]
    draft_model = name.split('_')[1]
    dataset = name.split('_')[2]
    temperature = name.split('_')[3]
    i = int(name.split('_')[4])
    j = int(name.split('_')[5])

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
    axs[i, j].set_ylabel('Accept Prob (P/Q) Value')

plt.legend()
plt.tight_layout()
plt.savefig('probs_plot.png')
