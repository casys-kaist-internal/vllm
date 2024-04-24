import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of model configurations
models_datasets = [
    ('bloom-7b1', 'bloomz-560m', 'gsm8k'),
    ('bloom-7b1', 'bloomz-560m', 'humaneval'),
    ('bloom-7b1', 'bloomz-560m', 'mt-bench'),
    ('bloom-7b1', 'bloomz-560m', 'alpaca'),
    # ('llama-2-7b-chat-hf', 'Llama-68M-Chat-v1', 'gsm8k'),
    # ('llama-2-7b-chat-hf', 'Llama-68M-Chat-v1', 'humaneval'),
    # ('llama-2-7b-chat-hf', 'Llama-68M-Chat-v1', 'mt-bench'),
    # ('llama-2-7b-chat-hf', 'Llama-68M-Chat-v1', 'alpaca'),
    # ('opt-6.7b', 'opt-125m', 'gsm8k'),
    # ('opt-6.7b', 'opt-125m', 'humaneval'),
    # ('opt-6.7b', 'opt-125m', 'mt-bench'),
    # ('opt-13b', 'opt-125m', 'gsm8k'),
    # ('opt-13b', 'opt-125m', 'humaneval'),
    # ('opt-13b', 'opt-125m', 'mt-bench'),
    # ('opt-13b', 'opt-125m', 'alpaca'),
    # ('opt-13b', 'opt-125m', 'sharegpt'),
    # ('opt-13b', 'opt-350m', 'gsm8k'),
    # ('opt-13b', 'opt-350m', 'humaneval'),
    # ('opt-13b', 'opt-350m', 'mt-bench'),
    # ('opt-13b', 'opt-350m', 'sharegpt'),
    # ('pythia-12b', 'pythia-70m', 'gsm8k'),
    # ('pythia-12b', 'pythia-70m', 'humaneval'),
    # ('pythia-12b', 'pythia-70m', 'mt-bench'),
    # ('pythia-12b', 'pythia-160m', 'gsm8k'),
    # ('pythia-12b', 'pythia-160m', 'humaneval'),
    # ('pythia-12b', 'pythia-160m', 'mt-bench'),
    # ('pythia-12b', 'pythia-410m', 'gsm8k'),
    # ('pythia-12b', 'pythia-410m', 'humaneval'),
    # ('pythia-12b', 'pythia-410m', 'mt-bench')
]

temperatures = ['0.5']
beta_dfs = {}
# Loop through all combinations
for i, (target_model, draft_model, dataset) in enumerate(models_datasets):
    for j, temperature in enumerate(temperatures):
        # Construct the file name and load data
        name = f"{target_model}_{draft_model}_{dataset}_{temperature}"
        index = f"_{i}_{j}"
        print(name+index)
        beta_dfs[name+index] = pd.read_csv(f"{name}/beta_list.csv", names=range(2048))


# Create a figure for plotting
fig, axs = plt.subplots(nrows=1, ncols=len(models_datasets), figsize=(40, 15))
n = 4 
# Loop through all combinations
for name, beta_df in beta_dfs.items():
  print(name)
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
      axs[i].plot(quartile_mean, label=f'Q{k+1}')

  axs[i].set_title(f'{target_model}, {draft_model}, {dataset}, Temp: {temperature}')
  axs[i].set_ylim(0, 1.1)
  axs[i].legend()

plt.savefig('beta_plot_.png')
