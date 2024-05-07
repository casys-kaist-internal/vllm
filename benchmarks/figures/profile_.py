import pandas as pd 
import numpy as np

# Read content of file in directory 
# csv_file = "5_2_result_1/fig1_sps_gsm8k.csv"
csv_file = "5_2_result_2/fig1_sps_gsm8k.csv"

header = ['temp', 'target_model', 'draft_model', 'batch_size', 'draft_size', 'target_attention', 'dummy', 'total_throughput', 'output_throughput']
df = pd.read_csv(csv_file, names=header)
temp_0_5 = df[df['temp'] == 0.5]
temp_0_75 = df[df['temp'] == 0.75]

batch_sizes = temp_0_5['batch_size'].unique()
temp_0_5_dfs = []
for batch_size in temp_0_5['batch_size'].unique():
    temp_0_5_dfs.append(temp_0_5[temp_0_5['batch_size'] == batch_size])

# Split the datas into each 9 rows
for i in range(len(temp_0_5_dfs)):
    temp_0_5_dfs[i] = np.array_split(temp_0_5_dfs[i], len(temp_0_5_dfs[i]) / 9)

for i in range(len(temp_0_5_dfs)):
    for j in range(len(temp_0_5_dfs[i])):
        temp_0_5_dfs[i][j] = temp_0_5_dfs[i][j].reset_index(drop=True)
        df = temp_0_5_dfs[i][j]

        # Calculate the speedup of the output throughput 
        # the baseline is when draft_size is 0

        baseline = df.iloc[1]['output_throughput']
        df['speedup'] =  df['output_throughput'] / baseline

# Calculate the average speedup of the output throughput
draft_size_unique = temp_0_5['draft_size'].unique()
speedup_list = []
for i in range(len(temp_0_5_dfs)):
    speedup = {}
    df_list = temp_0_5_dfs[i]
    for j in range(len(df_list)):
        df = df_list[j]
        # for unique draft_size
        for draft_size in df['draft_size'].unique():
            if draft_size not in speedup:
                speedup[draft_size] = []
            speedup[draft_size].append(df[df['draft_size'] == draft_size]['speedup'].mean())
    speedup_list.append(speedup)

# for i in range(len(speedup_list)):
#     print("--------------------")
#     print(f"Batch size {batch_sizes[i]}")
#     for key, value in speedup_list[i].items():
#         # average of value 
#         print(f"Draft size {key}: {sum(value) / len(value)}")

for i in range(len(speedup_list)):
    for key, value in speedup_list[i].items():
        print(f"0.5, {batch_sizes[i]}, {key}, {sum(value) / len(value)}")


batch_sizes = temp_0_75['batch_size'].unique()
temp_0_75_dfs = []
for batch_size in temp_0_75['batch_size'].unique():
    temp_0_75_dfs.append(temp_0_75[temp_0_75['batch_size'] == batch_size])

# Split the datas into each 9 rows
for i in range(len(temp_0_75_dfs)):
    temp_0_75_dfs[i] = np.array_split(temp_0_75_dfs[i], len(temp_0_75_dfs[i]) / 9 )

for i in range(len(temp_0_75_dfs)):
    for j in range(len(temp_0_75_dfs[i])):
        temp_0_75_dfs[i][j] = temp_0_75_dfs[i][j].reset_index(drop=True)
        df = temp_0_75_dfs[i][j]

        # Calculate the speedup of the output throughput 
        # the baseline is when draft_size is 0
        baseline = df.iloc[1]['output_throughput']
        df['speedup'] =  df['output_throughput'] / baseline


# Calculate the average speedup of the output throughput
draft_size_unique = temp_0_75['draft_size'].unique()
speedup_list = []
for i in range(len(temp_0_75_dfs)):
    speedup = {}
    df_list = temp_0_75_dfs[i]
    for j in range(len(df_list)):
        df = df_list[j]
        # for unique draft_size
        for draft_size in df['draft_size'].unique():
            if draft_size not in speedup:
                speedup[draft_size] = []
            speedup[draft_size].append(df[df['draft_size'] == draft_size]['speedup'].mean())
    speedup_list.append(speedup)

# for i in range(len(speedup_list)):
#     print("--------------------")
#     print(f"Batch size {batch_sizes[i]}")
#     for key, value in speedup_list[i].items():
#         # average of value 
#         print(f"Draft size {key}: {sum(value) / len(value)}")

for i in range(len(speedup_list)):
    for key, value in speedup_list[i].items():
        print(f"0.75, {batch_sizes[i]}, {key}, {sum(value) / len(value)}")
