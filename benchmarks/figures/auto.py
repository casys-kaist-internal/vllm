import os
import csv

# Base directory and output CSV file path
base = "/home/sjlim/workspace/vllm/benchmarks/figures"
output_csv_path = os.path.join(base, "averages_summary.csv")

# Datasets, batch sizes, and draft sizes setup
datasets = ["gsm8k", "humaneval", "alpaca", "mt-bench", "sharegpt"]
batch_sizes = [2**i for i in range(0, 8)]  # 1 to 128
draft_sizes = range(2, 9)  # 2 to 8

# Open the output CSV file for writing
with open(output_csv_path, 'w', newline='') as csvfile:
    # Define the CSV writer and write the header
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['dataset', 'batch_size', 'draft_size', 'alpha', 'beta', 'gamma', 'accept_cnt', 'accept_rate', 'expected_generated_tokens'])

    # Iterate over combinations, load the files, and calculate the averages
    for dataset in datasets:
        for batch_size in batch_sizes:
            for draft_size in draft_sizes:
                file_path = f"{base}/alpha/alpha_sps_{dataset}_{draft_size}_{batch_size}.csv"
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        lines = file.readlines()
                        
                        # Temporary storage for the current file's values
                        file_values = {"gamma": [], "alpha": [], "beta": [], "accept_cnt": []}
                        
                        for line in lines:
                            if line.startswith("result: "):
                                parts = line.split(" ", 2)
                                key = parts[1]
                                numbers = eval(parts[2].strip())
                                
                                # Append the numbers to the appropriate list for the current file
                                if key in file_values:
                                    file_values[key].extend(numbers)
                        
                        # Calculate the averages for the current file
                        file_averages = {key: sum(val) / len(val) for key, val in file_values.items() if val}
                        
                        # Calculate accept_rate as accept_cnt / gamma
                        gamma_average = file_averages.get('gamma', 0)
                        accept_cnt_average = file_averages.get('accept_cnt', 0)
                        accept_rate = accept_cnt_average / gamma_average if gamma_average else None
                        
                        # Calculate expected_generated_tokens
                        alpha = file_averages.get('alpha')
                        if alpha is not None and (1 - alpha) != 0:
                            print(1-alpha)
                            expected_generated_tokens = (1 - alpha ** (gamma_average + 1)) / (1 - alpha)
                        else:
                            expected_generated_tokens = None
                        
                        # Write the data row to the CSV file
                        csvwriter.writerow([
                            dataset,
                            batch_size,
                            draft_size,
                            file_averages.get('alpha'),
                            file_averages.get('beta'),
                            gamma_average,
                            accept_cnt_average,
                            accept_rate,
                            expected_generated_tokens
                        ])
