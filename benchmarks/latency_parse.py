import csv
import os
import statistics
import numpy as np
directory = "results/facebook"
models = ["opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b"]
batch_sizes = range(1, 257)

output_filepath = "output.csv"  # Specify the output file path

with open(output_filepath, "w") as output_file:  # Open the output file for writing
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(["model_name", "batch_size", "average_latency"])  # Write the header row

    for model in models:
        for batch_size in batch_sizes:
            filename = f"{model}_{batch_size}.csv"
            filepath = os.path.join(directory, filename)

            if os.path.isfile(filepath):
                with open(filepath, "r") as file:
                    csv_reader = csv.reader(file)
                    latencies = []

                    for row in csv_reader:
                        # Assuming the row format is ["dummy string", "batch_size", "latency"]
                        if len(row) == 3 and row[1].strip() == str(batch_size):
                            latency_ms = float(row[2])
                            latencies.append(latency_ms)

                    if latencies:
                        # Remove outliers using IQR method
                        q1 = np.percentile(latencies, 25)
                        q3 = np.percentile(latencies, 75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr

                        filtered_latencies = [latency for latency in latencies if lower_bound <= latency <= upper_bound]

                        if filtered_latencies:
                            average_latency = sum(filtered_latencies) / len(filtered_latencies)
                            csv_writer.writerow([model, batch_size, average_latency])  # Write the row to the output file

                    num_rows = 0
                    
                    for row in csv_reader:
                        # Assuming the row format is ["dummy string", "batch_size", "latency"]
                        if len(row) == 3 and row[1].strip() == str(batch_size):
                            latency_ms = float(row[2])
                            total_latency += latency_ms
                            num_rows += 1

                    if num_rows > 0:
                        average_latency = total_latency / num_rows
                        csv_writer.writerow([model, batch_size, average_latency])  # Write the row to the output file
                        
