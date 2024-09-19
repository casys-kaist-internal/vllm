import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV file
df = pd.read_csv('full_prefill.csv')

# Set the columns we want to plot
metrics_columns = ['p50_ttft', 'p99_ttft', 'p50_tpot', 'p99_tpot', 'p50_token_latency',
                   'p99_token_latency', 'token_throughput', 'request_throughput', 'token_latency']

# Function to sanitize model names


def sanitize_model_name(name):
    # Using the last part of the model name after the last slash
    return name.split('/')[-1]


# Group by the model pair, then iterate over each group
for (model_pair, group) in df.groupby(['target_model', 'draft_model']):
    # Sanitize model names
    sanitized_model_pair = (sanitize_model_name(
        model_pair[0]), sanitize_model_name(model_pair[1]))

    # Further group by dataset within each model pair
    dataset_group = group.groupby(['dataset'])

    # Iterate over each dataset
    for (dataset, dataset_group) in dataset_group:
        # Create a figure and axes for plotting
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
        axes = axes.flatten()

        # Group by temperature and request rate within each dataset
        temp_rate_group = dataset_group.groupby(
            ['temperature', 'request_rate'])

        # Plot each metric for each temperature and request rate combination
        for idx, column in enumerate(metrics_columns):
            for (key, subgroup) in temp_rate_group:
                # Calculate speedup relative to drop_threshold = 0
                base_value = subgroup[subgroup['drop_threshold']
                                      == 0][column].values[0]
                speedup = subgroup[column] / base_value

                # Plotting the line
                axes[idx].plot(subgroup['drop_threshold'], subgroup[column],
                               label=f'Temp={key[0]}, Rate={key[1]}')

                # Annotating the speedup
                for drop_threshold, spd in zip(subgroup['drop_threshold'], speedup):
                    if drop_threshold != 0:  # Avoid clutter at threshold 0
                        axes[idx].annotate(
                            f'{spd:.2f}x', (drop_threshold, subgroup[subgroup['drop_threshold'] == drop_threshold][column].values[0]))

            axes[idx].set_title(column)
            axes[idx].set_xlabel('Drop Threshold')
            axes[idx].set_ylabel('Value')
            axes[idx].legend(loc='upper right', fontsize='x-small')
            axes[idx].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the figure with a unique name for each dataset within each model pair
        filename = f'metrics_{sanitized_model_pair[0]}_{sanitized_model_pair[1]}_{dataset}.png'
        plt.savefig(filename)
        plt.close(fig)  # Close the figure to free up memory
