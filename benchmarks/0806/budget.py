import pandas as pd
import matplotlib.pyplot as plt


def process_and_plot(files):
    """
    Processes a list of text files to parse the data and plots the relationship
    between num_batched_tokens and elapsed_time for comparison.

    Parameters:
    files (list of str): List of file paths to the text files.

    Returns:
    None
    """
    data = []
    file_labels = []

    # Reading and parsing the content of each file
    for file_index, file_path in enumerate(files):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for i in range(0, len(lines), 2):
            budget_line = lines[i].split(',')
            elapsed_time_line = lines[i+1].split(',')

            budget = {
                'file': file_index,
                'token_budget': int(budget_line[1].strip()),
                'max_num_seqs': int(budget_line[2].strip()),
                'num_batched_tokens': int(budget_line[3].strip()),
                'num_curr_seqs': int(budget_line[4].strip()),
                'prefill_tokens': int(budget_line[5].strip()),
                'base_tokens': int(budget_line[6].strip()),
                'spec_tokens': int(budget_line[7].strip()),
                'elapsed_time': float(elapsed_time_line[2].strip())
            }
            data.append(budget)
        file_labels.append(file_path)

    # Creating the DataFrame
    df = pd.DataFrame(data)

    # Change elapsed time to milliseconds
    df['elapsed_time'] = df['elapsed_time'] * 1000

    # Remove rows where num_batched_tokens is 0
    df = df[df['num_batched_tokens'] != 0]

    # Plotting the relationship between num_batched_tokens and elapsed_time
    def plot_num_batched_tokens_vs_elapsed_time(df):
        plt.figure(figsize=(10, 6))
        for file_index, label in enumerate(file_labels):
            subset = df[df['file'] == file_index]
            plt.scatter(subset['num_batched_tokens'],
                        subset['elapsed_time'], alpha=0.7, label=label)
        plt.title('Relationship between Num Batched Tokens and Elapsed Time')
        plt.xlabel('Num Batched Tokens')
        plt.ylabel('Elapsed Time (ms)')
        plt.grid(True)
        plt.legend()
        plt.savefig('num_batched_tokens_vs_elapsed_time_comparison.png')

    # Plotting the relationship with token portions highlighted
    def plot_num_batched_tokens_vs_elapsed_time_with_tokens(df):
        plt.figure(figsize=(14, 8))
        for file_index, label in enumerate(file_labels):
            subset = df[df['file'] == file_index]
            subset['spec_token_portion'] = subset['spec_tokens'] / \
                (subset['spec_tokens'] +
                 subset['base_tokens'] + subset['prefill_tokens'])
            subset['base_token_portion'] = subset['base_tokens'] / \
                (subset['spec_tokens'] +
                 subset['base_tokens'] + subset['prefill_tokens'])
            scatter = plt.scatter(subset['num_batched_tokens'], subset['elapsed_time'],
                                  c=subset['spec_token_portion'], s=subset['base_token_portion']*200,
                                  alpha=0.7, cmap='viridis_r', label=label)
        plt.colorbar(scatter, label='Spec Token Portion')
        plt.title(
            'Relationship between Num Batched Tokens and Elapsed Time with Token Portions')
        plt.xlabel('Num Batched Tokens')
        plt.ylabel('Elapsed Time (ms)')
        plt.grid(True)
        plt.legend()
        plt.savefig(
            'num_batched_tokens_vs_elapsed_time_with_tokens_comparison.png')

    plot_num_batched_tokens_vs_elapsed_time(df)
    # plot_num_batched_tokens_vs_elapsed_time_with_tokens(df)


# Example usage
# Replace with your actual file names
file_list = ["result_cp_c_d4.txt", "result_cp_c_d7.txt"]
process_and_plot(file_list)
