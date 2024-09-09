import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def process_and_plot(file_path, ax, title, plot_handles, plot_labels):
    # Initialize a dictionary to store seq_id and corresponding accept_probs
    accept_prob_dict = {}
    skip = 0
    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            # skip the first 100 lines
            if skip < 100:
                skip += 1
                continue

            # if there is "warning" in the line, skip it
            if "WARN" in line:
                continue

            # Split the line by commas
            parts = line.strip().split(',')
            # Remove the last element if it is empty
            if parts[-1] == '':
                parts = parts[:-1]

            if len(parts) < 2:
                # Skip lines with less than 2 parts
                continue

            # The first part is the seq_id (after 'result_accept')
            # Extract seq_id after 'result_accept'
            seq_id = parts[0].split()[1]

            # The rest are the accept_probs, convert them to floats
            probs = [float(prob) for prob in parts[1:]]

            if seq_id not in accept_prob_dict:
                accept_prob_dict[seq_id] = []

            # Add the seq_id and probs to the dictionary
            accept_prob_dict[seq_id].extend(probs)

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(accept_prob_dict.items()),
                      columns=['seq_id', 'accept_probs'])

    # Calculate the average accept_prob for each seq_id
    df['avg_accept_prob'] = df['accept_probs'].apply(np.mean)

    # Sort the dataframe based on the average accept_prob in descending order
    df = df.sort_values(by='avg_accept_prob', ascending=False)

    # Divide the data into 4 quartiles based on the sorted average accept_probs
    df['quartile'] = pd.qcut(df['avg_accept_prob'], 4, labels=False)

    # Define colors for each quartile
    # colors = ['#E15759', '#F28E2B', '#4E79A7', '#59A14F']

    colors = ['#FFFFFF', '#D7E2F9', '#88BCFF', '#3864B9', '#1B345F']
    colors.reverse()

    # Define the labels for the quartiles as percentages
    labels = ['Top 25%', '25%-50%', '50%-75%', 'Bottom 25%']
    # Reverse the order for plotting from bottom to top quartile
    labels = labels[::-1]

    # Plot the average accept_probs for each quartile in the subplot
    for i in range(3, -1, -1):
        quartile_data = df[df['quartile'] == i]

        # Determine the maximum length of sequences in this quartile
        max_len = max(quartile_data['accept_probs'].apply(len))

        # Initialize a list to accumulate sums of accept_probs
        avg_probs = np.zeros(max_len)
        counts = np.zeros(max_len)

        # Accumulate sums of accept_probs and counts at each index
        for probs in quartile_data['accept_probs']:
            for j, prob in enumerate(probs):
                avg_probs[j] += prob
                counts[j] += 1

        # Calculate the average at each index
        avg_accept_probs = avg_probs / counts

        # Smooth the curve by averaging over a window of size 5
        avg_accept_probs = np.convolve(
            avg_accept_probs, np.ones(5) / 5, mode='same')

        # Cut at index 512
        avg_accept_probs = avg_accept_probs[5:512]

        # Plot the average line for the quartile
        handle, = ax.plot(range(len(avg_accept_probs)), avg_accept_probs,
                          label=labels[i], color=colors[i])

        # Store handles and labels for the legend
        plot_handles.append(handle)
        plot_labels.append(labels[i])

    # Add labels and title to the subplot
    ax.set_xlabel('Output Position', fontsize=12)
    ax.set_title(f'Temperature {title}')


# Initialize the figure and subplots, sharing the y-axis
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=True)

# Store plot handles and labels for the combined legend
plot_handles = []
plot_labels = []

# Process and plot the data for all three files
process_and_plot('accept_prob_025.csv',
                 axes[0], '0.25', plot_handles, plot_labels)
process_and_plot('accept_prob_050.csv',
                 axes[1], '0.5', plot_handles, plot_labels)
process_and_plot('accept_prob_075.csv',
                 axes[2], '0.75', plot_handles, plot_labels)

# Set y-axis label on the left subplot
axes[0].set_ylabel('Mean Accept Prob', fontsize=12)

# Create a single legend for all subplots
fig.legend(plot_handles[:4], plot_labels[:4], fontsize=12,
           loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.01))

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.90])

# Show the plot
plt.show()

# Optionally, save the plot
plt.savefig('accept_probs_comparison.png')
