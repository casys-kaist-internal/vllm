import pandas as pd
import os
import matplotlib.pyplot as plt

# Function to process each CSV file and extract relevant data
colors = ['#3864B9', '#D7E2F9', '#1B345F', '#88BCFF']


def process_csv(file_name):
    data = []
    skip = 0

    with open(file_name, 'r') as f:
        lines = f.readlines()

        # Skip the first two lines
        lines = lines[2:]

        for i in range(0, len(lines) - 1, 2):
            # Extract the budget line and the step_time line
            budget_line = lines[i].strip()
            step_time_line = lines[i + 1].strip()

            # Parse the budget line
            budget_values = budget_line.split(',')
            total_tokens = int(budget_values[2].strip())
            # Skip if total_tokens is 0
            if total_tokens == 0:
                continue

            # Skip first 3 data
            if skip < 3:
                skip += 1
                continue

            # Parse the step_time line
            step_time = float(step_time_line.split(',')[1].strip())

            # Append to data list
            data.append([total_tokens, step_time])

    # Create a DataFrame from the data
    columns = ['Tokens', 'Time']
    df = pd.DataFrame(data, columns=columns)
    return df


# Initialize an empty DataFrame to store all data
final_df = pd.DataFrame()

# Loop through batch_size CSV files
for batch_size in range(1, 128):
    file_name = f'batch_size_{batch_size}.csv'

    if os.path.exists(file_name):
        # Process the file and append the data to final_df
        df = process_csv(file_name)
        final_df = pd.concat([final_df, df], ignore_index=True)

# Save csv file
final_df.to_csv('final_data.csv', index=False)

# Group the data by 'Tokens' and calculate the mean of 'Time'
grouped_df = final_df.groupby('Tokens')['Time'].mean().reset_index()
grouped_df.columns = ['Total Tokens', 'Mean Step Time']

# Set up the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 2.8), sharex=True)

reduced_grouped_df = grouped_df.iloc[1::2, :]


# Plotting the original data (blue line)
ax1.plot(reduced_grouped_df['Total Tokens'],
         reduced_grouped_df['Mean Step Time'], marker='o', label='Single Batch',
         markersize=4, color=colors[2], linewidth=1,
         markeredgecolor='black', markeredgewidth=0.5)

# Adding vertical lines at every 64 tokens
max_tokens = grouped_df['Total Tokens'].max()
for x in range(64, max_tokens + 64, 64):
    ax1.axvline(x=x, color='gray', linestyle='--', linewidth=0.7)
for x in range(64, max_tokens + 64, 64):
    ax2.axvline(x=x, color='gray', linestyle='--', linewidth=0.7)

# Creating the new function: y value at x/2 doubled (green line)
new_y_values = []
slowdown_ratios = []

for x in grouped_df['Total Tokens']:
    half_x = x // 2
    if half_x in grouped_df['Total Tokens'].values:
        y_value = grouped_df[grouped_df['Total Tokens']
                             == half_x]['Mean Step Time'].values[0] * 2
    else:
        y_value = None  # Handle cases where half_x doesn't exist in the data
    new_y_values.append(y_value)

    # Calculate the slowdown ratio
    if y_value is not None:
        original_y = grouped_df[grouped_df['Total Tokens']
                                == x]['Mean Step Time'].values[0]
        slowdown_ratio = y_value / original_y if original_y != 0 else 0
        slowdown_ratios.append(slowdown_ratio)
    else:
        slowdown_ratios.append(None)

# Plotting the new function (green line)
ax1.plot(grouped_df['Total Tokens'], new_y_values, marker='D', markersize=4,
         linestyle='-', color=colors[3], label='Two Sub-Batches', linewidth=1,
         markeredgecolor='black', markeredgewidth=0.5)

# Set labels and grid for the first subplot
# ax1.set_title('Mean Step Time vs Total Tokens')
ax1.set_ylabel('Latency (s)')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=2, frameon=False)
# ax1.grid(True)
ax1.set_yticks([0.0, 0.1, 0.2])
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# start y axis from 0
ax1.set_ylim(ymin=0)
ax1.set_xlim(xmin=0, xmax=1024)

# Plot the slowdown ratios on the second subplot
ax2.plot(grouped_df['Total Tokens'], slowdown_ratios, marker='o', markersize=4,
         linestyle='-', color=colors[0], label='Slowdown Ratio', linewidth=0,
         markeredgecolor='black', markeredgewidth=0.5)

# Set the second y-axis label
ax2.set_xlabel('Total Tokens')
ax2.set_ylabel('Slowdown Ratio', fontsize=10)
# ax2.legend(loc='upper right')
# ax2.grid(True)
ax2.set_xticks(range(0, max_tokens + 64, 128))
ax2.set_ylim(ymax=2)

# Set the y-axis to be more fine-grained with gridlines
# Adjust the range and step as needed
# y tick from 1.0 to 2.0
ax2.set_yticks([1.0 + i * 0.4 for i in range(3)])
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)


# Adjust layout for better fit
# Adjust layout for better fit
# Adjust rect to make room for the legend at the top
plt.tight_layout(rect=[0, 0, 1, 0.9])

# Adjust the spacing between subplots and the top of the figure to fit the legend
fig.subplots_adjust(top=0.85)
# Save the plot as an image file
plt.savefig('tokens_latency_with_slowdown.pdf', format='pdf')
plt.show()
