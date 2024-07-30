import pandas as pd
import matplotlib.pyplot as plt

# Reading the budget.csv file
file_path = 'budget.csv'  # Update the path to your CSV file
df = pd.read_csv(file_path, header=None)

# Dropping the last row
df = df[:-1]

# Renaming columns for easier access
df.columns = ['Type', 'CurrentSeqs', 'TotalTokens',
              'PrefillTokens', 'BaseTokens', 'SpecTokens']

# Dropping the 'Type' column as it is not needed for the plot
df = df.drop(columns=['Type'])

df = df[470:500]

# Change type to integer
df = df.astype('int')

# Calculating cumulative sums for the required lines
df['Prefill_Base_Tokens'] = df['PrefillTokens'] + df['BaseTokens']
df['Prefill_Base_Spec_Tokens'] = df['PrefillTokens'] + \
    df['BaseTokens'] + df['SpecTokens']

# Creating subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

# Plotting the filled areas on the first subplot
ax1.fill_between(df.index, 0, df['PrefillTokens'],
                 color='red', alpha=0.5, label='Prefill Tokens')
ax1.fill_between(df.index, df['PrefillTokens'], df['Prefill_Base_Tokens'],
                 color='green', alpha=0.5, label='Base Tokens')
ax1.fill_between(df.index, df['Prefill_Base_Tokens'],
                 df['Prefill_Base_Spec_Tokens'], color='purple', alpha=0.5, label='Spec Tokens')

# Plotting the lines on the first subplot
ax1.plot(df.index, df['PrefillTokens'],
         color='red', label='Prefill Tokens Line')
ax1.plot(df.index, df['Prefill_Base_Tokens'],
         color='green', label='Prefill + Base Tokens Line')
ax1.plot(df.index, df['Prefill_Base_Spec_Tokens'],
         color='purple', label='Prefill + Base + Spec Tokens Line')

# Show all x ticks
ax1.set_xticks(df.index)

# Setting labels and title for the first subplot
ax1.set_ylabel('Count')
ax1.set_title('Budget')
ax1.legend(loc='upper left')
ax1.grid(True)

# Plotting the current_seq line on the second subplot
ax2.plot(df.index, df['CurrentSeqs'], color='blue',
         linestyle='-', linewidth=1, label='Current Seq')

# # Annotating values at certain intervals
# interval = len(df) // 10  # Change this value to adjust the interval
# for i in range(0, len(df), interval):
#     ax2.annotate(f'{df["CurrentSeqs"].iloc[i]}', xy=(i, df["CurrentSeqs"].iloc[i]), xytext=(i, df["CurrentSeqs"].iloc[i] + 0.05 * max(df["CurrentSeqs"])),
#                  horizontalalignment='center')

# Setting labels and title for the second subplot
ax2.set_xlabel('Index')
ax2.set_ylabel('Current Seq')
ax2.legend(loc='upper left')
ax2.grid(True)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('budget_plot_with_current_seq.png')

# Showing the plot
plt.show()
