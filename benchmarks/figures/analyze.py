# Read output.txt in pandas 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

df = pd.read_csv('beta_list_output_draft_probs.txt', sep=' ', header=None)
df.columns = ['dummy', 'accept_cnt', 'previous_beta_list', 'draft_probs', 'accept_cnt_list', 'current_beta_list']

# remove beta equal to 0.5 and Nan
df = df.dropna()

print(df)

# accept_cnt is int 
df['accept_cnt'] = df['accept_cnt'].astype(int)

# Define the objective function
def objective(beta):
    result = (1 - beta**8) / (1 - beta**1)
    return result


df['objective'] = df['beta'].apply(objective)

# # Pearson correlation coefficient
# corr, _ = pearsonr(df['beta'], df['accept_cnt'])
# print(corr)

# # Spearman correlation coefficient
# corr, _ = spearmanr(df['beta'], df['accept_cnt'])
# print(corr)
# df['beta_bin'] = pd.cut(df['beta'], bins=20)
# # Group data by 'beta_bin' to get median of 'accept_cnt' and mean of 'beta'
# df_grouped = df.groupby('beta_bin').agg({'accept_cnt': 'median', 'beta': 'mean'}).reset_index()

# # Extract beta and accept_cnt_medians
# beta_vals = df_grouped['beta'].values
# accept_cnt_medians = df_grouped['accept_cnt'].values

# # Fit a polynomial of degree 3 (you can change the degree based on your data's behavior)
# p = np.polyfit(beta_vals, accept_cnt_medians, 3)

# # Generate a polynomial function from the coefficients
# polynomial_function = np.poly1d(p)
# print(polynomial_function)
# # Generate beta values for plotting the fitted curve
# beta_fit = np.linspace(min(beta_vals), max(beta_vals), 400)

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.scatter(beta_vals, accept_cnt_medians, label='Data (Median Accept Count)')
# plt.plot(beta_fit, polynomial_function(beta_fit), label='Fitted Polynomial Model', color='red')
# plt.xlabel('Beta')
# plt.ylabel('Acceptance Count')
# plt.title('Polynomial Fit to Median Acceptance Count')
# plt.legend()
# plt.show()
# plt.savefig('output.png')


# Divide beta in 10 bins and calculate the mean of accept_cnt
df['beta_bin'] = pd.cut(df['beta'], bins=20)

# Correct the column selection to use a list instead of a tuple
df_grouped = df.groupby('beta_bin').agg({'accept_cnt': ['mean', 'median', 'std'], 'beta': 'mean'})

# Reset index to make 'beta_bin' a column again for easier plotting
# Fix the column names after aggregation
df_grouped.columns = ['accept_cnt_mean','accept_cnt_median', 'accept_cnt_std', 'beta_mean']  # Simplify column names
df_grouped.reset_index(inplace=True)

# Calculate the objective function for the mean beta values
df_grouped['objective'] = df_grouped['beta_mean'].apply(objective)

# Convert 'beta_bin' to string for display purposes
df_grouped['beta_bin'] = df_grouped['beta_bin'].astype(str)

# Create a figure and a bar plot for 'accept_cnt'
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df_grouped['beta_bin'], df_grouped['accept_cnt_mean'], yerr=df_grouped['accept_cnt_std'], color='skyblue', capsize=5, label='Mean Accept Count')
ax.plot(df_grouped['beta_bin'], df_grouped['objective'], color='red', marker='o', linestyle='-', label='Objective Function')
ax.plot(df_grouped['beta_bin'], df_grouped['accept_cnt_median'], color='blue', marker='x', linestyle='-', label='Median Accept Count')

ax.set_xlabel('Beta Bins')
ax.set_ylabel('Average Acceptance Count', color='blue')
ax.tick_params(axis='y', labelcolor='blue')
ax.set_xticklabels(df_grouped['beta_bin'], rotation=45)

# legend
ax.legend()

# Title and layout
plt.title('Average Acceptance Count and Objective Function by Beta Bins')
fig.tight_layout()  # Adjust layout to make room for label rotation


# # Save the plot to a file
plt.savefig('combined_output.png')
# plt.show()  # Display the plot


# # Divide beta in 10 bins and calculate the mean of accept_cnt
# # df['beta_bin'] = pd.cut(df['beta'], bins=20)
# # df_grouped = df.groupby('beta_bin')['accept_cnt'].agg(['mean', 'median', 'std'])

# # # Reset index to make 'beta_bin' a column again for easier plotting
# # df_grouped = df_grouped.reset_index()

# # # Convert 'beta_bin' to string to display on x-axis properly
# # df_grouped['beta_bin'] = df_grouped['beta_bin'].astype(str)

# # # Create a bar plot for mean values
# # plt.figure(figsize=(10, 6))
# # plt.bar(df_grouped['beta_bin'], df_grouped['mean'], yerr=df_grouped['std'], color='skyblue', capsize=5, label='Mean')
# # # plot median
# # plt.plot(df_grouped['beta_bin'], df_grouped['median'], color='red', marker='o', label='Median')
# # plt.xlabel('Beta Bins')
# # plt.ylabel('Average Acceptance Count')
# # plt.title('Average Acceptance Count by Beta Bins with Std Deviation')
# # plt.xticks(rotation=45)  # Rotate labels to improve readability
# # plt.tight_layout()  # Adjust layout to make room for label rotation

# # # Add a legend
# # plt.legend()

# # # Save the plot to a file
# # plt.savefig('output_with_std.png')
# # plt.show()  # Display the plot

# # # df_grouped = df.groupby('beta_bin').mean()


# # # #save the plot
# # plt.savefig('output.png')