# Read output.txt in pandas 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from ast import literal_eval
import math

# def safe_literal_eval(s):
#     try:
#         # Attempt to evaluate the string as a Python literal
#         if s == '[]':
#             return []
#         return literal_eval(s)
#     except ValueError:
#         # Return the original string if literal_eval fails
#         return s

# # read only the first 100 lines
# df = pd.read_csv('beta_list_output_draft_probs.txt', sep='|', header=None)
# df.columns = ['dummy', 'accept_cnt', 'previous_beta_list', 'draft_probs', 'accept_cnt_list', 'accept_probs', 'current_beta_list']

# df['accept_cnt'] = df['accept_cnt'].astype(int)
# df['previous_beta_list'] = df['previous_beta_list'].apply(lambda x: safe_literal_eval(x.strip()))
# df['draft_probs'] = df['draft_probs'].apply(lambda x: safe_literal_eval(x.strip()))
# df['accept_cnt_list'] = df['accept_cnt_list'].apply(lambda x: safe_literal_eval(x.strip()))
# df['accept_probs'] = df['accept_probs'].apply(lambda x: safe_literal_eval(x.strip()))
# df['current_beta_list'] = df['current_beta_list'].apply(lambda x: safe_literal_eval(x.strip()))

# df = df.dropna()
# df = df.drop(columns=['dummy'])

# # if len of previous_beta_list is 0, drop the row
# df = df[df['previous_beta_list'].map(len) > 0]

# # Define the objective function
# def objective(beta):
#     result = (1 - beta**8) / (1 - beta**1)
#     return result


# # calculate beta_ema from previous_beta_list
# def beta_ema(beta_list, factor):
#     beta_ema = beta_list[0]

#     for i in range(1, len(beta_list)):
#         beta_ema = beta_ema * (1 - factor) + beta_list[i] * factor

#     return beta_ema

# # apply beta_ema to previous_beta_list
# df['beta_ema'] = df['previous_beta_list'].apply(lambda x: beta_ema(x, 0.5))

# print(df)

# # Save dataframe as object for later use
# df.to_pickle('beta_list_output_draft_probs.pkl')

# Define the objective function
def objective(beta):
    result = (1 - beta**8) / (1 - beta**1)
    return result

# Define the objective function
def objective_round(beta):
    a = 8
    b = 1
    result = (1 - beta**a) / (1 - beta**b)

    return round(result - 1)

# Define the objective function
# def polynomial_fit(beta):
#     a = 25.48
#     b = -31.4
#     c = 12.35
#     d = -0.2614
#     result = a * beta**3 + b * beta**2 + c * beta + d
#     return round(result)

# 58.97 x - 96.03 x + 49.74 x - 6.746 x + 0.8232 
def polynomial_fit(beta):
    a = 58.97
    b = -96.03
    c = 49.74
    d = -6.746
    e = 0.8232
    result = a * beta**4 + b * beta**3 + c * beta**2 + d * beta + e
    return round(result)

# read pickle file
df = pd.read_pickle('beta_list_output_draft_probs.pkl')

print(df)


#################################################################################################################################
# 1. Plot the distribution of accept_cnt 
#################################################################################################################################
# # Determine the range for the bins
# max_value = df['accept_cnt'].max()
# bins = np.arange(-0.5, max_value + 1.5)  # Extending to max_value + 1 and starting from -0.5

# # Plotting the distribution of accept_cnt
# plt.figure()
# plt.hist(df['accept_cnt'], bins=bins, edgecolor='black', align='mid')  # Aligns the bins centered on integers
# plt.xlabel('accept_cnt')
# plt.ylabel('Frequency')
# plt.title('Distribution of accept_cnt')
# plt.xticks(range(max_value + 1))  # Setting x-axis ticks to cover all integer values from 0 to max_value
# plt.savefig('accept_cnt_distribution.png')
# plt.show()  # Display the plot

#################################################################################################################################
# 2. Plot the distribution of beta_ema
#################################################################################################################################

# # the beta values are between 0 and 1
# bins = np.linspace(0, 1, 21)  # 20 bins between 0 and 1

# # Plotting the distribution of beta_ema
# plt.figure()
# plt.hist(df['beta_ema'], bins=bins, edgecolor='black', align='mid')  # Aligns the bins centered on integers
# plt.xlabel('beta_ema')
# plt.ylabel('Frequency')
# plt.title('Distribution of beta_ema')
# plt.savefig('beta_ema_distribution.png')

#################################################################################################################################
# 3. Divide beta in 20 bins and calculate the mean of accept_cnt
#################################################################################################################################

# Define the bin edges
# bin_edges = np.linspace(0, 1, 21)  # 20 bins between 0 and 1

# # Assign bin numbers
# df['beta_bin'] = pd.cut(df['beta_ema'], bins=bin_edges, labels=False, include_lowest=True)

# # Group by 'beta_bin' and calculate statistics
# df_grouped = df.groupby('beta_bin').agg({'accept_cnt': ['mean', 'median', 'std'], 'beta_ema': 'mean'})
# df_grouped.columns = ['accept_cnt_mean', 'accept_cnt_median', 'accept_cnt_std', 'beta_mean']  # Simplify column names
# df_grouped.reset_index(inplace=True)

# # Calculate average standard deviation
# average_std = df_grouped['accept_cnt_std'].mean()

# # Generate bin labels for x-axis
# bin_labels = [f"{edge:.2f} - {bin_edges[i+1]:.2f}" for i, edge in enumerate(bin_edges[:-1])]
# df_grouped['beta_bin_labels'] = bin_labels

# # Calculate the objective function value for each bin
# df_grouped['objective'] = df_grouped['beta_mean'].apply(objective)
# df_grouped['objective_round'] = df_grouped['beta_mean'].apply(objective_round)
# df_grouped['polynomial_fit'] = df_grouped['beta_mean'].apply(polynomial_fit)

# # # Fit polynomial to median acceptance counts
# beta_vals = df_grouped['beta_mean'].values
# accept_cnt_medians = df_grouped['accept_cnt_median'].values
# p = np.polyfit(beta_vals, accept_cnt_medians, 4)
# polynomial_function = np.poly1d(p)

# # Print the polynomial function equation
# print("Polynomial Function Equation:")
# print(polynomial_function)

# # Plotting
# fig, ax = plt.subplots(figsize=(20, 12))
# bar_plot = ax.bar(df_grouped['beta_bin_labels'], df_grouped['accept_cnt_mean'], yerr=df_grouped['accept_cnt_std'], color='skyblue', capsize=5)
# line_plot = ax.plot(df_grouped['beta_bin_labels'], df_grouped['accept_cnt_median'], color='blue', marker='x', linestyle='-')[0]
# objective_plot = ax.plot(df_grouped['beta_bin_labels'], df_grouped['objective'], color='red', marker='o', linestyle='-')[0]
# # objective_round_plot = ax.plot(df_grouped['beta_bin_labels'], df_grouped['objective_round'], color='black', marker='o', linestyle='-')[0]
# polynomial_fit_plot = ax.plot(df_grouped['beta_bin_labels'], df_grouped['polynomial_fit'], color='black', linestyle='-')[0]

# ax.set_xlabel('Beta Bins')
# ax.set_ylabel('Average Acceptance Count')
# ax.set_xticks(range(len(bin_labels)))  # Set x-tick positions
# ax.set_xticklabels(bin_labels, rotation=45)  # Set x-tick labels and rotate them

# # Properly setting the legend by providing handles and labels
# ax.legend([bar_plot, line_plot, objective_plot, polynomial_fit_plot], [f'Mean Accept Count (Avg Std-Dev: {average_std:.2f})', 'Median Accept Count', '(1-beta^8)/(1-beta)', '25.48*beta^3 - 31.4*beta^2 + 12.35*beta - 0.2614'], loc='upper left')

# plt.savefig('accept_cnt_vs_beta.png')  # Save the plot
# plt.show()  # Display the plot

#################################################################################################################################
# 4. Relationship between draft_prob and whether it is accepted or rejected 
#################################################################################################################################


# if the index is smaller than accept_cnt, then it is accepted
# if the index is exactly same to accept_cnt, then it is rejected
# if the index is larger than accept_cnt, then it is not considered

# Initialize an empty list to store data
# data = []

# # Iterate over each row in the original dataframe
# for index, row in df.iterrows():
#     beta_ema = row['beta_ema']
#     # Iterate over each draft_prob in the row
#     for i, draft_prob in enumerate(row['draft_probs']):
#         # Check if the current index matches or is less than the accept count
#         if i < row['accept_cnt']:
#             accepted = 1
#         elif i == row['accept_cnt']:
#             accepted = 0
#         else:
#             continue

#         # Append a tuple to the list
#         data.append((beta_ema, draft_prob, accepted))

# # Create the DataFrame from the list
# df_draft = pd.DataFrame(data, columns=['beta_ema', 'draft_prob', 'accepted'])

# # Print the DataFrame
# print(df_draft)

# Save the DataFrame to a pickle file for later use
# df_draft.to_pickle('beta_list_output_draft_probs_accepted.pkl')

# Read the DataFrame from the pickle file
df_draft = pd.read_pickle('beta_list_output_draft_probs_accepted.pkl')

# Define bins
beta_ema_bins = np.linspace(0, 1, 21)
draft_prob_bins = np.linspace(0, 1, 21)

# Binning the data
df_draft['beta_ema_binned'] = pd.cut(df_draft['beta_ema'], bins=beta_ema_bins, labels=np.linspace(0, 1, 20, endpoint=False))
df_draft['draft_prob_binned'] = pd.cut(df_draft['draft_prob'], bins=draft_prob_bins, labels=np.linspace(0, 1, 20, endpoint=False))

# Group and aggregate data
df_grouped = df_draft.groupby(['beta_ema_binned', 'draft_prob_binned']).agg({'accepted': 'mean'})
df_grouped.reset_index(inplace=True)

# Convert categories to codes to be used in regression
df_grouped['beta_ema_binned_code'] = df_grouped['beta_ema_binned'].cat.codes
df_grouped['draft_prob_binned_code'] = df_grouped['draft_prob_binned'].cat.codes

# Prepare data for regression
X = df_grouped[['beta_ema_binned_code', 'draft_prob_binned_code']].values
y = df_grouped['accepted'].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

# Linear regression
model = LinearRegression()
model.fit(X_poly, y)

# Make predictions using the model over the grid
X_grid = np.array([(beta, draft) for beta in range(21) for draft in range(21)])
X_grid_poly = poly.transform(X_grid)
y_grid_pred = model.predict(X_grid_poly).reshape(21, 21)

# Pivot table for the heatmap
df_pivot = df_grouped.pivot('beta_ema_binned', 'draft_prob_binned', 'accepted')

# Plotting the actual data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(df_pivot, cmap='Accent', interpolation='nearest', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar()
plt.xlabel('Draft Probability')
plt.ylabel('Beta EMA')
plt.title('Actual Acceptance Rate Heatmap')

# Plotting predicted data
plt.subplot(1, 2, 2)
plt.imshow(y_grid_pred, cmap='Accent', interpolation='nearest', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar()
plt.xlabel('Draft Probability')
plt.ylabel('Beta EMA')
plt.title('Predicted Acceptance Rate Heatmap')

plt.savefig('acceptance_rate_heatmap_comparison.png')
plt.show()