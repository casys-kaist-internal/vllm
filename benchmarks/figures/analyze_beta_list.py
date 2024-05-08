# Read output.txt in pandas 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from ast import literal_eval
import math

# current date time 
from datetime import datetime
now = datetime.now()
dir_name = now.strftime("%Y%m%d_%H")

import os
os.makedirs(dir_name, exist_ok=True)


def safe_literal_eval(s):
    try:
        # Attempt to evaluate the string as a Python literal
        if s == '[]':
            return []
        return literal_eval(s)
    except ValueError:
        # Return the original string if literal_eval fails
        return s

# read only the first 100 lines
df = pd.read_csv('/home/sjchoi/workspace/beta_list_output_draft_probs_apps_temp_0_75_5_7_penalty_0_3.txt', sep='|', header=None)
df.columns = ['dummy', 'accept_cnt', 'previous_beta_list', 'previous_accept_prob_list', 'draft_probs', 'accept_cnt_list', 'accept_probs', 'current_beta_list']

# drop the last row
df = df.drop(df.tail(1).index)

df['accept_cnt'] = df['accept_cnt'].astype(int)
df['previous_beta_list'] = df['previous_beta_list'].apply(lambda x: safe_literal_eval(x.strip()))
df['previous_accept_prob_list'] = df['previous_accept_prob_list'].apply(lambda x: safe_literal_eval(x.strip()))
df['draft_probs'] = df['draft_probs'].apply(lambda x: safe_literal_eval(x.strip()))
df['accept_cnt_list'] = df['accept_cnt_list'].apply(lambda x: safe_literal_eval(x.strip()))
df['accept_probs'] = df['accept_probs'].apply(lambda x: safe_literal_eval(x.strip()))
df['current_beta_list'] = df['current_beta_list'].apply(lambda x: safe_literal_eval(x.strip()))
df = df.dropna()
df = df.drop(columns=['dummy'])

# if len of previous_beta_list is 0, drop the row
df = df[df['previous_beta_list'].map(len) > 30]

# Define the objective function
def objective(beta):
    result = (1 - beta**8) / (1 - beta**1)
    return result

def cut_over_1(accept_prob_list):
    for i in range(len(accept_prob_list)):
        if accept_prob_list[i] > 1:
            accept_prob_list[i] = 1
    return accept_prob_list


# calculate beta_ema from previous_beta_list
def beta_ema(beta_list, factor):
    beta_ema = beta_list[0]

    for i in range(1, len(beta_list)):
        beta_ema = beta_ema * (1 - factor) + beta_list[i] * factor

    return beta_ema
    
# # apply beta_ema to previous_beta_list
df['beta_ema_0_90'] = df['previous_beta_list'].apply(lambda x: beta_ema(x, 0.9))
df['beta_ema_0_80'] = df['previous_beta_list'].apply(lambda x: beta_ema(x, 0.8))
df['beta_ema_0_70'] = df['previous_beta_list'].apply(lambda x: beta_ema(x, 0.7))
df['beta_ema_0_60'] = df['previous_beta_list'].apply(lambda x: beta_ema(x, 0.6))
df['beta_ema_0_50'] = df['previous_beta_list'].apply(lambda x: beta_ema(x, 0.5))
df['beta_ema_0_40'] = df['previous_beta_list'].apply(lambda x: beta_ema(x, 0.4))
df['beta_ema_0_30'] = df['previous_beta_list'].apply(lambda x: beta_ema(x, 0.3))
df['beta_ema_0_20'] = df['previous_beta_list'].apply(lambda x: beta_ema(x, 0.2))
df['beta_ema_0_10'] = df['previous_beta_list'].apply(lambda x: beta_ema(x, 0.1))
df['beta_ema_0_05'] = df['previous_beta_list'].apply(lambda x: beta_ema(x, 0.05))
df['beta_ema_0_01'] = df['previous_beta_list'].apply(lambda x: beta_ema(x, 0.01))

# apply cut_over_1 to accept_probs
df['previous_accept_prob_list'] = df['previous_accept_prob_list'].apply(lambda x: cut_over_1(x))

# apply accept_prob ema to previous_accept_prob_list
df['accept_prob_ema_0_90'] = df['previous_accept_prob_list'].apply(lambda x: beta_ema(x, 0.9))
df['accept_prob_ema_0_80'] = df['previous_accept_prob_list'].apply(lambda x: beta_ema(x, 0.8))
df['accept_prob_ema_0_70'] = df['previous_accept_prob_list'].apply(lambda x: beta_ema(x, 0.7))
df['accept_prob_ema_0_60'] = df['previous_accept_prob_list'].apply(lambda x: beta_ema(x, 0.6))
df['accept_prob_ema_0_50'] = df['previous_accept_prob_list'].apply(lambda x: beta_ema(x, 0.5))
df['accept_prob_ema_0_40'] = df['previous_accept_prob_list'].apply(lambda x: beta_ema(x, 0.4))  
df['accept_prob_ema_0_30'] = df['previous_accept_prob_list'].apply(lambda x: beta_ema(x, 0.3))
df['accept_prob_ema_0_20'] = df['previous_accept_prob_list'].apply(lambda x: beta_ema(x, 0.2))
df['accept_prob_ema_0_10'] = df['previous_accept_prob_list'].apply(lambda x: beta_ema(x, 0.1))
df['accept_prob_ema_0_05'] = df['previous_accept_prob_list'].apply(lambda x: beta_ema(x, 0.05))
df['accept_prob_ema_0_01'] = df['previous_accept_prob_list'].apply(lambda x: beta_ema(x, 0.01))

# # Save dataframe as object for later use
df.to_pickle('beta_list_output_draft_probs.pkl')

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

# 11.14 x - 20.5 x + 10.24 x - 0.2109 x + 0.4864
def polynomial_fit(beta):
    a = 11.14
    b = -20.5
    c = 10.24
    d = -0.2109
    e = 0.4864
    result = a * beta**4 + b * beta**3 + c * beta**2 + d * beta + e
    return round(result)

# read pickle file
df = pd.read_pickle('beta_list_output_draft_probs.pkl')

decay_rate = 0.5
name = 'beta_ema_0_50'
# name = 'accept_prob_ema_0_50'
print(df)

# beta mean value of previous_beta_list
df['beta_mean'] = df['previous_beta_list'].apply(lambda x: sum(x) / len(x))
df['accept_prob_mean'] = df['previous_accept_prob_list'].apply(lambda x: sum(x) / len(x))

df['beta_ema'] = df[name]

#################################################################################################################################
# 1. Plot the distribution of accept_cnt 
#################################################################################################################################
# max_value = df['accept_cnt'].max()
# bins = np.arange(-0.5, max_value + 1.5)  # Extending to max_value + 1 and starting from -0.5

# # Plotting the distribution of accept_cnt
# plt.figure()
# plt.hist(df['accept_cnt'], bins=bins, edgecolor='black', align='mid')  # Aligns the bins centered on integers
# plt.xlabel('accept_cnt')
# plt.ylabel('Frequency')
# plt.title('Distribution of accept_cnt')
# plt.xticks(range(max_value + 1))  # Setting x-axis ticks to cover all integer values from 0 to max_value
# plt.savefig(f'{dir_name}/1_accept_cnt_distribution.png')
# plt.show()  # Display the plot

#################################################################################################################################
# 2. Plot the distribution of beta_ema
#################################################################################################################################

# # the beta values are between 0 and 1
# bins = np.linspace(0, 1, 21)  # 20 bins between 0 and 1

# # Plotting the distribution of beta_ema 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5 in several subfigures
# plt.figure(figsize=(20, 12))
# plt.suptitle('Distribution of beta_ema', fontsize=16)

# # Plotting beta_ema 0.01
# plt.subplot(3, 3, 1)
# plt.hist(df['beta_ema_0_01'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('beta_ema_0_01')
# plt.ylabel('Frequency')
# plt.title('Distribution of beta_ema 0.01')

# # Plotting beta_ema 0.05
# plt.subplot(3, 3, 2)
# plt.hist(df['beta_ema_0_05'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('beta_ema_0_05')
# plt.ylabel('Frequency')
# plt.title('Distribution of beta_ema 0.05')

# # Plotting beta_ema 0.1
# plt.subplot(3, 3, 3)
# plt.hist(df['beta_ema_0_10'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('beta_ema_0_10')
# plt.ylabel('Frequency')
# plt.title('Distribution of beta_ema 0.1')

# # Plotting beta_ema 0.2
# plt.subplot(3, 3, 4)
# plt.hist(df['beta_ema_0_20'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('beta_ema_0_20')
# plt.ylabel('Frequency')
# plt.title('Distribution of beta_ema 0.2')

# # Plotting beta_ema 0.3
# plt.subplot(3, 3, 5)
# plt.hist(df['beta_ema_0_30'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('beta_ema_0_30')
# plt.ylabel('Frequency')
# plt.title('Distribution of beta_ema 0.3')

# # Plotting beta_ema 0.4
# plt.subplot(3, 3, 6)
# plt.hist(df['beta_ema_0_40'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('beta_ema_0_40')
# plt.ylabel('Frequency')
# plt.title('Distribution of beta_ema 0.4')

# # Plotting beta_ema 0.5
# plt.subplot(3, 3, 7)
# plt.hist(df['beta_ema_0_50'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('beta_ema_0_50')
# plt.ylabel('Frequency')
# plt.title('Distribution of beta_ema 0.5')

# # Plotting beta_ema 0.60
# plt.subplot(3, 3, 8)
# plt.hist(df['beta_ema_0_60'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('beta_ema_0_60')
# plt.ylabel('Frequency')
# plt.title('Distribution of beta_ema 0.6')

# # Plotting beta_ema 0.70
# plt.subplot(3, 3, 9)
# plt.hist(df['beta_mean'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('beta_mean')
# plt.ylabel('Frequency')
# plt.title('Distribution of beta_ema mean')

# plt.tight_layout()
# plt.savefig(f'{dir_name}/2_beta_ema_distribution.png')
# plt.show()  # Display the plot


# # the beta values are between 0 and 1
# bins = np.linspace(0, 1, 21)  # 20 bins between 0 and 1

# # Plotting the distribution of beta_ema 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5 in several subfigures
# plt.figure(figsize=(20, 12))
# plt.suptitle('Distribution of accept_probs', fontsize=16)

# # Plotting accept_prob ema 0.01
# plt.subplot(3, 3, 1)
# plt.hist(df['accept_prob_ema_0_01'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('accept_prob_ema_0_01')
# plt.ylabel('Frequency')
# plt.title('Distribution of accept_prob_ema 0.01')

# # Plotting accept_prob ema 0.05
# plt.subplot(3, 3, 2)
# plt.hist(df['accept_prob_ema_0_05'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('accept_prob_ema_0_05')
# plt.ylabel('Frequency')
# plt.title('Distribution of accept_prob_ema 0.05')

# # Plotting accept_prob ema 0.1
# plt.subplot(3, 3, 3)
# plt.hist(df['accept_prob_ema_0_10'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('accept_prob_ema_0_10')
# plt.ylabel('Frequency')
# plt.title('Distribution of accept_prob_ema 0.10')

# # Plotting accept_prob ema 0.2
# plt.subplot(3, 3, 4)
# plt.hist(df['accept_prob_ema_0_20'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('accept_prob_ema_0_20')
# plt.ylabel('Frequency')
# plt.title('Distribution of accept_prob_ema 0.20')

# # Plotting accept_prob ema 0.3
# plt.subplot(3, 3, 5)
# plt.hist(df['accept_prob_ema_0_30'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('accept_prob_ema_0_30')
# plt.ylabel('Frequency')
# plt.title('Distribution of accept_prob_ema 0.30')

# # Plotting accept_prob ema 0.4
# plt.subplot(3, 3, 6)
# plt.hist(df['accept_prob_ema_0_40'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('accept_prob_ema_0_40')
# plt.ylabel('Frequency')
# plt.title('Distribution of accept_prob_ema 0.40')

# # Plotting accept_prob ema 0.5
# plt.subplot(3, 3, 7)
# plt.hist(df['accept_prob_ema_0_50'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('accept_prob_ema_0_50')
# plt.ylabel('Frequency')
# plt.title('Distribution of accept_prob_ema 0.50')

# # Plotting accept_prob ema 0.60 
# plt.subplot(3, 3, 8)
# plt.hist(df['accept_prob_ema_0_60'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('accept_prob_ema_0_60')
# plt.ylabel('Frequency')
# plt.title('Distribution of accept_prob_ema 0.60')

# # Plotting accept_prob ema 0.70 
# plt.subplot(3, 3, 9)
# plt.hist(df['accept_prob_ema_0_70'], bins=bins, edgecolor='black', align='mid')
# plt.xlabel('accept_prob_ema_0_70')
# plt.ylabel('Frequency')
# plt.title('Distribution of accept_prob_ema 0.70')

# plt.tight_layout()
# plt.savefig(f'{dir_name}/2_accept_prob_ema_distribution.png')
# plt.show()  # Display the plot  




#################################################################################################################################
# 3. Divide beta in 20 bins and calculate the mean of accept_cnt
#################################################################################################################################

# Define the bin edges
# bin_edges = np.linspace(0, 1, 41)  # 20 bins between 0 and 1

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
# bin_labels = bin_labels[:len(df_grouped)]  # Truncate to fit the DataFrame
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

# plt.savefig(f'{dir_name}/3_accept_cnt_vs_beta_{decay_rate}.png')  # Save the plot
# plt.show()  # Display the plot

#################################################################################################################################
# 4. Relationship between draft_prob, beta_ema and accept_prob
#################################################################################################################################

# Initialize an empty list to store data
data = []

# Iterate over each row in the original dataframe
for index, row in df.iterrows():
    beta_ema = row['beta_ema']
    # Iterate over each draft_prob in the row
    for i, draft_prob in enumerate(row['draft_probs']):
        # Append a tuple to the list
        # data.append((beta_ema, draft_prob, min(row['accept_probs'][i], 1)))
        data.append((beta_ema, draft_prob, min(row['accept_probs'][i], 1)))

# Create the DataFrame from the list
df_accept = pd.DataFrame(data, columns=['beta_ema', 'draft_prob', 'accept_prob'])
print(df_accept)


# # # Save the DataFrame to a pickle file for later use
df_accept.to_pickle('beta_list_output_draft_probs_accepted_prob.pkl')

# Read the DataFrame from the pickle file
df_accept = pd.read_pickle('beta_list_output_draft_probs_accepted_prob.pkl')

# Define bins
beta_ema_bins = np.linspace(0, 1, 21)
draft_prob_bins = np.linspace(0, 1, 21)

# Binning the data  
df_accept['beta_ema_binned'] = pd.cut(df_accept['beta_ema'], bins=beta_ema_bins, labels=np.linspace(0, 1, 20, endpoint=False))
df_accept['draft_prob_binned'] = pd.cut(df_accept['draft_prob'], bins=draft_prob_bins, labels=np.linspace(0, 1, 20, endpoint=False))
                                        

# # Group and aggregate data
# df_grouped = df_draft.groupby(['beta_ema_binned', 'draft_prob_binned']).agg({'accepted': 'mean'})
# df_grouped.reset_index(inplace=True)

# # Pivot table for the actual heatmap
# df_pivot_actual = df_grouped.pivot('beta_ema_binned', 'draft_prob_binned', 'accepted')

# Group and aggregate data with unique names
df_grouped = df_accept.groupby(['beta_ema_binned', 'draft_prob_binned']).agg(
    accept_prob_median=('accept_prob', 'median'),
    accept_prob_mean=('accept_prob', 'mean'),
    accept_prob_std=('accept_prob', 'std')
)
df_grouped.reset_index(inplace=True)

# Pivot table for the actual heatmap
df_pivot_actual_mean = df_grouped.pivot('beta_ema_binned', 'draft_prob_binned', 'accept_prob_mean')
df_pivot_actual_median = df_grouped.pivot('beta_ema_binned', 'draft_prob_binned', 'accept_prob_median')
df_pivot_actual_std = df_grouped.pivot('beta_ema_binned', 'draft_prob_binned', 'accept_prob_std')

# Plotting heatmap 
# subplot 3 x 1
fig, axes = plt.subplots(3, 1, figsize=(20, 18))
# first subplot
im_actual_mean = axes[0].imshow(df_pivot_actual_mean, cmap='Accent', interpolation='nearest', origin='lower', extent=[0, 1, 0, 1])
axes[0].set_title('Actual Acceptance Probability Heatmap (Mean)')
axes[0].set_xlabel('Draft Probability')
axes[0].set_ylabel('Beta EMA')
fig.colorbar(im_actual_mean, ax=axes[0])

# second subplot
im_actual_median = axes[1].imshow(df_pivot_actual_median, cmap='Accent', interpolation='nearest', origin='lower', extent=[0, 1, 0, 1])
axes[1].set_title('Actual Acceptance Probability Heatmap (Median)')
axes[1].set_xlabel('Draft Probability')
axes[1].set_ylabel('Beta EMA')
fig.colorbar(im_actual_median, ax=axes[1])

# third subplot
im_actual_std = axes[2].imshow(df_pivot_actual_std, cmap='Accent', interpolation='nearest', origin='lower', extent=[0, 1, 0, 1])
axes[2].set_title('Actual Acceptance Probability Heatmap (Std)')
axes[2].set_xlabel('Draft Probability')
axes[2].set_ylabel('Beta EMA')
fig.colorbar(im_actual_std, ax=axes[2])

plt.savefig(f'{dir_name}/4_acceptance_probability_heatmap_{decay_rate}.png')

##################################################################################################################################
# 5. Plot the distribution of representative bins for beta_ema and draft_prob
##################################################################################################################################

# Define beta_ema and draft_prob representative ranges
beta_ema_bins = np.linspace(0.05, 0.85, 9)
draft_prob_bins = np.linspace(0.05, 0.85, 9)

# Create a figure with subplots of size 9x9
fig, axes = plt.subplots(len(beta_ema_bins), len(draft_prob_bins), figsize=(20, 20), sharex=True, sharey=True)

# Iterate through each combination of beta_ema and draft_prob starting from the bottom-left
for row_idx, beta_ema in enumerate(reversed(beta_ema_bins)):
    for col_idx, draft_prob in enumerate(draft_prob_bins):
        ax = axes[row_idx, col_idx]

        # Filter the data for the specific bin
        df_bin = df_accept[(df_accept['beta_ema'] >= beta_ema - 0.05) & (df_accept['beta_ema'] < beta_ema + 0.05) &
                           (df_accept['draft_prob'] >= draft_prob - 0.05) & (df_accept['draft_prob'] < draft_prob + 0.05)]

        # Plot the histogram of accept_prob for the bin in the current axis
        counts, bins, patches = ax.hist(df_bin['accept_prob'], bins=20, edgecolor='black', align='mid', density=True)

        ax.set_title(f'β_ema: {beta_ema:.2f}, Draft Prob: {draft_prob:.2f}', fontsize=8)
        ax.set_xlabel('accept_prob', fontsize=6)
        ax.set_ylabel('density', fontsize=6)

        # Adjust tick parameters for readability
        ax.tick_params(axis='both', labelsize=6)

# Adjust the layout and set the figure title
fig.suptitle(f'Percentage Distribution of accept_prob for All Beta EMA {decay_rate} & Draft Probability Bins', fontsize=18)
plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])  # Adjust layout to leave space for the title

# Save the plot
plt.savefig(f'{dir_name}/5_accept_prob_percentage_distribution_9x9_grid_{decay_rate}.png')
plt.show()


# #################################################################################################################################
# # 6. Training a polynomial regression model to predict accept prob
# #################################################################################################################################

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Select features and target
X = df_accept[['beta_ema', 'draft_prob']]
y = df_accept['accept_prob']

# Define the degree of the polynomial
degree = 2  # You can adjust this degree based on the complexity you want to model

# Create a pipeline that includes polynomial feature creation and linear regression
polynomial_model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('linear', LinearRegression(fit_intercept=True))
])

# Fit the model
polynomial_model.fit(X, y)

# Extract the linear regression model after polynomial transformation
linear_model = polynomial_model.named_steps['linear']

# Display coefficients and intercept
print("Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)

# Generate predictions across the grid for visualization
X_predict = np.array([[beta, draft] for beta in np.linspace(0, 1, 20) for draft in np.linspace(0, 1, 20)])
y_predict = polynomial_model.predict(X_predict).reshape(20, 20)

# Convert the predictions to a DataFrame for easier plotting (if needed)
df_predict = pd.DataFrame(y_predict, index=np.linspace(0, 1, 20), columns=np.linspace(0, 1, 20))

# Plotting predicted data heatmap
plt.figure()
plt.imshow(df_predict, cmap='Accent', interpolation='nearest', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar()
plt.title('Predicted Polynomial Acceptance Probability Heatmap')
plt.xlabel('Draft Probability')
plt.ylabel('Beta EMA')
plt.show()
plt.savefig(f'{dir_name}/polynomial_acceptance_probability_heatmap.png')

# print the polynomial function
def create_polynomial_equation(model, feature_names):
    # Extract coefficients and intercept from the model
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Construct equation terms with coefficients
    terms = [f"{coeff:.2f}*{name}" for coeff, name in zip(coefficients, feature_names) if coeff != 0]
    
    # Add intercept
    equation = " + ".join(terms) + f" + {intercept:.2f}"
    
    return equation

# Feature names from PolynomialFeatures
feature_names = polynomial_model.named_steps['poly'].get_feature_names_out(['beta_ema', 'draft_prob'])

# Create polynomial function string
polynomial_function_string_mean = create_polynomial_equation(linear_model, feature_names)

# Print the polynomial function string
print("Polynomial Function String:")
print(polynomial_function_string_mean)


# Determine the common scale for the colorbar
vmin = min(df_pivot_actual_median.min().min(), df_predict.min().min())
vmax = max(df_pivot_actual_median.max().max(), df_predict.max().max())

# Plotting both actual and predicted heatmaps with function names in titles
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Define a figure with two subplots (side by side) 

# Actual data heatmap
im_actual = axes[0].imshow(df_pivot_actual_mean, cmap='Accent', interpolation='nearest', origin='lower', extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
axes[0].set_title(f'{polynomial_function_string_mean}')
axes[0].set_xlabel('Draft Probability')
axes[0].set_ylabel('Beta EMA')

# Predicted data heatmap
im_predicted = axes[1].imshow(df_predict, cmap='Accent', interpolation='nearest', origin='lower', extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
axes[1].set_xlabel('Draft Probability')
axes[1].set_ylabel('Beta EMA')

# Adding a colorbar to the actual heatmap



# Create a colorbar that is shared between the subplots
fig.colorbar(im_predicted, ax=axes, orientation='vertical', fraction=.1).set_label('Acceptance Probability')

# Adjust layout to prevent overlap
# plt.tight_layout()

# Save the figure
plt.savefig(f'{dir_name}/6_polynomial_regression_acceptance_probability_heatmap_with_mean.png')
plt.show()

# #################################################################################################################################
# # 7. Training a polynomial regression model to predict accept prob median
# #################################################################################################################################


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Aggregate the data to calculate the median acceptance probability
beta_ema_bins = np.linspace(0, 1, 21)
draft_prob_bins = np.linspace(0, 1, 21)

# Bin data
df_accept['beta_ema_binned'] = pd.cut(df_accept['beta_ema'], bins=beta_ema_bins, labels=np.linspace(0, 1, 20, endpoint=False))
df_accept['draft_prob_binned'] = pd.cut(df_accept['draft_prob'], bins=draft_prob_bins, labels=np.linspace(0, 1, 20, endpoint=False))

# Group and calculate the median acceptance probability
df_grouped = df_accept.groupby(['beta_ema_binned', 'draft_prob_binned']).agg({'accept_prob': 'median'})
df_grouped.reset_index(inplace=True)

# Ensure numeric conversion of the binned features
X = df_grouped[['beta_ema_binned', 'draft_prob_binned']].apply(pd.to_numeric, errors='coerce')
y = df_grouped['accept_prob']

# Checking for NaN values in the grouped data
missing_values = y.isna().sum()
if missing_values > 0:
    print(f"Number of missing values in 'accept_prob': {missing_values}")
else:
    print("No missing values in 'accept_prob'")

# Drop rows with missing values in 'accept_prob'
df_grouped_clean = df_grouped.dropna(subset=['accept_prob'])

# Prepare features (binned beta_ema and draft_prob) and target (median accept_prob)
X = df_grouped_clean[['beta_ema_binned', 'draft_prob_binned']].apply(lambda x: pd.to_numeric(x, errors='coerce'))
y = df_grouped_clean['accept_prob']

# Ensure there are no more missing values in the target variable
if y.isna().sum() == 0:
    print("No missing values in target variable 'accept_prob'.")
else:
    raise ValueError("Input 'y' still contains NaN values.")

# Define the polynomial degree
degree = 2

# Create a pipeline with PolynomialFeatures and LinearRegression
polynomial_model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('linear', LinearRegression(fit_intercept=True))
])

# Fit the model
polynomial_model.fit(X, y)

# Extract the linear regression model after polynomial transformation
linear_model = polynomial_model.named_steps['linear']

# Display coefficients and intercept
print("Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)

# Generate predictions across a grid for visualization
X_predict = np.array([[beta, draft] for beta in np.linspace(0, 1, 20) for draft in np.linspace(0, 1, 20)])
X_predict_df = pd.DataFrame(X_predict, columns=['beta_ema_binned', 'draft_prob_binned'])
X_predict_poly = polynomial_model.named_steps['poly'].transform(X_predict_df)

y_predict = polynomial_model.named_steps['linear'].predict(X_predict_poly).reshape(20, 20)

# Convert predictions to DataFrame for easier plotting
df_predict = pd.DataFrame(y_predict, index=np.linspace(0, 1, 20), columns=np.linspace(0, 1, 20))

# Create the polynomial function string for visualization
def create_polynomial_equation(model, feature_names):
    coefficients = model.coef_
    intercept = model.intercept_
    terms = [f"{coef:.2f}*{name}" for coef, name in zip(coefficients, feature_names) if coef != 0]
    equation = " + ".join(terms) + f" + {intercept:.2f}"
    return equation

# Feature names for the polynomial model
feature_names = polynomial_model.named_steps['poly'].get_feature_names_out(['beta_ema_binned', 'draft_prob_binned'])
polynomial_function_string = create_polynomial_equation(linear_model, feature_names)

# Determine common vmin and vmax for shared colorbar
vmin = min(df_grouped['accept_prob'].min(), df_predict.values.min())
vmax = max(df_grouped['accept_prob'].max(), df_predict.values.max())

# Plotting both actual (median) and predicted heatmaps with function names in titles
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual data heatmap (median values)
df_pivot_actual = df_grouped.pivot(index='beta_ema_binned', columns='draft_prob_binned', values='accept_prob')
im_actual = axes[0].imshow(df_pivot_actual, cmap='Accent', interpolation='nearest', origin='lower', extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
axes[0].set_title('Actual Median Acceptance Probability Heatmap')
axes[0].set_xlabel('Draft Probability')
axes[0].set_ylabel('Beta EMA')

# Predicted polynomial data heatmap
im_predicted = axes[1].imshow(df_predict, cmap='Accent', interpolation='nearest', origin='lower', extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
axes[1].set_title(f'Predicted Polynomial Acceptance Probability Heatmap\n{polynomial_function_string}')
axes[1].set_xlabel('Draft Probability')
axes[1].set_ylabel('Beta EMA')

# Create a shared colorbar for both plots
fig.colorbar(im_predicted, ax=axes, orientation='vertical', fraction=.1).set_label('Acceptance Probability')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{dir_name}/7_polynomial_regression_acceptance_probability_heatmap_with_median.png')
plt.show()

print("Mean")
print(polynomial_function_string_mean)
print("Median")
print(polynomial_function_string)

# #################################################################################################################################
# # 8. draft_prob and adjusted_beta_ema with accept_prob
# #################################################################################################################################
import pandas as pd

# # Function to predict accept_prob based on trained polynomial model
# def predict_accept_prob(beta_ema, draft_prob):
#     """
#     Predict acceptance probability given beta_ema and draft_prob.

#     Args:
#     - beta_ema (float): The exponential moving average of beta.
#     - draft_prob (float): The current draft probability.

#     Returns:
#     - float: Predicted acceptance probability.
#     """
#     input_data = pd.DataFrame([[beta_ema, draft_prob]], columns=['beta_ema_binned', 'draft_prob_binned'])
#     input_transformed = polynomial_model.named_steps['poly'].transform(input_data)
#     return polynomial_model.named_steps['linear'].predict(input_transformed)[0]

# # Initialize an empty list to store data
# data = []

# # Iterate over each row in the original dataframe
# for index, row in df.iterrows():
#     beta_ema = row['beta_ema']
#     draft_probs = row['draft_probs']
#     accept_probs = row['accept_probs']

#     # Initialize the first value of prev_predicted_accept_prob
#     prev_predicted_accept_prob = predict_accept_prob(beta_ema, draft_probs[0])

#     # Iterate over each draft_prob in the list, starting from the second element
#     for i in range(1, len(draft_probs)):
#         draft_prob = draft_probs[i]
#         prev_draft_prob = draft_probs[i - 1]

#         # Append a tuple to the list
#         accept_prob = min(accept_probs[i], 1)
#         data.append((beta_ema, prev_predicted_accept_prob, draft_prob, accept_prob))

# # Create the DataFrame from the list
# df_accept = pd.DataFrame(data, columns=['beta_ema', 'prev_predicted_accept_prob', 'draft_prob', 'accept_prob'])
# print(df_accept)

# # Save the DataFrame to a pickle file for later use
# df_accept.to_pickle('beta_list_output_prev_draft_probs_accepted_prob.pkl')

# # Read the DataFrame from the pickle file to ensure the data has been correctly saved
# df_accept = pd.read_pickle('beta_list_output_prev_draft_probs_accepted_prob.pkl')
# print(df_accept.head())  # Display the first few rows to verify the data
