import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Function to create a polynomial equation string
def create_polynomial_equation(model, feature_names):
    # Extract coefficients and intercept from the model
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Construct equation terms with coefficients
    terms = [f"{coeff}*{name}" for coeff, name in zip(coefficients, feature_names) if coeff != 0]
    
    # Add intercept
    equation = " + ".join(terms) + f" + {intercept:.2f}"
    
    return equation

# Function to read CSV files and add metadata columns
def read_csv_files_in_dir(directory):
    data_list = []
    for f in os.listdir(directory):
        if f.endswith('.csv'):
            parts = f.split('_')
            degree = parts[1]
            agg_type = parts[2]
            target_model = parts[3]
            draft_model = parts[4]
            temperature = parts[5] + "." + parts[6]
            dataset = parts[7].split('.')[0]
            df = pd.read_csv(os.path.join(directory, f))

            df.insert(0, 'degree', "degree3")
            df.insert(1, 'agg_type', agg_type)
            df.insert(2, 'target_model', target_model)
            df.insert(3, 'draft_model', draft_model)
            df.insert(4, 'temperature', temperature)
            df.insert(5, 'dataset', dataset)

            # Strip the column names
            df.columns = df.columns.str.strip()

            # Append the DataFrame to the list
            data_list.append(df)

    # Concatenate all DataFrames into one DataFrame
    all_data = pd.concat(data_list, ignore_index=True)
    return all_data

# Directory containing the CSV files
directory = './'

# Read all CSV files in the directory
all_data = read_csv_files_in_dir(directory)

print(all_data)

# Get unique pairs of target_model and draft_model
unique_pairs = all_data[['target_model', 'draft_model']].drop_duplicates()

# Generate a grid of beta_ema and draft_prob values
num_bins = 20
beta_values = np.linspace(0, 1, num_bins)
draft_values = np.linspace(0, 1, num_bins)
X_predict = np.array([[beta, draft] for beta in beta_values for draft in draft_values])

# Plotting for each unique pair
for _, pair in unique_pairs.iterrows():
    target_model = pair['target_model']
    draft_model = pair['draft_model']

    # Filter data for the current pair
    pair_data = all_data[(all_data['target_model'] == target_model) & (all_data['draft_model'] == draft_model)]

    # Get unique datasets and temperatures
    unique_datasets = pair_data['dataset'].unique()
    unique_temperatures = sorted(pair_data['temperature'].unique())

    # Initialize the subplots
    num_datasets = len(unique_datasets)
    num_temperatures = len(unique_temperatures)
    fig, axes = plt.subplots(num_datasets, num_temperatures, figsize=(5 * num_temperatures, 4 * num_datasets), sharex=True, sharey=True)

    for i, dataset in enumerate(unique_datasets):
        for j, temperature in enumerate(unique_temperatures):
            ax = axes[i, j]

            # Filter the data for the current combination of dataset and temperature
            filtered_data = pair_data[(pair_data['dataset'] == dataset) & (pair_data['temperature'] == temperature)]
            print(filtered_data)
            if filtered_data.empty:
                ax.axis('off')
                continue

            # Retrieve the coefficients and intercept
            coef = filtered_data['coef'].values
            intercept = filtered_data['intercept'].values[0]

            # Initialize polynomial features and linear model
            degree = int(filtered_data['degree'].values[0][-1])
            print(f'Degree: {degree}')
            poly_features = PolynomialFeatures(degree=degree, include_bias=True)
            poly_features.fit([[0, 0]])  # fit on dummy data to get feature names
            feature_names = poly_features.get_feature_names_out(['beta_ema', 'draft_prob'])

            # Transform the prediction grid
            X_poly = poly_features.transform(X_predict)

            # Initialize and set the linear model coefficients and intercept
            linear_model = LinearRegression(fit_intercept=True)
            linear_model.coef_ = coef
            linear_model.intercept_ = intercept

            print(f"c: {linear_model.coef_}, i: {linear_model.intercept_}")

            # Predict the acceptance probabilities for the grid
            y_predict = linear_model.predict(X_poly).reshape(num_bins, num_bins)

            # Plot the heatmap
            cax = ax.imshow(y_predict, cmap='Accent', interpolation='nearest', origin='lower', extent=[0, 1, 0, 1])
            ax.set_title(f'{dataset}, Temp: {temperature}', fontsize=10)
            ax.set_xlabel('Draft Probability')
            ax.set_ylabel('Beta EMA')

            # Add a colorbar to the right of the subplots
            fig.colorbar(cax, ax=ax, orientation='vertical')

    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(f'{target_model}_{draft_model}_heatmap.png')
    plt.close(fig)  # Close the figure to avoid displaying it in the notebook
