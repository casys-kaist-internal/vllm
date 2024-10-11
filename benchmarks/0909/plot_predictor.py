import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def plot_combined_graphs(prob_file_path):
    data_list = []
    current_temperature = None

    # Read and parse the CSV file
    with open(prob_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            parts = [p.strip() for p in parts]
            if len(parts) == 2:
                # Temperature line
                if parts[0] == 'accept_prob':
                    temp_value = parts[1]
                    try:
                        current_temperature = float(temp_value)
                    except ValueError:
                        current_temperature = None
                else:
                    current_temperature = None
            elif len(parts) == 3:
                # Data point line
                if current_temperature is None:
                    continue  # Skip if temperature is not set
                accepted_str = parts[1]
                predicted_accept_prob_str = parts[2]
                accepted = 1 if accepted_str == 'True' else 0
                try:
                    predicted_accept_prob = float(predicted_accept_prob_str)
                except ValueError:
                    continue
                data_list.append({
                    'accepted': accepted,
                    'predicted_accept_prob': predicted_accept_prob,
                    'temperature': current_temperature
                })

    # Create DataFrame from the parsed data
    data = pd.DataFrame(data_list)

    # Define bin edges for calibration curve
    bins = np.linspace(0, 1, 11)

    # Create subplots for Calibration Curve and ROC Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Specify the temperatures in the desired order
    temperatures_in_order = [0.0, 0.25, 0.5, 0.75]

    # Part 1: Calibration Curves
    for temp in temperatures_in_order:
        temp_data = data[data['temperature'] == temp].copy()
        if temp_data.empty:
            continue  # Skip if no data for this temperature
        # Bin the predicted probabilities
        temp_data['bins'] = pd.cut(temp_data['predicted_accept_prob'], bins, include_lowest=True)

        # Calculate mean actual acceptance and predicted probabilities per bin
        binned_data = temp_data.groupby('bins').agg({
            'accepted': 'mean',
            'predicted_accept_prob': 'mean',
            'bins': 'count'
        }).rename(columns={'bins': 'count'})

        # Calculate Expected Calibration Error (ECE)
        n = temp_data.shape[0]
        bin_counts = binned_data['count']
        ece = np.sum((bin_counts / n) * np.abs(binned_data['accepted'] - binned_data['predicted_accept_prob']))

        # Plot Calibration Curve
        ax1.plot(binned_data['predicted_accept_prob'], binned_data['accepted'], 'o-',
                 label=f'Temp {temp:.2f} (ECE = {ece:.3f})')

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1)
    ax1.set_xlabel('Predicted Acceptance Probability', fontsize=12)
    ax1.set_ylabel('Actual Acceptance Probability', fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True)
    ax1.set_title('Calibration Curves')

    # Part 2: ROC Curves
    for temp in temperatures_in_order:
        temp_data = data[data['temperature'] == temp]
        if temp_data.empty:
            continue

        actual_accept_prob_binary = temp_data['accepted']

        # Ensure at least one positive and one negative sample
        if len(np.unique(actual_accept_prob_binary)) < 2:
            continue

        # Calculate AUROC and ROC Curve
        auroc = roc_auc_score(actual_accept_prob_binary, temp_data['predicted_accept_prob'])
        fpr, tpr, _ = roc_curve(actual_accept_prob_binary, temp_data['predicted_accept_prob'])

        # Plot ROC Curve
        ax2.plot(fpr, tpr, lw=2, label=f'Temp {temp:.2f} (AUROC = {auroc:.2f})')

    # Baseline ROC Curve
    ax2.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.015])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True)
    ax2.set_title('ROC Curves')

    plt.tight_layout()
    plt.savefig('roc.png', dpi=300)
    plt.savefig('roc.pdf', format='pdf', dpi=300)
    plt.show()

    # Part 3: Separate Density Plots for Each Temperature
    # Create a new figure with subplots for density plots
    num_temps = len(temperatures_in_order)
    cols = 2  # Number of columns in the subplot grid
    rows = (num_temps + 1) // cols  # Compute the number of rows needed

    fig_density, axes_density = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    axes_density = axes_density.flatten()  # Flatten in case of single row

    for idx, temp in enumerate(temperatures_in_order):
        temp_data = data[data['temperature'] == temp]
        if temp_data.empty:
            continue  # Skip if no data for this temperature

        ax = axes_density[idx]
        # Plot density histogram for each temperature
        ax.hist(temp_data['predicted_accept_prob'], bins=20, alpha=0.7,
                color='blue', edgecolor='black')
        ax.set_xlabel('Predicted Acceptance Probability', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Temperature {temp:.2f}', fontsize=14)
        ax.grid(True)

    # Remove any empty subplots
    for idx in range(len(temperatures_in_order), len(axes_density)):
        fig_density.delaxes(axes_density[idx])

    plt.tight_layout()
    plt.savefig('density_plots.png', dpi=300)
    plt.savefig('density_plots.pdf', format='pdf', dpi=300)
    plt.show()


# Example usage
plot_combined_graphs('ece_all.csv')
# plot_combined_graphs('ece_0.csv')
# plot_combined_graphs('ece_0_25.csv')
# plot_combined_graphs('ece_0_5.csv')
# plot_combined_graphs('ece_0_75.csv')
# plot_combined_graphs('pretemp_ece_all.csv')
# plot_combined_graphs('pretemp_ece_0.csv')
# plot_combined_graphs('pretemp_ece_0_25.csv')
# plot_combined_graphs('pretemp_ece_0_5.csv')
# plot_combined_graphs('pretemp_ece_0_75.csv')
