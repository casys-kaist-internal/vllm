import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

def plot_combined_graphs(prob_file_path, roc_file_path):
    # Part 1: Calibration Curve and ECE Calculation
    # Load and process the data
    colors = ['#FFFFFF', '#D7E2F9', '#88BCFF', '#3864B9', '#1B345F']

    data = pd.read_csv(prob_file_path, header=None, names=['result', 'accepted', 'predicted_accept_prob'])

    # Clean and map 'accepted' column to integer (True becomes 1, False becomes 0)
    data['accepted'] = data['accepted'].astype(str).str.strip()
    data['accepted'] = data['accepted'].map({'True': 1, 'False': 0})

    # Define bin edges (e.g., 10 bins between 0 and 1)
    bins = np.linspace(0, 1, 11)

    # Bin the predicted accept probabilities
    data['bins'] = pd.cut(data['predicted_accept_prob'], bins, include_lowest=True)

    # Group by bins and calculate the actual acceptance rate (mean of accepted values)
    binned_data = data.groupby('bins').agg({
        'accepted': 'mean',  # Actual acceptance rate per bin
        'predicted_accept_prob': 'mean',  # Mean predicted probability per bin
        'bins': 'count'  # Number of samples per bin
    }).rename(columns={'bins': 'count'})

    # Get bin counts
    bin_counts = binned_data['count']

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate ECE
    n = data.shape[0]  # Total number of samples
    ece = np.sum((bin_counts / n) * np.abs(binned_data['accepted'] - binned_data['predicted_accept_prob']))

    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    # Prepare data for plotting
    mean_predicted_probs = binned_data['predicted_accept_prob']
    mean_actual_acceptance = binned_data['accepted']

    # Part 2: ROC Curve Plot
    temperatures = []
    actual_accept_probs = []
    predicted_accept_probs = []

    # Manually read and parse the CSV file for ROC
    with open(roc_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')

            if len(parts) == 2:
                current_temp = float(parts[1].strip())
            elif len(parts) == 3:
                actual_prob = float(parts[1].strip())
                predicted_prob = float(parts[2].strip())

                temperatures.append(current_temp)
                actual_accept_probs.append(actual_prob)
                predicted_accept_probs.append(predicted_prob)

    temperatures = np.array(temperatures)
    actual_accept_probs = np.array(actual_accept_probs)
    predicted_accept_probs = np.array(predicted_accept_probs)

    # Define the color palette
    roc_colors = ['#1B345F', '#3864B9', '#88BCFF', '#D7E2F9']

    # Create side-by-side subplots (2 columns) for a 2-column paper layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.6))  # Adjusted width and height for better fit

    # Part 1: Plot the Calibration Curve
    # Plot the perfect calibration line
    ax1.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1)

    # Plot the calibration curve
    line1, = ax1.plot(bin_centers, mean_actual_acceptance, 'o-', color=colors[3],
                      linewidth=2, label=f'Calibration (ECE = {ece:.3f})', markersize=5)

    # Plot the bin density as percentage on a secondary y-axis
    ax1_bar = ax1.twinx()
    bin_percentage = (bin_counts / bin_counts.sum()) * 100  # Convert counts to percentage
    bars = ax1_bar.bar(bin_centers, bin_percentage, width=0.05, alpha=0.4, color='#333333', label='Density (%)')
    ax1_bar.set_ylabel('Density (%)', fontsize=12)
    ax1_bar.set_ylim(0, bin_percentage.max() + 5)  # Adjust ylim for better visualization

    ax1.set_xlabel('Predicted Acceptance Prob', fontsize=12)
    ax1.set_ylabel('Actual Acceptance Prob', fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True)

    # Reduce font size for ticks
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1_bar.tick_params(axis='y', which='major', labelsize=12)

    # Combine legends from both ax1 and ax1_bar into one
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_bar.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=10)



    # Part 2: ROC Curve Plot
    unique_temps = np.unique(temperatures)
    lines = []
    labels = []

    for i, temp in reversed(list(enumerate(unique_temps))):
        mask = temperatures == temp
        actual_accept_prob_temp = actual_accept_probs[mask]
        predicted_accept_prob_temp = predicted_accept_probs[mask]

        # Binarize the actual accept_prob with a threshold of 0.5
        actual_accept_prob_binary = (actual_accept_prob_temp >= 0.5).astype(int)

        # Calculate the AUROC
        auroc = roc_auc_score(actual_accept_prob_binary, predicted_accept_prob_temp)

        # Calculate the ROC curve
        fpr, tpr, _ = roc_curve(actual_accept_prob_binary, predicted_accept_prob_temp)

        # Plot the ROC curve with the specified color and black border
        line, = ax2.plot(fpr, tpr, lw=2, color=roc_colors[i % len(roc_colors)],
                         label=f'Temp {temp:.2f} (AUROC = {auroc:.2f})')

        # Add black border around the line
        line.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])

        # Store line and label for the legend
        lines.append(line)
        labels.append(f'Temp {temp:.2f} (AUROC = {auroc:.2f})')

    # Plot the baseline
    ax2.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1)

    # Set up the second subplot (ROC curve)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.015])
    ax2.set_xlabel('False Positive Rate', fontsize=12)

    # Move Y-axis label and ticks to the right for the ROC plot
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.grid(True, linestyle=':', lw=0.5, color='#D7E2F9')

    # Create the legend in reverse order for ax2
    ax2.legend(reversed(lines), reversed(labels), loc="lower right", fontsize=10)

    # Final layout adjustments to fit everything nicely
    plt.tight_layout(pad=0.25)  # Add some padding to avoid crowding

    # Save the plot as both PNG and PDF
    plt.savefig('cumulative_accept_prob_auroc_.png', dpi=300)
    plt.savefig('cumulative_accept_prob_auroc_.pdf', format='pdf', dpi=300)
    plt.show()

# Example usage
# plot_combined_graphs('accept_prob.csv', 'accept_probs.csv')
plot_combined_graphs('ece.csv', 'auroc.csv')
