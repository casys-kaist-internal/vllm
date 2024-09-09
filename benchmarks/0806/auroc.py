import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects


def plot_roc_curve_from_csv(file_path):
    temperatures = []
    actual_accept_probs = []
    predicted_accept_probs = []

    # Manually read and parse the CSV file
    with open(file_path, 'r') as file:
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

    # Define the color palette based on your provided colors
    colors = ['#1B345F', '#3864B9', '#88BCFF', '#D7E2F9']

    # Plot ROC curve for each unique temperature in reverse order
    unique_temps = np.unique(temperatures)
    lines = []  # Store line handles to create the legend
    labels = []  # Store labels for the legend
    plt.figure(figsize=(5, 2.5))  # Adjusted for 2-column paper

    for i, temp in reversed(list(enumerate(unique_temps))):
        mask = temperatures == temp
        actual_accept_prob_temp = actual_accept_probs[mask]
        predicted_accept_prob_temp = predicted_accept_probs[mask]

        # Binarize the actual accept_prob with a threshold of 0.5
        actual_accept_prob_binary = (
            actual_accept_prob_temp >= 0.5).astype(int)

        # Calculate the AUROC
        auroc = roc_auc_score(actual_accept_prob_binary,
                              predicted_accept_prob_temp)

        # Calculate the ROC curve
        fpr, tpr, _ = roc_curve(actual_accept_prob_binary,
                                predicted_accept_prob_temp)

        # Plot the ROC curve with the specified color and black border
        line, = plt.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                         label=f'Temp {temp:.2f} (AUROC = {auroc:.2f})')

        # Add black border around the line
        line.set_path_effects(
            [PathEffects.withStroke(linewidth=3, foreground='black')])

        # Store line and label for the legend
        lines.append(line)
        labels.append(f'Temp {temp:.2f} (AUROC = {auroc:.2f})')

    # Plotting the baseline
    plt.plot([0, 1], [0, 1], color='black',
             linestyle='--', lw=1)  # Black baseline

    # Setting up plot details
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.015])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)

    # Create the legend with the original order (reverse the lists)
    plt.legend(reversed(lines), reversed(labels),
               loc="lower right", fontsize=10)

    plt.grid(True, linestyle=':', lw=0.5, color='#D7E2F9')  # Lighter grid
    plt.tight_layout()  # Ensure everything fits in the figure
    plt.savefig('roc_curve_all_temperatures.png', dpi=300)
    plt.show()


# Example usage
plot_roc_curve_from_csv('accept_probs.csv')
