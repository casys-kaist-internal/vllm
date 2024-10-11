import numpy as np
from typing import List

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from vllm.sequence import Sequence, SequenceGroup
from vllm.utils import nvtx_range


class SpecDecodeAcceptProbPredictor:
    """
    Predicts the `accept_prob` based on input features:
    - `temperature`
    - `mean_accept_prob` (inter-request variability)
    - `draft_prob` (intra-request variability)
    """

    HISTORY_SIZE = 10000  # Constant for the history size
    PLOT = False  # Plot the ROC curve
    MIN_OUTPUT_CNT = 100

    def __init__(self):
        self.history = {
            "draft_probs": np.zeros(self.HISTORY_SIZE),
            "accept_probs": np.zeros(self.HISTORY_SIZE),
        }
        self.position = 0

    def is_trained(self):
        return hasattr(self, 'regression_model')

    @nvtx_range("update_history")
    def update_history(self, seq_group_list: List[SequenceGroup]):
        for seq_group in seq_group_list:
            seq = seq_group.get_seqs()[0]

            if self.is_trained():
                seq.sampled_draft_probs.clear()
                continue

            if seq.get_output_len() < self.MIN_OUTPUT_CNT:
                seq.sampled_draft_probs.clear()
                continue

            sampled_draft_probs = np.array(seq.sampled_draft_probs)
            accept_probs = np.array(seq.accept_probs)
            seq.sampled_draft_probs.clear()

            assert len(sampled_draft_probs) == len(accept_probs)

            # Calculate the new end position
            end_position = self.position + len(sampled_draft_probs)

            # Insert data into the history arrays, wrapping around and dropping overflow
            if end_position <= self.HISTORY_SIZE:
                # No overflow, direct assignment
                self.history["draft_probs"][self.position:end_position] = sampled_draft_probs
                self.history["accept_probs"][self.position:end_position] = accept_probs
            else:
                # Handle wrapping around
                overflow = end_position - self.HISTORY_SIZE
                valid_size = len(sampled_draft_probs) - overflow

                self.history["draft_probs"][self.position:
                                            self.HISTORY_SIZE] = sampled_draft_probs[:valid_size]
                self.history["accept_probs"][self.position:
                                             self.HISTORY_SIZE] = accept_probs[:valid_size]

                self.train_linear_regression()

            self.position = end_position % self.HISTORY_SIZE

    def plot_draft_vs_accept(self):
        # Bin the draft_probs
        num_bins = 20
        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(
            self.history["draft_probs"][:self.position], bins) - 1

        # Calculate the mean or median of accept_probs for each bin
        mean_accept_probs = np.array([self.history["accept_probs"][:self.position][bin_indices == i].mean()
                                      for i in range(num_bins)])
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Calculate the standard deviation of accept_probs for each bin
        std_accept_probs = np.array([self.history["accept_probs"][:self.position][bin_indices == i].std()
                                     for i in range(num_bins)])

        # Plot the binned draft_probs against mean accept_probs
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, mean_accept_probs, '-o',
                 color='blue', label='Mean Accept Prob')

        # Plot the standard deviation of accept_probs for each bin
        plt.errorbar(bin_centers, mean_accept_probs, yerr=std_accept_probs,
                     fmt='o', color='blue', label='Mean Accept Prob')

        plt.xlabel('Draft Probability (Binned)')
        plt.ylabel('Mean Acceptance Probability')
        plt.title('Draft Probability vs. Acceptance Probability')
        plt.grid(True)
        plt.legend()
        plt.savefig('draft_vs_accept.png', dpi=300)
        plt.close()

    def train_linear_regression(self):
        # self.plot_draft_vs_accept()

        # Collect data for the current temperature
        # Reshaping to 2D array
        X = self.history["draft_probs"][:self.HISTORY_SIZE].reshape(-1, 1)
        y = self.history["accept_probs"][:self.HISTORY_SIZE]

        assert X.shape[0] == y.size

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Train the model for this temperature
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on test data and calculate mean squared error
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Equation string
        equation = f"y = {model.intercept_:.2f}"
        for i, coef in enumerate(model.coef_):
            equation += f" + {coef:.2f} * x{i}"
        print(
            f"[INFO] Linear regression: {equation} MSE: {mse}")

        # Store the trained model and equation for the current temperature
        self.regression_model = model
        self.regression_equation = equation

        # Optionally, plot model predictions for the trained models
        if self.PLOT:
            self.plot_auroc()

    def plot_auroc(self):
        if not hasattr(self, 'regression_model'):
            raise ValueError("Regression model is not trained yet.")

        # Create a figure with a single subplot
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

        # Define a threshold for binarization
        threshold = 0.5
        binarizer = Binarizer(threshold=threshold)

        # Collect draft probabilities and acceptance probabilities
        draft_probs = self.history["draft_probs"][:self.HISTORY_SIZE]
        accept_probs = self.history["accept_probs"][:self.HISTORY_SIZE]

        # Binarize the accept_probs based on the threshold
        binary_accept_probs = binarizer.fit_transform(
            accept_probs.reshape(-1, 1)).ravel()

        # Prepare the data for prediction
        X = draft_probs.reshape(-1, 1)

        # Predict the probabilities using the trained regression model
        y_pred = self.regression_model.predict(X)

        # Compute the ROC curve and AUC
        fpr, tpr, _ = roc_curve(binary_accept_probs, y_pred)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}', color='blue')

        # Plot the diagonal line for reference
        ax.plot([0, 1], [0, 1], 'r--', lw=1, color='gray')

        # Set plot titles and labels
        ax.set_title('ROC Curve', fontsize=16)
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.legend(loc="lower right")

        # Add the regression model equation as text on the plot
        equation = self.regression_equation
        ax.text(0.05, 0.3, equation, fontsize=12, transform=ax.transAxes,
                color='blue', bbox=dict(facecolor='white', alpha=0.5))

        # Save the figure
        plt.savefig('auroc.png', dpi=300)
        plt.close()

    def get_seq_features(self, seq_group: SequenceGroup) -> float:
        """
        Extracts the last draft probability from the first sequence in the sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing sequences.

        Returns:
            float: The last draft probability.
        """
        # Directly access the first sequence and the last draft probability
        seq = seq_group.get_seqs()[0]
        return seq.sampled_draft_probs[-1]

    # def predict_accept_probs(self, seq_group_list: List[SequenceGroup]):
    #     # Check if the regression model is trained
    #     if not hasattr(self, 'regression_model'):
    #         raise ValueError("Regression model is not trained yet.")

    #     # Collect features for all sequence groups
    #     draft_probs = np.array([self.get_seq_features(seq_group)
    #                             for seq_group in seq_group_list])

    #     # Predict using the trained regression model
    #     predictions = self.regression_model.predict(draft_probs)

    #     # Clip the predictions to ensure they are within [0, 1]
    #     predictions = np.clip(predictions, 0, 1)

    #     # Update each sequence group with the predicted acceptance probabilities
    #     for i, seq_group in enumerate(seq_group_list):
    #         seq = seq_group.get_seqs()[0]
    #         predicted_accept_prob = float(predictions[i])
    #         last_predicted_accept_prob = seq.predicted_cumulated_accept_probs[-1] if len(
    #             seq.predicted_cumulated_accept_probs) > 0 else 1
    #         # last_predicted_accept_prob = 1 # debug
    #         cumulative_predicted_accept_prob = last_predicted_accept_prob * predicted_accept_prob
    #         seq.predicted_cumulated_accept_probs.append(
    #             cumulative_predicted_accept_prob)

    def predict_accept_probs(self, seq_group_list: List[SequenceGroup]):
        """
        Predicts acceptance probabilities for a list of sequence groups and updates each sequence.

        Args:
            seq_group_list (List[SequenceGroup]): List of sequence groups to process.
        """
        if not hasattr(self, 'regression_model'):
            raise ValueError("Regression model is not trained yet.")

        # Extract sequences from sequence groups
        seqs = [seq_group.get_seqs()[0] for seq_group in seq_group_list]

        # Collect draft probabilities using a list comprehension
        draft_probs = np.array([seq.sampled_draft_probs[-1] for seq in seqs], dtype=np.float32)

        # Reshape draft_probs to match the expected input shape for the regression model
        draft_probs = draft_probs.reshape(-1, 1)

        # Predict using the trained regression model
        predictions = self.regression_model.predict(draft_probs)

        # Clip the predictions to ensure they are within [0, 1]
        predictions = np.clip(predictions, 0, 1)

        # Collect last predicted acceptance probabilities
        last_predicted_accept_probs = np.array([
            seq.predicted_cumulated_accept_probs[-1] if seq.predicted_cumulated_accept_probs else 1.0
            for seq in seqs
        ], dtype=np.float32)

        # Compute cumulative predicted acceptance probabilities
        cumulative_predicted_accept_probs = last_predicted_accept_probs * predictions

        # Update each sequence with the new cumulative predicted acceptance probability
        for seq, cumulative_prob in zip(seqs, cumulative_predicted_accept_probs):
            seq.predicted_cumulated_accept_probs.append(float(cumulative_prob))