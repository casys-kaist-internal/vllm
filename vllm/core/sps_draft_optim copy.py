from abc import abstractmethod
from typing import List

import matplotlib.pyplot as plt
import time
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

from vllm.sequence import Sequence, SequenceGroup, SequenceStatus

PLOT_HEATMAP = False
RETRAIN = False

# print the polynomial function
def create_polynomial_equation(model, feature_names):
    # Extract coefficients and intercept from the model
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Construct equation terms with coefficients
    terms = [f"{coeff}*{name}" for coeff, name in zip(coefficients, feature_names) if coeff != 0]
    
    # Add intercept
    equation = " + ".join(terms) + f" + {intercept:.2f}"
    
    return equation

class DraftSizeOptimizer:
    @abstractmethod
    def update_draft_sizes(self, seq_group_list: List[SequenceGroup]):
        raise NotImplementedError()

    @abstractmethod
    def update_draft_size_seq(self, seq: Sequence):
        raise NotImplementedError()


class BetaEMADraftSizeOptimizer(DraftSizeOptimizer):
    def __init__(self):
        self.retrain_index = 0
        self.retrain_period = 100000
        self.history_size = 1000000
        self.num_bins = 20
        self.predictor_degree = 2
        self.agg_type = "median"
        self.predictor = self._init_predictor()
        self.create_lookup_table()  # Create the lookup table after the predictor is initialized
        self.draft_history = {"beta_ema": [], "draft_prob": [], "accept_prob": []}

    def update_draft_sizes(self, seq_group_list: List[SequenceGroup]):
        for seq_group in seq_group_list:
            seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
            self.update_draft_size_seq(seq)

    def update_draft_size_seq(self, seq: Sequence):
        if RETRAIN:
            # indicator for retraining the predictor
            self.retrain_index += 1
            if self.retrain_index >= self.retrain_period:
                self.retrain_index = 0
                self._train_predictor()
                self._drop_draft_history()

            self._update_drafted_accepted_df(seq)

        if self._check_early_stop(seq):
            seq.draft_size = seq.get_draft_len()

    def _drop_draft_history(self):
        # drop the first element to keep the history size
        if len(self.draft_history["beta_ema"]) > self.history_size:
            self.draft_history["beta_ema"] = self.draft_history["beta_ema"][-self.history_size :]
            self.draft_history["draft_prob"] = self.draft_history["draft_prob"][-self.history_size :]
            self.draft_history["accept_prob"] = self.draft_history["accept_prob"][-self.history_size :]

    def _init_draft_history(self):
        self.draft_history = {"beta_ema": [], "draft_prob": [], "accept_prob": []}

    def _init_predictor(self) -> Pipeline:
        poly_features = PolynomialFeatures(
            degree=self.predictor_degree, include_bias=True
        )
        # fit with dummy
        dummy_df = pd.DataFrame([[0, 0]], columns=["beta_ema", "draft_prob"])
        poly_features.fit(dummy_df) 

        # predictor = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True)
        predictor = LinearRegression(fit_intercept=True)
        # # Set initial parameters.
        predictor.coef_ = np.array(
            [ 0., 0.91908315, 1.62168467, -0.2826796, -0.7090701, -0.64855703 ]
        )
        predictor.intercept_ = 0.0014159779056236443

        # predictor.coef_ = np.array(
        #     [0, 0.240050765,1.42375935,-3.22546415,-0.03299826,-0.78081028,1.66473911,-0.04958285,-0.6562083,0.41092696]
        # )
        # predictor.intercept_ = -0.21273702585116017

        # plot initial predictor
        if PLOT_HEATMAP:
            # Create a grid of (beta, draft) pairs using a Pandas DataFrame with feature names
            beta_values = np.linspace(0, 1, self.num_bins)
            draft_values = np.linspace(0, 1, self.num_bins)

            # Generate pairs as a DataFrame
            X_predict = pd.DataFrame(
                [[beta, draft] for beta in beta_values for draft in draft_values],
                columns=["beta_ema", "draft_prob"]
            )
            y_initial_predict = predictor.predict(poly_features.transform(X_predict)).reshape(self.num_bins, self.num_bins)

            # Create the plot
            plt.imshow(y_initial_predict, cmap='Accent', interpolation='nearest', origin='lower', extent=[0, 1, 0, 1])
            plt.colorbar()
            plt.title('Initial Acceptance Probability Heatmap')
            plt.xlabel('Draft Probability')
            plt.ylabel('Beta EMA')
            os.makedirs('heatmap', exist_ok=True)
            plt.savefig('heatmap/initial_accept_prob.png')

        return Pipeline([("poly", poly_features), ("linear", predictor)])

    def create_lookup_table(self):
        # Create a grid of (beta, draft) pairs using a Pandas DataFrame
        beta_values = np.linspace(0, 1, self.num_bins + 1)
        draft_values = np.linspace(0, 1, self.num_bins + 1)
        
        # Midpoints of bins for prediction purposes
        beta_midpoints = (beta_values[:-1] + beta_values[1:]) / 2
        draft_midpoints = (draft_values[:-1] + draft_values[1:]) / 2
        
        # Generate all combinations of beta and draft midpoints
        X_predict = pd.DataFrame(
            [[beta, draft] for beta in beta_midpoints for draft in draft_midpoints],
            columns=["beta_ema", "draft_prob"]
        )
        
        # Transform data using polynomial features if necessary
        if hasattr(self, 'predictor') and 'poly' in self.predictor.named_steps:
            X_poly = self.predictor.named_steps['poly'].transform(X_predict)
            accept_probs = self.predictor.named_steps['linear'].predict(X_poly)
        else:
            accept_probs = self.predictor.predict(X_predict)

        # Reshape to the number of bins for easy lookup
        accept_probs = accept_probs.reshape(self.num_bins, self.num_bins)
        
        # Store the accept_probs as a lookup table
        self.lookup_table = accept_probs

    def _predict_accept_prob_from_lookup(self, beta_ema: float, draft_prob: float) -> float:
        # Find the index for beta_ema
        beta_idx = int(beta_ema * self.num_bins)
        draft_idx = int(draft_prob * self.num_bins)
        
        # Make sure indices are within the range
        beta_idx = min(beta_idx, self.num_bins - 1)
        draft_idx = min(draft_idx, self.num_bins - 1)
        
        # Retrieve the predicted accept_prob from the lookup table
        accept_prob = self.lookup_table[beta_idx, draft_idx]
        return accept_prob
        
    def _check_early_stop(self, seq: Sequence) -> bool:
        # Check if the sequence should be stopped early
        # Get probability of last draft token
        last_draft_token_id = seq.data.get_last_draft_token_id()
        draft_prob = seq.data.get_draft_probs()[-1][last_draft_token_id].item()
        beta_ema = seq.get_beta_ema()

        if RETRAIN:
            predicted_accept_prob = self._predict_accept_prob(beta_ema, draft_prob)
        else:
            predicted_accept_prob = self._predict_accept_prob_from_lookup(beta_ema, draft_prob)
        
        seq.cumulative_accept_prob *= predicted_accept_prob
        random_accept_prob = np.random.uniform(0, 1)
        if seq.cumulative_accept_prob < random_accept_prob:
            seq.cumulative_accept_prob = 1
            return True
        
        return False

    def _update_drafted_accepted_df(self, seq: Sequence):
        beta_emas, draft_probs, accept_probs = seq.get_new_draft_history()
        assert len(beta_emas) == len(draft_probs) == len(accept_probs)
        if len(beta_emas) == 0:
            return

        self.draft_history["beta_ema"].extend(beta_emas)
        self.draft_history["draft_prob"].extend(draft_probs)
        self.draft_history["accept_prob"].extend(accept_probs)

    def _predict_accept_prob(self, beta_ema: float, draft_prob: float) -> float:
        # Prepare the input data with appropriate feature names
        X = pd.DataFrame([[beta_ema, draft_prob]], columns=["beta_ema", "draft_prob"])
        
        # Apply polynomial feature transformation with the proper feature names
        X_poly = self.predictor.named_steps['poly'].transform(X)
        
        # Predict acceptance probability using the linear model
        accept_prob = self.predictor.named_steps['linear'].predict(X_poly)
        
        # Clip the result to ensure it's within [0, 1]
        return np.clip(accept_prob[0], 0, 1)
 

    def _train_predictor(self):
        start_time = time.monotonic()
        binned_df = self._get_binned_draft_history_df()
        binning_time = time.monotonic()

        X = binned_df[['beta_ema', 'draft_prob']].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        y = binned_df['accept_prob']

        # Check if the input data is empty
        if X.shape[0] == 0 or y.shape[0] == 0:
            print("No data available for training the predictor. Skipping this training step.")
            return

        # Set the grid dimensions
        grid_extent = [0, 1, 0, 1]

        if PLOT_HEATMAP:
            # Predict the "before" state
            # Create a grid of (beta, draft) pairs using a Pandas DataFrame with feature names
            beta_values = np.linspace(0, 1, self.num_bins)
            draft_values = np.linspace(0, 1, self.num_bins)

            # Generate pairs as a DataFrame
            X_predict = pd.DataFrame(
                [[beta, draft] for beta in beta_values for draft in draft_values],
                columns=["beta_ema", "draft_prob"]
            )
            y_initial_predict = self.predictor.predict(X_predict).reshape(self.num_bins, self.num_bins)
        
        # Check if the input data is empty
        if X.shape[0] == 0 or y.shape[0] == 0:
            print("No data available for training the predictor. Skipping this training step.")
            return

        # Linear regression training
        self.predictor.fit(X, y)
        end_time = time.monotonic()
        elapsed_time_ms = (end_time - start_time) * 1000
        binning_time_ms = (binning_time - start_time) * 1000
        print(f"Trained predictor in {elapsed_time_ms:.2f} ms")
        print(f"Binning time: {binning_time_ms:.2f} ms")
        print(f"{self.predictor.named_steps['linear'].coef_}")
        print(f"{self.predictor.named_steps['linear'].intercept_}")

        if PLOT_HEATMAP:
            # Create a matrix for real acceptance probability heatmap
            pivot_df = binned_df.pivot(index="beta_ema", columns="draft_prob", values="accept_prob")

            # Predict the "after" state
            y_predict = self.predictor.predict(X_predict).reshape(self.num_bins, self.num_bins)

            # Create the four subplots: Before, Real Data, After, Token Distribution
            fig, axs = plt.subplots(1, 3, figsize=(24, 6), sharey=True)

            # Plot "Before" (Initial Model Prediction)
            axs[0].imshow(y_initial_predict, cmap='Accent', interpolation='nearest', origin='lower', extent=grid_extent)
            axs[0].set_title('Before (Initial Model Prediction)')
            axs[0].set_xlabel('Draft Probability')
            axs[0].set_ylabel('Beta EMA')

            # Plot "Real Data" (Acceptance Probability Heatmap)
            axs[1].imshow(pivot_df, cmap='Accent', interpolation='nearest', origin='lower', extent=grid_extent)
            axs[1].set_title('Real Acceptance Probability Heatmap')
            axs[1].set_xlabel('Draft Probability')

            # Plot "After" (Final Model Prediction)
            axs[2].imshow(y_predict, cmap='Accent', interpolation='nearest', origin='lower', extent=grid_extent)
            axs[2].set_title('After (Final Model Prediction)')
            axs[2].set_xlabel('Draft Probability')

            # Retrieve the final coefficients and intercept
            feature_names = self.predictor.named_steps['poly'].get_feature_names_out(['beta_ema', 'draft_prob'])
            linear_model =  self.predictor.named_steps['linear']

            polynomial_function_string = create_polynomial_equation(linear_model, feature_names)

            # Add the coefficients and intercept as a text annotation
            fig.text(0.5, 0.01, polynomial_function_string, ha='center', va='bottom', fontsize=10)

            # Adjust the layout and add a colorbar
            fig.colorbar(axs[1].images[0], ax=axs, orientation='vertical', pad=0.03)
            fig.suptitle('Before vs. Real vs. After Acceptance Probability Heatmaps')

            # Create the directory for saving the plots if it doesn't exist
            os.makedirs('heatmap', exist_ok=True)
            current_time = time.strftime("%Y%m%d-%H%M%S")
            plt.savefig(f'heatmap/accept_prob_{current_time}.png')

            # Define the bins
            beta_ema_bins = np.linspace(0, 1, self.num_bins + 1)
            draft_prob_bins = np.linspace(0, 1, self.num_bins + 1)
            
            # Create the figure with subplots
            fig, axes = plt.subplots(self.num_bins, self.num_bins, figsize=(24, 24), sharex=True, sharey=True)

            # Convert categorical to numeric if they are not meant to be categorical
            binned_df['beta_ema'] = pd.to_numeric(binned_df['beta_ema'], errors='coerce')
            binned_df['draft_prob'] = pd.to_numeric(binned_df['draft_prob'], errors='coerce')

            # Iterate over each bin combination
            for i in range(self.num_bins):
                for j in range(self.num_bins):
                    # Select data in the current bin
                    bin_filter = (
                        (binned_df['beta_ema'] >= beta_ema_bins[i]) & (binned_df['beta_ema'] < beta_ema_bins[i + 1]) &
                        (binned_df['draft_prob'] >= draft_prob_bins[j]) & (binned_df['draft_prob'] < draft_prob_bins[j + 1])
                    )
                    bin_data = binned_df[bin_filter]
                    if bin_data['accept_prob'].count() > 10:
                        # Plot the distribution of 'accept_prob' in this bin
                        ax = axes[i, j]
                        ax.hist(bin_data['accept_prob'], bins=10, color='blue', alpha=0.7, edgecolor='black')
                        ax.set_title(f'β_ema: {beta_ema_bins[i]:.2f}-{beta_ema_bins[i+1]:.2f}, Draft Prob: {draft_prob_bins[j]:.2f}-{draft_prob_bins[j+1]:.2f}', fontsize=8)
                        ax.set_xlabel('accept_prob', fontsize=6)
                        ax.set_ylabel('Frequency', fontsize=6)
                        ax.tick_params(axis='both', which='major', labelsize=6)

            # Adjust the layout and set the figure title
            fig.suptitle('Distribution of accept_prob for All Beta EMA & Draft Probability Bins', fontsize=18)
            plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])  # Adjust layout to leave space for the title
            plt.savefig(f'heatmap/accept_prob_distribution_{current_time}.png')


    def _get_binned_draft_history_df(self) -> pd.DataFrame:
        draft_accepted_df = pd.DataFrame(self.draft_history)

        # Define bins
        beta_ema_bins = np.linspace(0, 1, self.num_bins + 1)
        draft_prob_bins = np.linspace(0, 1, self.num_bins + 1)

        # Binning the data
        draft_accepted_df["beta_ema"] = pd.cut(
            draft_accepted_df["beta_ema"],
            bins=beta_ema_bins,
            labels=np.linspace(0, 1, self.num_bins, endpoint=False),
        )
        draft_accepted_df["draft_prob"] = pd.cut(
            draft_accepted_df["draft_prob"],
            bins=draft_prob_bins,
            labels=np.linspace(0, 1, self.num_bins, endpoint=False),
        )

        # clip accept_prob [0, 1]
        draft_accepted_df["accept_prob"] = draft_accepted_df["accept_prob"].clip(0, 1)

        # Group and aggregate data
        binned_df = (
            draft_accepted_df.groupby(["beta_ema", "draft_prob"])
            .agg(accept_prob=pd.NamedAgg(
                    column="accept_prob", aggfunc=self.agg_type
                ))
            .dropna()
        )
        binned_df.reset_index(inplace=True)

        return binned_df

    def reset(self):
        self.predictor = self._init_predictor()
        self._init_draft_history()
