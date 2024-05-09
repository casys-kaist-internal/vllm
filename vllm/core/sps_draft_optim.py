from abc import abstractmethod
from typing import Dict, List

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

PLOT_HEATMAP = True

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
        self.draft_history = {"beta_ema": [], "draft_prob": [], "accept_prob": []}

    def update_draft_sizes(self, seq_group_list: List[SequenceGroup]):
        for seq_group in seq_group_list:
            seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
            self.update_draft_size_seq(seq)

    def update_draft_size_seq(self, seq: Sequence):
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

    def _init_predictor(self) -> LinearRegression:
        # predictor = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True)
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=self.predictor_degree, include_bias=False)),
            ('linear', LinearRegression(fit_intercept=True))
        ])

        linear_model = pipeline.named_steps['linear']
        linear_model.coef_ = np.array([0.85221723, 0.67564867, -0.29379885, -0.38084159, 0.24302686])
        linear_model.intercept_ = -0.015180676837696971
        
        return pipeline

    def _check_early_stop(self, seq: Sequence) -> bool:
        # Check if the sequence should be stopped early
        # Get probability of last draft token
        last_draft_token_id = seq.data.get_last_draft_token_id()
        draft_prob = seq.data.get_draft_probs()[-1][last_draft_token_id].item()
        beta_ema = seq.get_beta_ema()

        predicted_accept_prob = self._predict_accept_prob(beta_ema, draft_prob)

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
        # Check if the predictor is not fitted; return a default value
        try:
            # Prepare the input features
            X = np.array([[beta_ema, draft_prob]])
            
            # Apply polynomial feature transformation
            X_poly = self.predictor.named_steps['poly'].transform(X)
            
            # Predict acceptance probability using the linear model
            accept_prob = self.predictor.named_steps['linear'].predict(X_poly)
            
            # Clip the result to ensure it's within [0, 1]
            return np.clip(accept_prob[0], 0, 1)
        
        except NotFittedError:
            # Return the default acceptance probability
            return 1

    def _train_predictor(self):
        start_time = time.monotonic()
        binned_df = self._get_binned_draft_history_df()

        # Prepare data for regression
        X = binned_df[["beta_ema_binned_code", "draft_prob_binned_code"]].values
        y = binned_df["accept_prob"].values

        # Set the grid dimensions
        grid_extent = [0, 1, 0, 1]

        if PLOT_HEATMAP:
            try:
                # Predict the "before" state
                X_predict = np.array([[beta, draft] for beta in np.linspace(0, 1, self.num_bins) for draft in np.linspace(0, 1, self.num_bins)])
                y_initial_predict = self.predictor.predict(X_predict).reshape(self.num_bins, self.num_bins)
            except NotFittedError:
                y_initial_predict = np.zeros((self.num_bins, self.num_bins))

        # Linear regression training
        self.predictor.fit(X, y)
        end_time = time.monotonic()
        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"Trained predictor in {elapsed_time_ms:.2f} ms")

        if PLOT_HEATMAP:
            # Create a matrix for real acceptance probability heatmap
            pivot_df = binned_df.pivot(index="beta_ema", columns="draft_prob", values="accept_prob")

            # Predict the "after" state
            y_predict = self.predictor.predict(X_predict).reshape(self.num_bins, self.num_bins)

            # Create the three subplots: Before, Real Data, After
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

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
            .agg({"accept_prob": self.agg_type})
            .dropna()
        )
        binned_df.reset_index(inplace=True)

        # Convert categories to codes to be used in regression
        binned_df["beta_ema_binned_code"] = binned_df["beta_ema"].cat.codes
        binned_df["draft_prob_binned_code"] = binned_df["draft_prob"].cat.codes

        return binned_df

    def reset(self):
        self.predictor = self._init_predictor()
        self._init_draft_history()
