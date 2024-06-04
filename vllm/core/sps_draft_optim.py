import datetime
import os
from abc import abstractmethod
from threading import Thread, Semaphore
from typing import List, Union

import matplotlib.pyplot as plt
import time
import os
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from torch.cuda import nvtx

from vllm.config import SpSConfig
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus

PLOT_HEATMAP = False
DEFER_EXIT = False

def defer_exit(delay: float):
    time.sleep(delay)
    os._exit(0)

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
    def update_draft_size_seq(self, seq_list: List[Sequence]):
        raise NotImplementedError()


class BetaEMADraftSizeOptimizer(DraftSizeOptimizer):
    def __init__(self, sps_config: SpSConfig):
        self.sps_config = sps_config
        self.retrain_index = 0
        self.retrain_period = 10000
        self.history_size = 100000
        self.num_bins = 20
        self.predictor = self._init_predictor()
        self.lookup_table = self._init_lookup_table()
        self.draft_history = {"beta_ema": [], "draft_prob": [], "accept_prob": []}
        self.retrain = True
        self.use_lookup_table = self.sps_config.use_lookup_table

        if sps_config.use_async_training:
            self.worker = Thread(target=self._train_predictor_async)
            self.worker_sema = Semaphore(0)
            self.worker.start()
        else:
            self.worker = None
            self.worker_sema = None

        self.initialize("/home/noppanat/workspace/vllm/benchmarks/predictor/lookup_table.csv") # TODO(noppanat): remove this

    # I think we should put this back to SpSConfig because SpSScheduler might need this. 
    def get_tile_size(self):
        if self.sps_config.use_tile_constraint == "none":
            return 100000
        elif self.sps_config.use_tile_constraint == "cut-128":
            return 128
        else:
            raise NotImplementedError(f"Unsupported tile constraint: {self.sps_config.use_tile_constraint}")

    def update_draft_size_seq(self, 
                              running_seq_list: List[Sequence],
                              seq_list: List[Sequence]):
        # Extract features to predict the acceptance probability
        X = [self._get_seq_features(seq) for seq in running_seq_list]

        # Predict
        nvtx.range_push(f"predict {X[0][0]} {X[0][1]}")
        if self.use_lookup_table:
            accept_probs = np.clip(self._lookup_predict(X), 0, 1)
        else:
            accept_probs = np.clip(self.predictor.predict(X), 0, 1)
        nvtx.range_pop()

        nvtx.range_push(f"early_exit_policy {accept_probs}")
        # Early exit policy
        num_tokens_to_generate = 0
        # if self.sps_config.use_dynamic_draft_size:
        #     random_accept_probs = np.random.uniform(0, 1, len(accept_probs)) 
        # else:
        #     random_accept_probs = np.zeros(len(accept_probs))

        # for seq, accept_prob, random_accept_prob in zip(running_seq_list, accept_probs, random_accept_probs):
        #     seq.cumulative_accept_prob *= accept_prob
        #     if seq.cumulative_accept_prob < random_accept_prob: # draft load imbalance & fill 
        #         seq.draft_size = seq.get_draft_len()
        #         seq.cumulative_accept_prob = 1
        #     else:
        #         num_tokens_to_generate += 1
        if self.sps_config.use_dynamic_draft_size:
            num_tokens_to_generate = 0
            for seq, accept_prob in zip(running_seq_list, accept_probs):
                if accept_prob < seq.exit_threshold:
                    seq.draft_size = seq.get_draft_len()
                else:
                    num_tokens_to_generate += 1

        else:
            num_tokens_to_generate = len(running_seq_list)
        nvtx.range_pop()

        # Tile constraint policy
        num_tokens_to_target = 0
        for seq in seq_list:
            num_tokens_to_target += (seq.get_draft_len() + 1)

        if num_tokens_to_generate + num_tokens_to_target <= self.get_tile_size():
            # Maybe fill 
            return
        else:
            nvtx.range_push("tile_constraint_policy")
            if self.sps_config.use_dynamic_draft_size:
                num_tokens_to_cut = num_tokens_to_generate + num_tokens_to_target - self.get_tile_size()
                running_seq_list.sort(key=lambda x: x.cumulative_accept_prob)
                for seq in running_seq_list:
                    if seq.get_draft_len() == seq.draft_size:
                        continue
                    
                    num_tokens_to_cut -= 1
                    seq.draft_size = seq.get_draft_len()
                    # seq.cumulative_accept_prob = 1
                    
                    if num_tokens_to_cut == 0:
                        break
            else:
                for seq in running_seq_list:
                    if seq.get_draft_len() == seq.draft_size:
                        continue
                    seq.draft_size = seq.get_draft_len()
                    # seq.cumulative_accept_prob = 1
            nvtx.range_pop()

    def _get_seq_features(self, seq: Sequence) -> List[float]:
        # indicator for retraining the predictor
        self.retrain_index += 1
        if self.retrain and self.retrain_index >= self.retrain_period:
            self.retrain_index = 0
            print("[debug] (noppanat) time to retrain:", str(datetime.datetime.now()), flush=True)
            if self.worker is None:
                self._train_predictor_sync()
            else:
                self.worker_sema.release()

        self._update_drafted_accepted_df(seq)

        last_draft_token_id = seq.data.get_last_draft_token_id()
        draft_prob = seq.data.get_draft_probs()[-1][last_draft_token_id].item()
        beta_ema = seq.get_beta_ema()
    
        return [beta_ema, draft_prob]

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
            degree=self.sps_config.predictor_degree, include_bias=True
        )
        # fit with dummy
        dummy_df = [[0, 0]]
        poly_features.fit(dummy_df)

        # predictor = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True)
        predictor = LinearRegression(fit_intercept=True)
        # # Set initial parameters.
        # predictor.coef_ = np.array(
        #     [0.85221723, 0.67564867, -0.29379885, -0.38084159, 0.24302686]
        # )
        # predictor.intercept_ = -0.015180676837696971

        predictor.coef_ = np.array(
            [ 0., 0.93413575, 1.22473454, -0.32001846, -0.71799723, -1.63476777, -0.24324052, 0.4377291, -0.07319985, 1.27301201] 
        )
        predictor.intercept_ = 0.09655343164046182

        return Pipeline([("poly", poly_features), ("linear", predictor)])
    
    def _init_lookup_table(self):
        # Initialize an empty lookup table with the specified number of bins
        lookup_table = np.zeros((self.num_bins, self.num_bins))
        return lookup_table
    
    def _lookup_predict(self, X):
        # Predict using the lookup table
        X = np.array(X)
        beta_indices = (X[:, 0] * (self.num_bins - 1)).astype(int)
        draft_indices = (X[:, 1] * (self.num_bins - 1)).astype(int)
        accept_probs = self.lookup_table[beta_indices, draft_indices]
        return accept_probs
    
    def _update_drafted_accepted_df(self, seq: Sequence):
        beta_emas, draft_probs, accept_probs = seq.get_new_draft_history()
        assert len(beta_emas) == len(draft_probs) == len(accept_probs)
        if len(beta_emas) == 0:
            return

        self.draft_history["beta_ema"].extend(beta_emas)
        self.draft_history["draft_prob"].extend(draft_probs)
        self.draft_history["accept_prob"].extend(accept_probs)

    def _train_predictor_sync(self):
        if self.use_lookup_table:
            predictor = self.lookup_table
        else:
            predictor = self.predictor
        
        self._train_predictor(predictor)

    def _train_predictor_async(self):
        while self.worker_sema.acquire():
            # Make a copy of the draft optimizer states.
            if self.use_lookup_table:
                predictor = np.zeros_like(self.lookup_table)
            else:
                predictor = clone(self.predictor)
            
            self._train_predictor(predictor)

            # Update the predictor
            if self.use_lookup_table:
                self.lookup_table = predictor
            else:
                self.predictor = predictor

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
                    column="accept_prob", aggfunc=self.sps_config.predictor_agg_type
                ))
            .dropna()
        )
        binned_df.reset_index(inplace=True)

        return binned_df

    def initialize(self, path):
        self.path = path
        # if use_lookup_table then path should have lookup_table.csv
        if self.use_lookup_table:
            assert path.endswith("lookup_table.csv"), "Lookup table path should end with 'lookup_table.csv'"

        if os.path.exists(path):
            if self.use_lookup_table:
                # Load the lookup table
                self.lookup_table = pd.read_csv(path, index_col=0).values
                print("Lookup table loaded successfully.")
            else:
                # Read the CSV file
                data = pd.read_csv(path)

                # Assuming the CSV has columns 'coef' and 'intercept'
                coef = data['coef'].values
                intercept = data['intercept'].values[0]

                # Initialize the predictor
                pipeline = self._init_predictor()
                predictor = pipeline.named_steps['linear']
                
                # Set the loaded coefficients and intercept
                predictor.coef_ = coef
                predictor.intercept_ = intercept

                # Update the pipeline with the loaded predictor
                pipeline.named_steps['linear'] = predictor

                # Assign the pipeline to self.predictor
                self.predictor = pipeline

                print("Predictor loaded successfully.")
        else:
            print(f"The file at {path} does not exist.")
            self.predictor = self._init_predictor()
            self.retrain = True

        self._init_draft_history()

    def save_predictor(self):
        if self.use_lookup_table:
            # Save the lookup table
            lookup_df = pd.DataFrame(self.lookup_table)
            
            # If path does not exist, create the directory
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            
            # Save the lookup table to a CSV file
            lookup_df.to_csv(self.path, index=False)
            print(f"Lookup table saved to {self.path}.")
        else:
            # Retrain last time before saving
            self._train_predictor(self.predictor)
            
            # Extract the coefficients and intercept from the predictor
            predictor = self.predictor.named_steps['linear']
            coef = predictor.coef_
            intercept = predictor.intercept_

            # Create a DataFrame to save the coefficients and intercept
            data = pd.DataFrame({'coef': coef})
            data['intercept'] = intercept  # Broadcast intercept to match the length of coef

            # If path does not exist, create the directory
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

            # Save the DataFrame to a CSV file
            data.to_csv(self.path, index=False)

            print(f"Predictor saved to {self.path}.")
            self.retrain = False

    def _train_predictor(self, predictor: Union[Pipeline, np.ndarray]):
        nvtx.range_push("retrain predictor")

        global DEFER_EXIT
        if not DEFER_EXIT:
            print("[debug] (noppanat) time:", str(datetime.datetime.now()), flush=True)
            torch.cuda.cudart().cudaProfilerStart()
            Thread(target=defer_exit, args=(10,)).start()
            DEFER_EXIT = True
        
        if self.use_lookup_table:
            assert isinstance(predictor, np.ndarray)
        else:
            assert isinstance(predictor, Pipeline)

        nvtx.range_push("bin draft history")
        binned_df = self._get_binned_draft_history_df()
        nvtx.range_pop()

        X = (
            binned_df[["beta_ema", "draft_prob"]]
            .apply(lambda x: pd.to_numeric(x, errors="coerce"))
            .to_numpy()
        )
        y = binned_df["accept_prob"].to_numpy()

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
            X_predict = [[beta, draft] for beta in beta_values for draft in draft_values]

        # Check if the input data is empty
        if X.shape[0] == 0 or y.shape[0] == 0:
            print("No data available for training the predictor. Skipping this training step.")
            return
        
        if self.use_lookup_table:
            # Fill the lookup table based on the specified aggregation type
            predictor.fill(0)  # Reset the lookup table
            beta_indices = (X[:, 0] * (self.num_bins - 1)).astype(int)
            draft_indices = (X[:, 1] * (self.num_bins - 1)).astype(int)

            if self.sps_config.predictor_agg_type == "mean":
                np.add.at(predictor, (beta_indices, draft_indices), y)
                counts = np.zeros_like(predictor)
                np.add.at(counts, (beta_indices, draft_indices), 1)
                predictor = np.divide(predictor, counts, where=counts != 0)
            elif self.sps_config.predictor_agg_type == "median":
                for i in range(self.num_bins):
                    for j in range(self.num_bins):
                        mask = (beta_indices == i) & (draft_indices == j)
                        if mask.any():
                            predictor[i, j] = np.median(y[mask])
            print("Lookup table updated successfully.")
        else:
            # Linear regression training
            nvtx.range_push("fit predictor")
            predictor.fit(X, y)
            nvtx.range_pop()

        if PLOT_HEATMAP:
            if self.use_lookup_table:
                y_predict = predictor
            else:
                beta_values = np.linspace(0, 1, self.num_bins)
                draft_values = np.linspace(0, 1, self.num_bins)
                X_predict = [[beta, draft] for beta in beta_values for draft in draft_values]
                y_predict = predictor.predict(X_predict).reshape(self.num_bins, self.num_bins)

            # Create a matrix for real acceptance probability heatmap
            pivot_df = binned_df.pivot(index="beta_ema", columns="draft_prob", values="accept_prob")

            # Create the 2 subplots: Real, Predict
            fig, axs = plt.subplots(1, 2, figsize=(24, 6), sharey=True)

            # Plot "Real Data" (Acceptance Probability Heatmap)
            axs[0].imshow(pivot_df, cmap='Accent', interpolation='nearest', origin='lower', extent=grid_extent)
            axs[0].set_title('Real Acceptance Probability Heatmap')
            axs[0].set_xlabel('Draft Probability')

            # Plot "After" (Final Model Prediction)
            axs[1].imshow(y_predict, cmap='Accent', interpolation='nearest', origin='lower', extent=grid_extent)
            if self.use_lookup_table:
                axs[1].set_title('After (Lookup Table)')
            else:
                axs[1].set_title('After (Final Model Prediction)')
            axs[1].set_xlabel('Draft Probability')

            if not self.use_lookup_table:
                # Retrieve the final coefficients and intercept
                feature_names = predictor.named_steps['poly'].get_feature_names_out()
                linear_model =  predictor.named_steps['linear']

                # Print the coef_ and intercept_ of the linear model
                # print("Coefficients: ", linear_model.coef_)
                # print("Intercept: ", linear_model.intercept_)

                polynomial_function_string = create_polynomial_equation(linear_model, feature_names)
                fig.text(0.5, 0.01, polynomial_function_string, ha='center', va='bottom', fontsize=10)

            # Adjust the layout and add a colorbar
            fig.colorbar(axs[0].images[0], ax=axs, orientation='vertical', pad=0.03)
            fig.suptitle('Real vs. After Acceptance Probability Heatmaps')

            # Create the directory for saving the plots if it doesn't exist
            # os.makedirs('heatmap', exist_ok=True)
            # current_time = time.strftime("%Y%m%d-%H%M%S")
            # If path does not exist, create the directory
            png_path = self.path.replace('.csv', '.png')
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            plt.savefig(png_path)
            print(f"Plot saved to {png_path}.")

        nvtx.range_pop()
        self._drop_draft_history()
