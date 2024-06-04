import datetime
import os
from abc import abstractmethod
from threading import Thread
from queue import SimpleQueue
from typing import Dict, List, Union

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
from scipy.ndimage import gaussian_filter

from vllm.config import SpSConfig
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus

PLOT_HEATMAP = False
DEFER_EXIT = False
CMAP = "viridis_r"

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
        self.history_size = 1000000
        self.num_bins = 20
        self.predictor = self._init_predictor()
        self.lookup_table = self._init_lookup_table()
        self.draft_history = self._init_draft_history()
        self.retrain = False
        self.linspace = np.linspace(0, 1, self.num_bins + 1)
        self.initialize_exit_threshold_adjustment()

        if sps_config.use_async_training:
            self.worker = Thread(target=self._train_predictor_async)
            self.worker_queue = SimpleQueue()
            self.worker.start()
        else:
            self.worker = None
            self.worker_queue = None

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
                self._train_predictor_sync(seq.temperature)
            else:
                self.worker_queue.put(seq.temperature)

        self._update_drafted_accepted_df(seq)

        last_draft_token_id = seq.data.get_last_draft_token_id()
        draft_prob = np.clip(seq.data.get_draft_probs()[-1][last_draft_token_id].item(), 0, 0.999)
        beta_ema = np.clip(seq.get_beta_ema(), 0, 0.999)
    
        return [beta_ema, draft_prob]

    def _drop_draft_history(self):
        # drop the first element to keep the history size
        if len(self.draft_history["beta_ema"]) > self.history_size:
            self.draft_history["beta_ema"] = self.draft_history["beta_ema"][-self.history_size :]
            self.draft_history["draft_prob"] = self.draft_history["draft_prob"][-self.history_size :]
            self.draft_history["accept_prob"] = self.draft_history["accept_prob"][-self.history_size :]

    def _init_predictor(self) -> Pipeline:
        poly_features = PolynomialFeatures(
            degree=self.sps_config.predictor_degree, include_bias=True
        )
        # fit with dummy
        dummy_df = [[0, 0]]
        poly_features.fit(dummy_df)

        predictor = LinearRegression(fit_intercept=True)

        predictor.coef_ = np.array(
            [ 0., 0.93413575, 1.22473454, -0.32001846, -0.71799723, -1.63476777, -0.24324052, 0.4377291, -0.07319985, 1.27301201] 
        )
        predictor.intercept_ = 0.09655343164046182

        return Pipeline([("poly", poly_features), ("linear", predictor)])

    def _init_lookup_table(self):
        # Initialize an empty lookup table with the specified number of bins
        lookup_table = {}

        for temp in [0, 0.25, 0.5, 0.75, 1]:
            lookup_table[temp] = np.zeros((self.num_bins, self.num_bins))
            
        return lookup_table
    
    def _init_draft_history(self):
        draft_history = {}

        for temp in [0, 0.25, 0.5, 0.75, 1]:
            draft_history[temp] = {"beta_ema": [], "draft_prob": [], "accept_prob": []}

        return draft_history
    
    def _update_drafted_accepted_df(self, seq: Sequence):
        beta_emas, draft_probs, accept_probs = seq.get_new_draft_history()
        assert len(beta_emas) == len(draft_probs) == len(accept_probs)
        if len(beta_emas) == 0:
            return

        self.draft_history[seq.temperature]["beta_ema"].extend(beta_emas)
        self.draft_history[seq.temperature]["draft_prob"].extend(draft_probs)
        self.draft_history[seq.temperature]["accept_prob"].extend(accept_probs)

    def _drop_draft_history(self):
        # drop the first element to keep the history size
        for temperature in [0, 0.25, 0.5, 0.75, 1]:
            if len(self.draft_history[temperature]["beta_ema"]) > self.history_size:
                self.draft_history[temperature]["beta_ema"] = self.draft_history[temperature]["beta_ema"][-self.history_size :]
                self.draft_history[temperature]["draft_prob"] = self.draft_history[temperature]["draft_prob"][-self.history_size :]
                self.draft_history[temperature]["accept_prob"] = self.draft_history[temperature]["accept_prob"][-self.history_size :]

    def _lookup_predict(self, X, temperature):
        X = np.array(X)
        beta_indices = np.digitize(X[0], self.linspace) - 1
        draft_indices = np.digitize(X[1], self.linspace) - 1
        accept_probs = self.lookup_table[temperature][beta_indices, draft_indices]
        return accept_probs
    
    def _get_binned_draft_history_df(self, temperature) -> pd.DataFrame:
        draft_accepted_df = pd.DataFrame(self.draft_history[temperature])

        beta_ema_bins = self.linspace
        draft_prob_bins = self.linspace

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

        draft_accepted_df["accept_prob"] = draft_accepted_df["accept_prob"].clip(0, 1)

        binned_df = (
            draft_accepted_df.groupby(["beta_ema", "draft_prob"])
            .agg(accept_prob=pd.NamedAgg(
                    column="accept_prob", aggfunc=self.sps_config.predictor_agg_type
                ))
            .dropna()
        )

        idx = pd.MultiIndex.from_product(
            [np.linspace(0, 1, self.num_bins, endpoint=False), 
             np.linspace(0, 1, self.num_bins, endpoint=False)],
            names=["beta_ema", "draft_prob"]
        )

        binned_df = binned_df.reindex(idx, fill_value=0).reset_index()

        return binned_df

    def _get_binned_draft_history_df_std_dev(self, temperature) -> pd.DataFrame:
        draft_accepted_df = pd.DataFrame(self.draft_history[temperature])

        # Define bins
        beta_ema_bins = self.linspace
        draft_prob_bins = self.linspace

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
                    column="accept_prob", aggfunc="std"
                ))
            .dropna()
        )

        # Create a multi-index for all possible bin combinations
        idx = pd.MultiIndex.from_product(
            [np.linspace(0, 1, self.num_bins, endpoint=False), 
             np.linspace(0, 1, self.num_bins, endpoint=False)],
            names=["beta_ema", "draft_prob"]
        )

        # Reindex the binned_df to include all combinations, filling missing values with 0
        binned_df = binned_df.reindex(idx, fill_value=0).reset_index()

        return binned_df
    
    def _get_binned_draft_history_df_density(self, temperature) -> pd.DataFrame:
        draft_accepted_df = pd.DataFrame(self.draft_history[temperature])

        beta_ema_bins = self.linspace
        draft_prob_bins = self.linspace

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

        draft_accepted_df["accept_prob"] = draft_accepted_df["accept_prob"].clip(0, 1)

        binned_df = (
            draft_accepted_df.groupby(["beta_ema", "draft_prob"])
            .agg(accept_prob=pd.NamedAgg(
                    column="accept_prob", aggfunc="count"
                ))
            .dropna()
        )

        idx = pd.MultiIndex.from_product(
            [np.linspace(0, 1, self.num_bins, endpoint=False), 
             np.linspace(0, 1, self.num_bins, endpoint=False)],
            names=["beta_ema", "draft_prob"]
        )

        binned_df = binned_df.reindex(idx, fill_value=0).reset_index()
        binned_df["accept_prob"] = binned_df["accept_prob"] / binned_df["accept_prob"].sum()

        return binned_df

    ##################################################################################################

    # I think we should put this back to SpSConfig because SpSScheduler might need this. 
    def get_tile_size(self):
        if self.sps_config.use_tile_constraint == "none":
            return 100000
        elif self.sps_config.use_tile_constraint == "cut-128":
            return 128
        elif self.sps_config.use_tile_constraint == "skip-128-192":
            return [128, 192]
        else:
            raise NotImplementedError(f"Unsupported tile constraint: {self.sps_config.use_tile_constraint}")

    def initialize_exit_threshold_adjustment(self):
        self.exit_threshold = 0.5
        self.threshold_step = 0.01
        self.min_exit_threshold = 0.3
        self.max_exit_threshold = 0.7

    def calculate_draft_size(self, seq: Sequence) -> int:
        draft_size = 0
        for pred_prob in seq.predicton_probs:
            if pred_prob >= self.exit_threshold:
                draft_size += 1
            else:
                break
        return draft_size

    def calculate_load_imbalance(self, seq_list: List[Sequence]) -> float:
        draft_sizes = [self.calculate_draft_size(seq) for seq in seq_list]
        if len(draft_sizes) == 0:
            return 0.0
        
        mean_draft_size = np.mean(draft_sizes)
        if mean_draft_size == 0:
            return 0.0
        
        std_draft_size = np.std(draft_sizes)
        load_imbalance = std_draft_size / mean_draft_size

        return load_imbalance

    def find_optimal_exit_threshold(self, seq_list: List[Sequence], max_iterations: int = 10):
        acceptable_imbalance = 1  # This can be adjusted or calculated dynamically

        for _ in range(max_iterations):
            load_imbalance = self.calculate_load_imbalance(seq_list)

            if load_imbalance > acceptable_imbalance:
                self.exit_threshold -= self.threshold_step
            else:
                self.exit_threshold += self.threshold_step

            self.exit_threshold = np.clip(self.exit_threshold, self.min_exit_threshold, self.max_exit_threshold)


    def calculate_num_tokens(self, seq_list: List[Sequence]) -> int:
        num_tokens_to_target = 0

        for seq in seq_list:
            num_tokens_to_target += (seq.get_draft_len() + 1)

        return num_tokens_to_target

    def calculate_num_tokens_with_exit_threshold(self, seq_list: List[Sequence]) -> int:
        num_tokens_to_target = len(seq_list)

        for seq in seq_list:
            for pred_prob in seq.predicton_probs:
                if pred_prob >= self.exit_threshold:
                    num_tokens_to_target += 1
                else:
                    break
        return num_tokens_to_target

    def calculate_alive_seq(self, seq_list: List[Sequence]) -> int:
        alive_seq = len(seq_list)
        for seq in seq_list:
            if seq.predicton_probs[-1] < self.exit_threshold:
                alive_seq -= 1
        return alive_seq
    
    def _finalize_draft_size(self, seq_list: List[Sequence]):
        for seq in seq_list:
            seq.draft_size = 0
            for pred_prob in seq.predicton_probs:
                if pred_prob >= self.exit_threshold:
                    seq.draft_size += 1
                else:
                    break

        # draft_sizes = []
        # for seq in seq_list:
        #     draft_sizes.append(seq.draft_size)        
        
        # print(f"Draft sizes: {draft_sizes}")

    def _fill_until_tile_size(self, seq_list: List[Sequence], tile_size, num_tokens_to_target_threshold):
        num_tokens_to_fill = tile_size - num_tokens_to_target_threshold

        while num_tokens_to_fill > 0:
            max_accept_prob = 0
            max_accept_prob_seq = None

            for seq in seq_list:
                if seq.draft_size < seq.get_draft_len() - 1:
                    if seq.predicton_probs[seq.draft_size] > max_accept_prob:
                        max_accept_prob = seq.predicton_probs[seq.draft_size]
                        max_accept_prob_seq = seq

            if max_accept_prob_seq == None:
                break

            max_accept_prob_seq.draft_size += 1
            num_tokens_to_fill -= 1

    def _cut_to_tile_size(self, seq_list: List[Sequence], tile_size, num_tokens_to_target_threshold):
        num_tokens_to_cut = num_tokens_to_target_threshold - tile_size

        while num_tokens_to_cut > 0:
            min_accept_prob = 1
            min_accept_prob_seq = None

            for seq in seq_list:
                if seq.draft_size > 0:
                    if seq.predicton_probs[seq.draft_size - 1] <= min_accept_prob:
                        min_accept_prob = seq.predicton_probs[seq.draft_size - 1]
                        min_accept_prob_seq = seq

            if min_accept_prob_seq == None:
                break

            min_accept_prob_seq.draft_size -= 1
            num_tokens_to_cut -= 1

    def _ensure_exact_tile_size(self, seq_list: List[Sequence], tile_size, num_tokens_to_target_threshold):

        if num_tokens_to_target_threshold == tile_size:
            return

        elif num_tokens_to_target_threshold < tile_size:
            self._fill_until_tile_size(seq_list, tile_size, num_tokens_to_target_threshold)

        else:
            self._cut_to_tile_size(seq_list, tile_size, num_tokens_to_target_threshold) 
    
    def _apply_static_cut_tile_policy(self, seq_list: List[Sequence]):
        num_tokens_to_target = 0
        for seq in seq_list:
            num_tokens_to_target += (seq.get_draft_len() + 1)
        
        # tokens that will be generated at the next iteration
        num_tokens_to_generate = len(seq_list)

        if num_tokens_to_target + num_tokens_to_generate > self.get_tile_size():
            for seq in seq_list:
                seq.draft_size = seq.get_draft_len()

    def _apply_no_tile_policy(self, seq_list: List[Sequence]):
        alive_seq = len(seq_list)

        for seq in seq_list:
            for pred_prob in seq.predicton_probs:
                if pred_prob < self.exit_threshold:
                    alive_seq -= 1
                    break

        if alive_seq == 0:
            self._finalize_draft_size(seq_list)

    def _apply_dynamic_cut_tile_policy(self, seq_list: List[Sequence], tile_size):
        alive_seq = self.calculate_alive_seq(seq_list)
        num_tokens_to_target_threshold = self.calculate_num_tokens_with_exit_threshold(seq_list)
        last_iteration = seq_list[0].get_draft_len() == seq_list[0].draft_size

        if alive_seq == 0 or num_tokens_to_target_threshold >= tile_size or last_iteration:
            self._finalize_draft_size(seq_list)
            self._ensure_exact_tile_size(seq_list, tile_size, num_tokens_to_target_threshold)
            return

    def _apply_dynamic_skip_tile_policy(self, seq_list: List[Sequence], tile_size_cut, tile_size_skip):
        alive_seq = self.calculate_alive_seq(seq_list)
        num_tokens_to_target = self.calculate_num_tokens(seq_list)
        num_tokens_to_target_threshold = self.calculate_num_tokens_with_exit_threshold(seq_list)
        last_iteration = seq_list[0].get_draft_len() == seq_list[0].draft_size

        # first decide if alive_seq is 0
        if alive_seq == 0:
            self._finalize_draft_size(seq_list)
            self._ensure_exact_tile_size(seq_list, tile_size_cut,num_tokens_to_target_threshold)
            return
        
        if num_tokens_to_target_threshold < tile_size_cut:
            return 

        if alive_seq < len(seq_list) / 2:
            self._apply_dynamic_cut_tile_policy(seq_list, tile_size_cut)

        else:
            if num_tokens_to_target < tile_size_skip and not last_iteration:
                return
            
            else:
                for seq in seq_list:
                    seq.draft_size = seq.get_draft_len()

    # Main method to update draft size with adjusted exit threshold
    def update_draft_size_seq(self, seq_list: List[Sequence]):
        use_dynamic_draft_size = self.sps_config.use_dynamic_draft_size
        use_tile_constraint = self.sps_config.use_tile_constraint
        
        X = [self._get_seq_features(seq) for seq in seq_list]

        if not use_dynamic_draft_size:
            if use_tile_constraint == "none":
                return
            
            elif use_tile_constraint == "cut-128":
                self._apply_static_cut_tile_policy(seq_list)
                return

        elif use_dynamic_draft_size:
            # nvtx.range_push(f"predict {X[0][0]} {X[0][1]}")
            # if self.sps_config.use_lookup_table:
            #     accept_probs = np.clip(self._lookup_predict(X, seq_list), 0, 1)
            # else:
            #     accept_probs = np.clip(self.predictor.predict(X), 0, 1)
            # nvtx.range_pop()

            for i, seq in enumerate(seq_list):
                accept_prob = self._lookup_predict(X[i], seq.temperature)
                seq.cumulative_accept_prob *= accept_prob
                seq.predicton_probs.append(seq.cumulative_accept_prob)
                
            # Find the optimal exit threshold dynamically based on load imbalance
            # self.find_optimal_exit_threshold(seq_list)

            if use_tile_constraint == "none":
                self._apply_no_tile_policy(seq_list)
                return
            
            elif use_tile_constraint == "cut-128":
                tile_size = self.get_tile_size()
                self._apply_dynamic_cut_tile_policy(seq_list, tile_size)
                return
            
            elif use_tile_constraint == "skip-128-192":
                tile_size_cut = self.get_tile_size()[0]
                tile_size_skip = self.get_tile_size()[1]
                self._apply_dynamic_skip_tile_policy(seq_list, tile_size_cut, tile_size_skip)
                return
            
            else:
                raise NotImplementedError(f"Should not reach here. {use_dynamic_draft_size} {use_tile_constraint}")

        else:
            raise NotImplementedError(f"Should not reach here. {use_dynamic_draft_size} {use_tile_constraint}")

    ##################################################################################################

    def initialize(self, path):
        self.path = path

        if os.path.exists(path):
            if self.sps_config.use_lookup_table:
                # Load the lookup table for every temperature 
                for temperature in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    # path is directory. inside there should be 5 csv files
                    cleaned_temperature = temperature.replace(".", "_")
                    lookup_table_path = os.path.join(path, f"lookup_table_{cleaned_temperature}.csv")
                    assert os.path.exists(lookup_table_path), f"Lookup table for temperature {temperature} does not exist."
                    self.lookup_table[temperature] = pd.read_csv(lookup_table_path,  header=0).values
                    print(f"Lookup table loaded successfully from {path}.")

                # # gaussian filter
                # self.lookup_table = gaussian_filter(self.lookup_table, sigma=0.1)
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
            # Make dir
            os.makedirs(os.path.dirname(path), exist_ok=True)

            if self.sps_config.use_lookup_table:
                self.lookup_table = self._init_lookup_table()
                self.draft_history = self._init_draft_history()
                print("!!!!!!!", self.draft_history)
            else:
                self.predictor = self._init_predictor()
            self.retrain = True

    def save_predictor(self):
        if self.sps_config.use_lookup_table:
            for temperature in [0.0, 0.25, 0.5, 0.75, 1.0]:
                cleaned_temperature = str(temperature).replace(".", "_")
                lookup_table_path = os.path.join(self.path, f"lookup_table_{cleaned_temperature}.csv")
                # make dir 
                os.makedirs(os.path.dirname(lookup_table_path), exist_ok=True)
                lookup_df = pd.DataFrame(self.lookup_table[temperature])
                lookup_df.to_csv(lookup_table_path, index=False)

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

    def _train_predictor_sync(self, temperature: float):
        if self.sps_config.use_lookup_table:
            predictor = self.lookup_table
        else:
            predictor = self.predictor
        
        self._train_predictor(predictor, temperature)

    def _train_predictor_async(self):
        while True:
            temperature = self.worker_queue.get()
            # Make a copy of the draft optimizer states.
            if self.sps_config.use_lookup_table:
                predictor = self.lookup_table.copy()
                predictor[temperature] = np.zeros_like(self.lookup_table[temperature])
            else:
                predictor = clone(self.predictor)
            
            self._train_predictor(predictor, temperature)

            # Update the predictor
            if self.sps_config.use_lookup_table:
                self.lookup_table = predictor
            else:
                self.predictor = predictor

    def _train_predictor(self, predictor: Union[Pipeline, Dict[float, np.ndarray]], temperature: float):
        nvtx.range_push("retrain predictor")

        global DEFER_EXIT
        if DEFER_EXIT is not None and not DEFER_EXIT:
            print("[debug] (noppanat) time:", str(datetime.datetime.now()), flush=True)
            torch.cuda.cudart().cudaProfilerStart()
            Thread(target=defer_exit, args=(10,)).start()
            DEFER_EXIT = True
        
        if self.sps_config.use_lookup_table:
            assert isinstance(predictor, dict)
        else:
            assert isinstance(predictor, Pipeline)

        nvtx.range_push("bin draft history")
        binned_df = self._get_binned_draft_history_df(temperature)
        nvtx.range_pop()

        X = (
            binned_df[["beta_ema", "draft_prob"]]
            .apply(lambda x: pd.to_numeric(x, errors="coerce"))
            .to_numpy()
        )
        y = binned_df["accept_prob"].to_numpy()

        if X.shape[0] == 0 or y.shape[0] == 0:
            print("No data available for training the predictor. Skipping this training step.")
            return

        grid_extent = [0, 1, 0, 1]

        # Check if the input data is empty
        if X.shape[0] == 0 or y.shape[0] == 0:
            print("No data available for training the predictor. Skipping this training step.")
            return
        
        if self.sps_config.use_lookup_table:
            predictor[temperature].fill(0) 
            beta_indices = np.digitize(X[:, 0], self.linspace) - 1
            draft_indices = np.digitize(X[:, 1], self.linspace) - 1

            if self.sps_config.predictor_agg_type == "mean":
                np.add.at(predictor[temperature], (beta_indices, draft_indices), y)
                counts = np.zeros_like(predictor[temperature])
                np.add.at(counts, (beta_indices, draft_indices), 1)
                predictor[temperature] = np.divide(predictor[temperature], counts, where=counts != 0)
            elif self.sps_config.predictor_agg_type == "median":
                for i in range(self.num_bins):
                    for j in range(self.num_bins):
                        mask = (beta_indices == i) & (draft_indices == j)
                        if mask.any():
                            predictor[temperature][i, j] = np.median(y[mask])
            # predictor = gaussian_filter(predictor, sigma=0.1)
        else:
            # Linear regression training
            nvtx.range_push("fit predictor")
            predictor.fit(X, y)
            nvtx.range_pop()

        if PLOT_HEATMAP:
            if self.sps_config.use_lookup_table:
                y_predict = predictor[temperature]
            else:
                beta_values = self.linspace
                draft_values = self.linspace
                X_predict = [[beta, draft] for beta in beta_values for draft in draft_values]
                y_predict = predictor.predict(X_predict).reshape(self.num_bins, self.num_bins)

            pivot_df = binned_df.pivot(index="beta_ema", columns="draft_prob", values="accept_prob")
            std_binned_df = self._get_binned_draft_history_df_std_dev(temperature)
            std_pivot_df = std_binned_df.pivot(index="beta_ema", columns="draft_prob", values="accept_prob")
            density_binned_df = self._get_binned_draft_history_df_density(temperature)
            density_pivot_df = density_binned_df.pivot(index="beta_ema", columns="draft_prob", values="accept_prob")
            
            fig, axs = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

            axs[0].imshow(pivot_df, cmap=CMAP, interpolation='nearest', origin='lower', extent=grid_extent)
            axs[0].set_title(f'Real Acceptance Probability Heatmap {self.sps_config.predictor_agg_type}')
            axs[0].set_xlabel('Draft Probability')
            axs[0].set_ylabel('Beta EMA')

            axs[1].imshow(std_pivot_df, cmap=CMAP, interpolation='nearest', origin='lower', extent=grid_extent)
            axs[1].set_title('Standard Deviation')
            axs[1].set_xlabel('Draft Probability')

            # If density is equal to 0, color it red (to indicate no data)
            density_pivot_df = density_pivot_df.replace(0, np.nan)

            axs[2].imshow(density_pivot_df, cmap=CMAP, interpolation='nearest', origin='lower', extent=grid_extent)
            axs[2].set_title('Density')
            axs[2].set_xlabel('Draft Probability')

            axs[3].imshow(y_predict, cmap=CMAP, interpolation='nearest', origin='lower', extent=grid_extent)
            if self.sps_config.use_lookup_table:
                axs[3].set_title('After (Lookup Table)')
            else:
                axs[3].set_title('After (Regression Model)')
            axs[3].set_xlabel('Draft Probability')

            if not self.sps_config.use_lookup_table:
                feature_names = predictor.named_steps['poly'].get_feature_names_out()
                linear_model =  predictor.named_steps['linear']

                # Print the coef_ and intercept_ of the linear model
                # print("Coefficients: ", linear_model.coef_)
                # print("Intercept: ", linear_model.intercept_)

                polynomial_function_string = create_polynomial_equation(linear_model, feature_names)
                fig.text(0.5, 0.01, polynomial_function_string, ha='center', va='bottom', fontsize=10)

            fig.colorbar(axs[0].images[0], ax=axs, orientation='vertical', pad=0.03)
            fig.suptitle('Real vs. After Acceptance Probability Heatmaps')

            # png_path is the same as self.path but add {temp}.png dont replace just add 
            png_path = self.path + f"_{temperature}.png"
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            plt.savefig(png_path)
            print(f"Plot saved to {png_path}.")

        nvtx.range_pop()
        self._drop_draft_history()
