from abc import abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from vllm.sequence import Sequence, SequenceGroup, SequenceStatus


class DraftSizeOptimizer:
    @abstractmethod
    def update_draft_sizes(self, seq_group_list: List[SequenceGroup]):
        raise NotImplementedError()

    @abstractmethod
    def update_draft_size_seq(self, seq: Sequence):
        raise NotImplementedError()


class BetaEMADraftSizeOptimizer(DraftSizeOptimizer):
    def __init__(self):
        self.cumulative_accept_prob = 1
        self.retrain_period = 100000
        self.num_bins = 20
        self.agg_type = "mean"

        self.poly_features = PolynomialFeatures(degree=3, include_bias=False)
        self.predictor = self._init_predictor()
        self.draft_history = self._init_draft_history()

    def update_draft_sizes(self, seq_group_list: List[SequenceGroup]):
        for seq_group in seq_group_list:
            seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
            self.update_draft_size_seq(seq)

    def update_draft_size_seq(self, seq: Sequence):
        if len(self.draft_history["beta_ema"]) >= self.retrain_period:
            self._train_predictor()
            self.draft_history = self._init_draft_history()

        self._update_drafted_accepted_df(seq)

        if self._check_early_stop(seq):
            seq.draft_size = seq.get_draft_len()

    def _init_draft_history(self) -> Dict[str, List[float]]:
        return {
            "beta_ema": [],
            "draft_prob": [],
            "accept_prob": [],
        }

    def _init_predictor(self) -> LinearRegression:
        predictor = LinearRegression()
        predictor.coef_ = np.array(
            [1.81, 1.44, -1.95, -1.17, -1.09, 0.61, 1.13, -0.42, 0.75]
        )
        predictor.intercept_ = -0.03
        return predictor

    def _check_early_stop(self, seq: Sequence) -> bool:
        # Check if the sequence should be stopped early
        # Get probability of last draft token
        last_draft_token_id = seq.data.get_last_draft_token_id()
        draft_prob = seq.data.get_draft_probs()[-1][last_draft_token_id].item()
        beta_ema = seq.get_beta_ema()

        predicted_accept_prob = self._predict_accept_prob(beta_ema, draft_prob)

        self.cumulative_accept_prob *= predicted_accept_prob

        random_accept_prob = np.random.uniform(0, 1)
        if self.cumulative_accept_prob < random_accept_prob:
            self.cumulative_accept_prob = 1
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
        X = [[beta_ema, draft_prob]]
        X_poly = self.poly_features.fit_transform(X)
        return np.clip(self.predictor.predict(X_poly), 0, 1)  # prevent divergence

    def _train_predictor(self):
        binned_df = self._get_binned_draft_history_df()
        # Prepare data for regression
        X = binned_df[["beta_ema_binned_code", "draft_prob_binned_code"]].values
        y = binned_df["accept_prob"].values

        # Polynomial features
        X_poly = self.poly_features.fit_transform(X)

        # Linear regression
        self.predictor.fit(X_poly, y)

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
        self.draft_history = self._init_draft_history()
