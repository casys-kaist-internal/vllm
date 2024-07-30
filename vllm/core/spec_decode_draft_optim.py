import numpy as np
import pandas as pd
import time
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vllm.sequence import Sequence, SequenceGroup
from vllm.config import SpecDecodeConfig
from vllm.utils import nvtx_range

PLOT_ACCEPT_PROB = False
DEBUG = False
UPDATE_INTERVAL = 50
MIN_ACCEPT_CNT_LENGTH = 10


class SpecDecodeDraftSizeOptimizer:
    def __init__(self, spec_decode_config: SpecDecodeConfig):
        self.spec_decode_config = spec_decode_config
        self.num_bins_accept_cnt = self.spec_decode_config.draft_size
        self.num_bins_logprobs = 10
        self.num_bins_temperature = 5
        self.init_lookup_table()
        self.accept_cnt_linspace = np.linspace(
            1, self.spec_decode_config.draft_size, self.num_bins_accept_cnt - 1, endpoint=False)
        self.logprob_linspace = np.linspace(
            -5, 0, self.num_bins_logprobs - 1, endpoint=False)
        self.temperature_linspace = np.linspace(
            0.25, 1, self.num_bins_temperature - 1, endpoint=False)
        self.update_idx = 0
        self.spec_decode_history = {
            "accept_temperatures": [],
            "reject_temperatures": [],
            "accept_avg_cnt": [],
            "reject_avg_cnt": [],
            "accept_logprobs": [],
            "reject_logprobs": [],
        }

    def init_lookup_table(self):
        # Initialize a 3D lookup table with dimensions for temperature, accept/reject counts, and logprobs
        self.lookup_table = np.zeros(
            (self.num_bins_temperature, self.num_bins_accept_cnt, self.num_bins_logprobs, 2))

    def get_indices(self, temperature, avg_cnt, logprob):
        # Get the bin index for temperature
        temp_index = np.digitize(temperature, self.temperature_linspace)

        # Get the bin index for avg_cnt
        cnt_index = np.digitize(
            avg_cnt, self.accept_cnt_linspace)

        # Get the bin index for logprob
        logprob_index = np.digitize(logprob, self.logprob_linspace)

        if DEBUG:
            print("\n=== Debug Information ===")
            print("Accept Count Linspace:")
            print(f"  Bin 0: [0, {self.accept_cnt_linspace[0]:.2f})")
            for i in range(len(self.accept_cnt_linspace) - 1):
                print(
                    f"  Bin {i + 1}: [{self.accept_cnt_linspace[i]:.2f}, {self.accept_cnt_linspace[i + 1]:.2f})")
            print(
                f"  Bin {len(self.accept_cnt_linspace)}: [{self.accept_cnt_linspace[-1]:.2f}, {self.spec_decode_config.draft_size}]")

            print("\nLog Probability Linspace:")
            print(f"  Bin 0: [-inf, {self.logprob_linspace[0]:.2f})")
            for i in range(len(self.logprob_linspace) - 1):
                print(
                    f"  Bin {i + 1}: [{self.logprob_linspace[i]:.2f}, {self.logprob_linspace[i + 1]:.2f})")
            print(
                f"  Bin {len(self.logprob_linspace)}: [{self.logprob_linspace[-1]:.2f}, 0]")

            print("\nTemperature Linspace:")
            print(f"  Bin 0: [0, {self.temperature_linspace[0]:.2f})")
            for i in range(len(self.temperature_linspace) - 1):
                print(
                    f"  Bin {i + 1}: [{self.temperature_linspace[i]:.2f}, {self.temperature_linspace[i + 1]:.2f})")
            print(
                f"  Bin {len(self.temperature_linspace)}: [{self.temperature_linspace[-1]:.2f}, 1]")

            print(
                f"\nDebug: avg_cnt = {avg_cnt}, logprob = {logprob}, temperature = {temperature}")
            print(
                f"  Calculated Indices -> temp_index: {temp_index}, cnt_index: {cnt_index}, logprob_index: {logprob_index}")
            print("=========================\n")

        return temp_index, cnt_index, logprob_index

    def _plot_accept_prob_with_filter(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            total_density = self.lookup_table.sum(axis=(0, 1, 2, 3))
            threshold = total_density * 0.0001  # 0.1% of the total density
            density = self.lookup_table.sum(axis=3)
            accept_prob = np.true_divide(
                self.lookup_table[:, :, :, 0], density, where=(density > threshold))

        fig, axes = plt.subplots(
            nrows=2, ncols=self.num_bins_temperature, figsize=(40, 12))
        axes = axes.flatten()

        for i in range(self.num_bins_temperature):
            ax_accept_prob = axes[i]
            cax_accept_prob = ax_accept_prob.imshow(
                np.ma.masked_where(density[i] <= threshold, accept_prob[i]), cmap='viridis_r', origin='lower', aspect='auto')

            # Set the axis labels
            ax_accept_prob.set_xlabel('Log Probability')
            ax_accept_prob.set_ylabel('Average Accept Count')

            # Set the ticks for x and y axis
            ax_accept_prob.set_xticks(
                np.arange(len(self.logprob_linspace) + 1))
            ax_accept_prob.set_yticks(
                np.arange(len(self.accept_cnt_linspace) + 1))

            # Set the tick labels for x and y axis with ranges
            logprob_labels = [f'-inf - {self.logprob_linspace[0]:.2f}'] + \
                [f'{self.logprob_linspace[i]:.2f} - {self.logprob_linspace[i+1]:.2f}'
                 for i in range(0, len(self.logprob_linspace) - 1)] + [
                    f'{self.logprob_linspace[-1]: .2f} - 0']

            accept_cnt_labels = [f'0 - {self.accept_cnt_linspace[0]:.2f}'] + \
                [f'{self.accept_cnt_linspace[i]:.2f} - {self.accept_cnt_linspace[i+1]:.2f}'
                 for i in range(len(self.accept_cnt_linspace) - 1)] + [
                    f'{self.accept_cnt_linspace[-1]:.2f} - {self.spec_decode_config.draft_size}']

            ax_accept_prob.set_xticklabels(logprob_labels)
            ax_accept_prob.set_yticklabels(accept_cnt_labels)

            # Rotate the x labels for better readability
            ax_accept_prob.set_xticklabels(
                ax_accept_prob.get_xticklabels(), rotation=45)

            # Add a color bar
            cbar_accept_prob = fig.colorbar(
                cax_accept_prob, ax=ax_accept_prob, shrink=0.75)
            cbar_accept_prob.set_label('Acceptance Probability')

            # Add the temperature bin as the title
            if i == 0:
                temp_label = f'< {self.temperature_linspace[0]:.2f}'
            elif i == self.num_bins_temperature - 1:
                temp_label = f'>= {self.temperature_linspace[-1]:.2f}'
            else:
                temp_label = f'{self.temperature_linspace[i-1]:.2f} - {self.temperature_linspace[i]:.2f}'
            ax_accept_prob.set_title(f'Temperature: {temp_label}')

            ax_density = axes[i + self.num_bins_temperature]
            cax_density = ax_density.imshow(
                np.ma.masked_where(density[i] <= threshold, density[i]), cmap='plasma', origin='lower', aspect='auto')

            # Set the axis labels
            ax_density.set_xlabel('Log Probability')
            ax_density.set_ylabel('Average Accept Count')

            # Set the ticks for x and y axis
            ax_density.set_xticks(np.arange(len(self.logprob_linspace) + 1))
            ax_density.set_yticks(np.arange(len(self.accept_cnt_linspace) + 1))

            # Set the tick labels for x and y axis with ranges
            ax_density.set_xticklabels(logprob_labels)
            ax_density.set_yticklabels(accept_cnt_labels)

            # Rotate the x labels for better readability
            ax_density.set_xticklabels(
                ax_density.get_xticklabels(), rotation=45)

            # Add a color bar
            cbar_density = fig.colorbar(
                cax_density, ax=ax_density, shrink=0.75)
            cbar_density.set_label('Density')

        plt.tight_layout()
        plt.savefig('accept_prob_and_density.png')
        plt.close()

    def _plot_accept_prob(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            accept_prob = np.true_divide(self.lookup_table[:, :, :, 0],
                                         self.lookup_table.sum(axis=3))

        fig, axes = plt.subplots(
            nrows=2, ncols=self.num_bins_temperature, figsize=(40, 12))
        axes = axes.flatten()

        for i in range(self.num_bins_temperature):
            ax_accept_prob = axes[i]
            cax_accept_prob = ax_accept_prob.imshow(
                accept_prob[i], cmap='viridis_r', origin='lower', aspect='auto')

            # Set the axis labels
            ax_accept_prob.set_xlabel('Log Probability')
            ax_accept_prob.set_ylabel('Average Accept Count')

            # Set the ticks for x and y axis
            ax_accept_prob.set_xticks(
                np.arange(len(self.logprob_linspace) + 1))
            ax_accept_prob.set_yticks(
                np.arange(len(self.accept_cnt_linspace) + 1))

            # Set the tick labels for x and y axis with ranges
            logprob_labels = [f'-inf - {self.logprob_linspace[0]:.2f}'] + \
                [f'{self.logprob_linspace[i]:.2f} - {self.logprob_linspace[i+1]:.2f}'
                 for i in range(0, len(self.logprob_linspace) - 1)] + [
                    f'{self.logprob_linspace[-1]: .2f} - 0']

            accept_cnt_labels = [f'0 - {self.accept_cnt_linspace[0]:.2f}'] + \
                [f'{self.accept_cnt_linspace[i]:.2f} - {self.accept_cnt_linspace[i+1]:.2f}'
                 for i in range(len(self.accept_cnt_linspace) - 1)] + [
                    f'{self.accept_cnt_linspace[-1]:.2f} - {self.spec_decode_config.draft_size}']

            ax_accept_prob.set_xticklabels(logprob_labels)
            ax_accept_prob.set_yticklabels(accept_cnt_labels)

            # Rotate the x labels for better readability
            ax_accept_prob.set_xticklabels(
                ax_accept_prob.get_xticklabels(), rotation=45)

            # Add a color bar
            cbar_accept_prob = fig.colorbar(
                cax_accept_prob, ax=ax_accept_prob, shrink=0.75)
            cbar_accept_prob.set_label('Acceptance Probability')

            # Add the temperature bin as the title
            if i == 0:
                temp_label = f'< {self.temperature_linspace[0]:.2f}'
            elif i == self.num_bins_temperature - 1:
                temp_label = f'>= {self.temperature_linspace[-1]:.2f}'
            else:
                temp_label = f'{self.temperature_linspace[i-1]:.2f} - {self.temperature_linspace[i]:.2f}'
            ax_accept_prob.set_title(f'Temperature: {temp_label}')

            ax_density = axes[i + self.num_bins_temperature]
            cax_density = ax_density.imshow(
                self.lookup_table.sum(axis=3)[i], cmap='plasma', origin='lower', aspect='auto')

            # Set the axis labels
            ax_density.set_xlabel('Log Probability')
            ax_density.set_ylabel('Average Accept Count')

            # Set the ticks for x and y axis
            ax_density.set_xticks(np.arange(len(self.logprob_linspace) + 1))
            ax_density.set_yticks(np.arange(len(self.accept_cnt_linspace) + 1))

            # Set the tick labels for x and y axis with ranges
            ax_density.set_xticklabels(logprob_labels)
            ax_density.set_yticklabels(accept_cnt_labels)

            # Rotate the x labels for better readability
            ax_density.set_xticklabels(
                ax_density.get_xticklabels(), rotation=45)

            # Add a color bar
            cbar_density = fig.colorbar(
                cax_density, ax=ax_density, shrink=0.75)
            cbar_density.set_label('Density')

        plt.tight_layout()
        plt.savefig('accept_prob_and_density.png')
        plt.close()

    def _plot_4d_accept_prob(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            accept_prob = np.true_divide(self.lookup_table[:, :, :, 0],
                                         self.lookup_table.sum(axis=3))

        # Create a meshgrid for the plot
        T, A, L = np.meshgrid(self.temperature_linspace,
                              self.accept_cnt_linspace, self.logprob_linspace, indexing='ij')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Flatten the meshgrid and acceptance probability arrays
        T_flat = T.flatten()
        A_flat = A.flatten()
        L_flat = L.flatten()
        accept_prob_flat = accept_prob.flatten()

        # Plot the scatter with color mapping
        sc = ax.scatter(T_flat, A_flat, L_flat,
                        c=accept_prob_flat, cmap='viridis')

        # Set axis labels
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Average Accept Count')
        ax.set_zlabel('Log Probability')

        # Add a color bar
        cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Acceptance Probability')

        plt.tight_layout()
        plt.savefig('accept_prob_4d.png')
        plt.close()

    @ nvtx_range("update_spec_decode_history")
    def update_spec_decode_history(self, seq_group_list: List[SequenceGroup]):
        self.update_idx += 1

        accept_temperatures = []
        reject_temperatures = []
        accept_avg_cnts = []
        reject_avg_cnts = []
        all_accept_logprobs = []
        all_reject_logprobs = []

        for seq_group in seq_group_list:
            seq = seq_group.get_seqs()[0]
            temperature = seq_group.sampling_params.temperature
            if not seq.accept_cnts or len(seq.accept_cnts) < MIN_ACCEPT_CNT_LENGTH:
                seq.accept_logprobs.clear()
                seq.reject_logprobs.clear()
                continue

            accept_logprobs = np.array(seq.accept_logprobs)
            reject_logprobs = np.array(seq.reject_logprobs)

            seq.accept_logprobs.clear()
            seq.reject_logprobs.clear()

            avg_accept_cnt = np.mean(seq.accept_cnts)

            # If accept_logprobs is not empty, update history
            if len(accept_logprobs) > 0:
                accept_temperatures.extend(
                    [temperature] * len(accept_logprobs))
                accept_avg_cnts.extend([avg_accept_cnt] * len(accept_logprobs))
                all_accept_logprobs.extend(accept_logprobs)

            # If reject_logprobs is not empty, update history
            if len(reject_logprobs) > 0:
                reject_temperatures.extend(
                    [temperature] * len(reject_logprobs))
                reject_avg_cnts.extend([avg_accept_cnt] * len(reject_logprobs))
                all_reject_logprobs.extend(reject_logprobs)

        self.spec_decode_history["accept_temperatures"].extend(
            accept_temperatures)
        self.spec_decode_history["reject_temperatures"].extend(
            reject_temperatures)
        self.spec_decode_history["accept_avg_cnt"].extend(accept_avg_cnts)
        self.spec_decode_history["reject_avg_cnt"].extend(reject_avg_cnts)
        self.spec_decode_history["accept_logprobs"].extend(all_accept_logprobs)
        self.spec_decode_history["reject_logprobs"].extend(all_reject_logprobs)

        # Update lookup tables at intervals
        if self.update_idx % UPDATE_INTERVAL == 0:
            self.update_lookup_tables()

    @ nvtx_range("update_lookup_tables")
    def update_lookup_tables(self):
        print("Updating lookup tables")
        accept_temperatures = np.array(
            self.spec_decode_history["accept_temperatures"])
        reject_temperatures = np.array(
            self.spec_decode_history["reject_temperatures"])
        accept_avg_cnts = np.array(self.spec_decode_history["accept_avg_cnt"])
        reject_avg_cnts = np.array(self.spec_decode_history["reject_avg_cnt"])
        accept_logprobs = np.array(self.spec_decode_history["accept_logprobs"])
        reject_logprobs = np.array(self.spec_decode_history["reject_logprobs"])

        if accept_avg_cnts.size > 0:
            temp_indices, accept_cnt_indices, accept_logprob_indices = self.get_indices(
                accept_temperatures, accept_avg_cnts, accept_logprobs)
            np.add.at(
                self.lookup_table[:, :, :, 0], (temp_indices, accept_cnt_indices, accept_logprob_indices), 1)

        if reject_avg_cnts.size > 0:
            temp_indices, reject_cnt_indices, reject_logprob_indices = self.get_indices(
                reject_temperatures, reject_avg_cnts, reject_logprobs)
            np.add.at(
                self.lookup_table[:, :, :, 1], (temp_indices, reject_cnt_indices, reject_logprob_indices), 1)

        self.spec_decode_history = {
            "accept_temperatures": [],
            "reject_temperatures": [],
            "accept_avg_cnt": [],
            "reject_avg_cnt": [],
            "accept_logprobs": [],
            "reject_logprobs": [],
        }

        if PLOT_ACCEPT_PROB:
            self._plot_accept_prob()
            # self._plot_4d_accept_prob()

    @ nvtx_range("predict_accept_probs")
    def predict_accept_probs(self, seq_group_list: List[SequenceGroup]):
        seq_features = np.array([self.get_seq_features(seq_group)
                                for seq_group in seq_group_list])
        temp_indices, avg_accept_cnt_indices, logprob_indices = self.get_indices(
            seq_features[:, 0], seq_features[:, 1], seq_features[:, 2])

        accept_probs = self.lookup_table[temp_indices,
                                         avg_accept_cnt_indices, logprob_indices, 0]
        reject_probs = self.lookup_table[temp_indices,
                                         avg_accept_cnt_indices, logprob_indices, 1]

        # Add a small epsilon to avoid division by zero
        probs = accept_probs / (accept_probs + reject_probs + 1e-9)

        for i, seq in enumerate(seq_group_list):
            seq = seq.get_seqs()[0]
            if seq.predicted_cumulated_accept_probs:
                last_prob = seq.predicted_cumulated_accept_probs[-1]
            else:
                last_prob = 1.0

            if len(seq.accept_cnts) < MIN_ACCEPT_CNT_LENGTH:
                seq.predicted_cumulated_accept_probs.append(last_prob)
                continue

            cumulative_prob = last_prob * probs[i]
            seq.predicted_cumulated_accept_probs.append(cumulative_prob)

            assert (len(seq.predicted_cumulated_accept_probs) ==
                    seq.data.get_draft_len())

    def get_seq_features(self, seq_group: SequenceGroup) -> List[float]:
        # If the sequence has less than MIN_ACCEPT_CNT_LENGTH accept counts,
        # return the default draft size and 0 logprob and temperature
        seq = seq_group.get_seqs()[0]
        temperature = seq_group.sampling_params.temperature

        if len(seq.accept_cnts) < MIN_ACCEPT_CNT_LENGTH:
            return [temperature, self.spec_decode_config.draft_size, 0]

        avg_accept_cnt = np.mean(seq.accept_cnts)
        draft_logprob = seq.data.draft_logprobs[-1]

        return [temperature, avg_accept_cnt, draft_logprob]
