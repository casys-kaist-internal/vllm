import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read the data from prob.txt
data = []
with open('prob.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(' ')

        if len(parts) == 4:
            label = parts[1].strip()
            prob = float(parts[2].strip())
            logprob = float(parts[3].strip())
            data.append((label, prob, logprob))

# discard first 1000 samples
data = data[10000:]

# Step 2: Separate the data into accept and reject categories for prob
accept_probs = [prob for label, prob, logprob in data if label == 'accept']
reject_probs = [prob for label, prob,
                logprob in data if label in ['reject', 'discard']]

# Step 3: Create bins for the probabilities based on bin_cnt
bin_cnt = 20
bins = np.linspace(0, 1, bin_cnt + 1)

# Step 4: Calculate the ratio of accept probabilities in each bin for prob
accept_hist, _ = np.histogram(accept_probs, bins=bins)
reject_hist, _ = np.histogram(reject_probs, bins=bins)

total_hist = accept_hist + reject_hist
ratios = np.divide(accept_hist, total_hist, out=np.zeros_like(
    accept_hist, dtype=float), where=total_hist != 0)
densities = 100 * total_hist / sum(total_hist)

# Step 5: Separate the data into accept and reject categories for logprob
accept_logprobs = [logprob for label, prob,
                   logprob in data if label == 'accept']
reject_logprobs = [logprob for label, prob,
                   logprob in data if label in ['reject', 'discard']]

# Step 6: Create bins for the logprobabilities based on bin_cnt
log_bins = np.linspace(min(accept_logprobs + reject_logprobs),
                       max(accept_logprobs + reject_logprobs), bin_cnt + 1)

# Step 7: Calculate the ratio of accept logprobabilities in each bin for logprob
accept_log_hist, _ = np.histogram(accept_logprobs, bins=log_bins)
reject_log_hist, _ = np.histogram(reject_logprobs, bins=log_bins)

total_log_hist = accept_log_hist + reject_log_hist
log_ratios = np.divide(accept_log_hist, total_log_hist, out=np.zeros_like(
    accept_log_hist, dtype=float), where=total_log_hist != 0)
log_densities = 100 * total_log_hist / sum(total_log_hist)

# Step 8: Plot the histograms showing the ratios
bin_centers = 0.5 * (bins[:-1] + bins[1:])
log_bin_centers = 0.5 * (log_bins[:-1] + log_bins[1:])

fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot for prob
axs[0].bar(bin_centers, ratios, width=1.0/bin_cnt,
           color='blue', alpha=0.5, label='Accept / Total')
axs[0].set_xlabel('Predicted Probability Bins')
axs[0].set_ylabel('Actual Probability')
axs[0].set_title('Ratio of Accept to Total Probabilities in Each Bin')
axs[0].grid(True)
axs[0].legend()

ax0_twin = axs[0].twinx()
ax0_twin.plot(bin_centers, densities, color='black',
              marker='o', linestyle='dashed', label='Density')
ax0_twin.set_ylabel('Density (%)')

for bar, ratio in zip(axs[0].patches, ratios):
    height = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{ratio:.2f}', ha='center', va='bottom')

# Plot for logprob
axs[1].bar(log_bin_centers, log_ratios, width=(log_bins[1] -
           log_bins[0]), color='green', alpha=0.5, label='Accept / Total')
axs[1].set_xlabel('Log Probability Bins')
axs[1].set_ylabel('Actual Probability')
axs[1].set_title('Ratio of Accept to Total Log Probabilities in Each Bin')
axs[1].grid(True)
axs[1].legend()

ax1_twin = axs[1].twinx()
ax1_twin.plot(log_bin_centers, log_densities, color='black',
              marker='o', linestyle='dashed', label='Density')
ax1_twin.set_ylabel('Density (%)')

for bar, ratio in zip(axs[1].patches, log_ratios):
    height = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{ratio:.2f}', ha='center', va='bottom')

plt.tight_layout()

# Save and show the plot
plt.savefig('accept_reject_ratio_histogram.png')
plt.show()

# Print frequencies
accept_count = len(accept_probs)
reject_count = len(reject_probs)

print(f'Frequency of Accept: {accept_count}')
print(f'Frequency of Reject: {reject_count}')
