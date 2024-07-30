import matplotlib.pyplot as plt
import numpy as np

# Read the data from logprob.txt
data = []
with open('logprob.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) == 4:
            _, avg_accept_cnt, status, logprob = parts
            data.append((status.strip(), float(
                avg_accept_cnt.strip()), float(logprob.strip())))

# Convert data to numpy arrays for easier manipulation
avg_accept_cnts = np.array([avg_accept_cnt for _, avg_accept_cnt, _ in data])
logprobs = np.array([logprob for _, _, logprob in data])
statuses = np.array([1 if status == 'accept' else 0 for status, _, _ in data])

# Define bins for logprob and avg_accept_cnt
logprob_bins = np.linspace(logprobs.min(), logprobs.max(), 20)
avg_accept_cnt_bins = np.linspace(
    avg_accept_cnts.min(), avg_accept_cnts.max(), 20)

# Create a 2D histogram for counts
hist_counts, xedges, yedges = np.histogram2d(logprobs, avg_accept_cnts, bins=[
                                             logprob_bins, avg_accept_cnt_bins])

# Create a 2D histogram for acceptances
hist_accepts, _, _ = np.histogram2d(logprobs[statuses == 1], avg_accept_cnts[statuses == 1], bins=[
                                    logprob_bins, avg_accept_cnt_bins])

# Calculate accept proportions
accept_proportions = np.zeros_like(hist_counts)
accept_proportions[hist_counts > 0] = hist_accepts[hist_counts >
                                                   0] / hist_counts[hist_counts > 0]

# Plot the data
plt.figure(figsize=(10, 6))
plt.imshow(accept_proportions.T, origin='lower', aspect='auto',
           extent=[logprob_bins[0], logprob_bins[-1],
                   avg_accept_cnt_bins[0], avg_accept_cnt_bins[-1]],
           cmap='viridis_r')
plt.colorbar(label='Proportion of Accept')
plt.xlabel('Log Probability')
plt.ylabel('Average Accept Count')
plt.title(
    'Proportion of Accept Decisions by Log Probability and Average Accept Count')
plt.grid(False)  # Typically we turn off grid for heatmaps
plt.savefig('logprob_avg_accept_cnt_bins.png')
plt.show()
