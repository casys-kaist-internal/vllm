import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# CSV files for the three scenarios
latency_colocation_csv_file = "token_latencies_over_time_True_True.csv"
latency_nocolocation_csv_file = "token_latencies_over_time_False_True.csv"
latency_ar_csv_file = "token_latencies_over_time_AR.csv"

requests_colocation_csv_file = "requests_over_time_True_True.csv"
requests_nocolocation_csv_file = "requests_over_time_False_True.csv"
requests_ar_csv_file = "requests_over_time_AR.csv"

# Read the latency CSV data into pandas DataFrames
latency_data_colocation = pd.read_csv(latency_colocation_csv_file)
latency_data_nocolocation = pd.read_csv(latency_nocolocation_csv_file)
latency_data_ar = pd.read_csv(latency_ar_csv_file)

# Read the requests CSV data into pandas DataFrames
requests_data_colocation = pd.read_csv(requests_colocation_csv_file)
requests_data_nocolocation = pd.read_csv(requests_nocolocation_csv_file)
requests_data_ar = pd.read_csv(requests_ar_csv_file)

# Sort data by Time(s) in case it's not sorted
latency_data_colocation.sort_values('Time(s)', inplace=True)
latency_data_nocolocation.sort_values('Time(s)', inplace=True)
latency_data_ar.sort_values('Time(s)', inplace=True)

requests_data_colocation.sort_values('Time(s)', inplace=True)
requests_data_nocolocation.sort_values('Time(s)', inplace=True)
requests_data_ar.sort_values('Time(s)', inplace=True)

# Apply a moving average to smooth the latencies
window_size_latency = 200  # Adjusted smoothing window for a smoother curve

# For latency data
latency_data_colocation['Smoothed_Latency'] = latency_data_colocation['Token_Latency(s/token)'].rolling(
    window=window_size_latency, center=True).mean()
latency_data_nocolocation['Smoothed_Latency'] = latency_data_nocolocation['Token_Latency(s/token)'].rolling(
    window=window_size_latency, center=True).mean()
latency_data_ar['Smoothed_Latency'] = latency_data_ar['Token_Latency(s/token)'].rolling(
    window=window_size_latency, center=True).mean()

# Apply a moving average to smooth the number of running requests
window_size_requests = 200  # Adjusted smoothing window for a smoother curve

# For requests data
requests_data_colocation['Num_Requests'] = requests_data_colocation['Num_Requests'].rolling(
    window=window_size_requests, center=True).mean()
requests_data_nocolocation['Num_Requests'] = requests_data_nocolocation['Num_Requests'].rolling(
    window=window_size_requests, center=True).mean()
requests_data_ar['Num_Requests'] = requests_data_ar['Num_Requests'].rolling(
    window=window_size_requests, center=True).mean()

# Set figure size suitable for a two-column paper (approx. 5 inches wide)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 3), sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})

# Plot the smoothed latencies for all datasets on ax1
ax1.plot(latency_data_colocation['Time(s)'], latency_data_colocation['Smoothed_Latency'],
         label='With Colocation', color='#228B22', linestyle='-', linewidth=1.2)
ax1.plot(latency_data_nocolocation['Time(s)'], latency_data_nocolocation['Smoothed_Latency'],
         label='Without Colocation', color='#3864B9', linestyle='-', linewidth=1.2)
ax1.plot(latency_data_ar['Time(s)'], latency_data_ar['Smoothed_Latency'],
         label='AR', color='red', linestyle='--', linewidth=1.2)

# Set log scale for y-axis and add labels
ax1.set_yscale('log')
ax1.set_ylabel('Token Latency (s/token)', fontsize=8)

# ax1.set_ylim(0.01, 0.5)
ax1.set_xlim(0, 300)

# Remove x-axis labels from ax1 to avoid overlap
ax1.tick_params(axis='x', which='both', labelbottom=False)

# y tick font size 
ax1.tick_params(axis='both', which='major', labelsize=8)


# Plot the number of running requests over time on ax2
ax2.plot(requests_data_colocation['Time(s)'], requests_data_colocation['Num_Requests'],
         label='With Colocation', color='#228B22', linestyle='-', linewidth=1.2)
ax2.plot(requests_data_nocolocation['Time(s)'], requests_data_nocolocation['Num_Requests'],
         label='Without Colocation', color='#3864B9', linestyle='-', linewidth=1.2)
ax2.plot(requests_data_ar['Time(s)'], requests_data_ar['Num_Requests'],
         label='AR', color='red', linestyle='--', linewidth=1.2)

# Add labels to ax2
ax2.set_xlabel('Time (s)', fontsize=8)
ax2.set_ylabel('Batch Size', fontsize=8)
ax2.set_xticks([0, 60, 120, 180, 240, 300])
ax2.tick_params(axis='both', which='major', labelsize=6)

# Define the time intervals for request rates
total_duration = 5 * 60  # Total duration in seconds (5 minutes)
phase_durations = [total_duration * 0.2] * 5  # Five phases of equal duration

# Calculate the start and end times for each phase
phase_starts = [sum(phase_durations[:i]) for i in range(len(phase_durations))]
phase_ends = [start + duration for start, duration in zip(phase_starts, phase_durations)]

# Blueish shades for the request rate regions
colors = ['#e0f3f8', '#a6bddb', '#3690c0', '#a6bddb', '#e0f3f8']  # Light to darker blues

# Labels for the regions and request rates
rate_texts = ['Low', 'Mid', 'High', 'Mid', 'Low']

# Shade the regions and add text annotations
for i, (start, end, color, rate_text) in enumerate(zip(phase_starts, phase_ends, colors, rate_texts)):
    ax1.axvspan(start, end, color=color, alpha=0.3)
    ax2.axvspan(start, end, color=color, alpha=0.3)

    # Position the text annotations slightly above the bottom of the y-axis
    ax1.text((start + end) / 2, ax1.get_ylim()[0]*0.63, rate_text,
             ha='center', va='bottom', fontsize=9, color='black')

# Adjust fonts for axes and ticks
ax1.tick_params(axis='both', which='major', labelsize=8)
ax2.tick_params(axis='both', which='major', labelsize=8)

# Align y-axis labels
fig.align_ylabels([ax1, ax2])

# Place the legend at the top center without a box on ax1
ax1.legend(loc='upper center', fontsize=9, frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.25))

# Adjust layout to make room for annotations and legend
plt.tight_layout()
plt.subplots_adjust(top=0.85, hspace=0.2)

# size between the two plots
plt.subplots_adjust(hspace=0.2)

# Save the plot as a high-quality PDF file
plt.savefig('token_latency_and_requests_over_time.pdf', bbox_inches='tight', format='pdf')

# Show the plot
plt.show()
