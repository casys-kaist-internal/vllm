import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
from sklearn.linear_model import LinearRegression

# Function to read and process a tokens file


def read_tokens_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    for i in range(0, len(lines), 2):
        tokens_line = lines[i].split(',')
        if len(tokens_line) < 6:
            continue
        time_line = float(lines[i + 1].strip().split(',')[1])

        # remove tokens more than 512
        if int(tokens_line[2]) > 512:
            continue

        total_tokens = int(tokens_line[2])
        prefill = int(tokens_line[3])
        base = int(tokens_line[4])
        spec = int(tokens_line[5])
        elapsed_time = time_line

        data.append([total_tokens, prefill, base, spec, elapsed_time])

    df = pd.DataFrame(
        data, columns=['total_tokens', 'prefill', 'base', 'spec', 'elapsed_time'])

    return df

# Function to fit linear regression for each segment of 64 token values with the same intercept


def fit_and_plot_segments(df, label, color):
    max_tokens = df['total_tokens'].max()
    segments = range(0, max_tokens + 64, 64)

    plt.scatter(df['total_tokens'], df['elapsed_time'],
                label=f'{label} data', s=5, color=color)

    # Determine the intercept from the first segment
    first_segment = df[(df['total_tokens'] >= 0) & (df['total_tokens'] < 64)]
    slope_first, intercept_first, _, _, _ = linregress(
        first_segment['total_tokens'], first_segment['elapsed_time'])
    plt.axhline(y=intercept_first, color='gray', linestyle='--')
    plt.text(-30, intercept_first,
             f"Intercept: {intercept_first:.2f}", color='black', ha='right')
    print(
        f"First Segment 0-64: Intercept = {intercept_first:.2f}, Slope = {slope_first:.2f}")

    slopes = []

    for start in segments:
        end = start + 64
        df_segment = df[(df['total_tokens'] >= start)
                        & (df['total_tokens'] < end)]
        if len(df_segment['total_tokens'].unique()) > 1:
            # Fit linear regression with fixed intercept
            X = df_segment['total_tokens'].values.reshape(-1, 1)
            y = df_segment['elapsed_time'].values
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X, y - intercept_first)
            slope = reg.coef_[0]
            slopes.append(slope)
            plt.plot(df_segment['total_tokens'], intercept_first + slope *
                     df_segment['total_tokens'], color=color, linestyle='--')
            plt.text(start + 32, intercept_first + slope * (start + 32) +
                     5, f"Slope: {slope:.2f}", color=color, ha='center')
            print(f"Segment {start}-{end}: Slope = {slope:.2f}")

    # if slopes:
    #     std_dev = np.std(slopes)
    #     print(f"Standard Deviation of Slopes: {std_dev:.2f}")
    #     plt.text(max_tokens - 200, intercept_first + slope_first * (max_tokens -
    #              200) + 50, f"Std Dev: {std_dev:.2f}", color=color, ha='center')


# Read and process the files
df1 = read_tokens_file('tokens_1_process.txt')
df2 = read_tokens_file('tokens_2_process.txt')

# Plot the data with linear fits
plt.figure(figsize=(14, 8))

fit_and_plot_segments(df1, '1 Process', 'blue')
fit_and_plot_segments(df2, '2 Process', 'green')

# show plot y axis from 0
plt.ylim(bottom=0)
plt.xlim(left=0)

plt.xlabel('Total Tokens')
plt.ylabel('Elapsed Time (ms)')
plt.title('Elapsed Time vs Total Tokens with Linear Fit Segments')
plt.legend()

# Add vertical lines at every 64 tokens
max_tokens = max(df1['total_tokens'].max(), df2['total_tokens'].max())
for x in range(64, max_tokens + 64, 64):
    plt.axvline(x=x, color='red', linestyle='--', linewidth=1)

plt.grid(False)
plt.savefig('plot_tokens_memory_compute_bound_segments.png')
plt.show()
