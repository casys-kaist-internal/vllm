import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
with open('accept_prob_temperature.csv', 'r') as file:
    lines = file.readlines()

# Initialize a dictionary to store temperatures and associated accept probabilities
result = {}

# Process each line in the file
for line in lines:
    parts = line.strip().split(',')

    if len(parts) < 2:
        continue

    # The first part is "result" (dummy), skip it
    temperature = parts[1].strip()  # The second float is temperature

    # The rest are accept probabilities
    accept_probs = list(map(float, parts[2:]))

    if temperature not in result:
        result[temperature] = []

    result[temperature].extend(accept_probs)

# Calculate the mean accept probability for each temperature
mean_accept_probs = {}
for temperature, accept_probs in result.items():
    mean_accept_probs[temperature] = sum(accept_probs) / len(accept_probs)

# Create a DataFrame
df = pd.DataFrame(list(mean_accept_probs.items()), columns=[
                  'Temperature', 'Mean Accept Probability'])

# Sort the DataFrame by Temperature to have a well-ordered histogram
df = df.sort_values('Temperature')

# Define the color scheme
colors = ['#FFFFFF', '#D7E2F9', '#88BCFF', '#3864B9', '#1B345F']
colors.reverse()


# Adjust the figure size for a two-column paper
plt.figure(figsize=(5, 2.3))  # Size in inches (width, height)

plt.grid(axis='y')  # Show grid lines on the y-axis


# Plot the data as a histogram (bar chart)
plt.bar(df['Temperature'], df['Mean Accept Probability'],
        color=colors, edgecolor='black')

plt.ylim(0, 1.0)  # Set the y-axis limits

# Set labels and title
plt.xlabel('Temperature', fontsize=10)
plt.ylabel('Mean Accept Probability', fontsize=10)

# Customize x-ticks and y-ticks
plt.xticks(df['Temperature'], fontsize=8)
plt.yticks(fontsize=8)

# Use a tight layout to fit everything within the small space
plt.tight_layout()

# Save the plot
plt.savefig('accept_prob_temperature.png', dpi=300,
            bbox_inches='tight', facecolor='#FFFFFF')

# Show the plot
plt.show()
