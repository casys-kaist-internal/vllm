import matplotlib.pyplot as plt

# Dummy data for demonstration
index = list(range(1, 11))  # Replace with your actual x-axis values if needed
beta_values = [0.5, 0.6, 0.7, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]  # Example beta values
accept_rate_values = [0.45, 0.55, 0.68, 0.60, 0.77, 0.82, 0.88, 0.93, 0.98, 1.05]  # Example accept_rate values

# Creating the dot graph
plt.figure(figsize=(10, 6))  # Set the figure size (width, height) in inches

# Plot beta values
plt.scatter(index, beta_values, color='blue', label='Beta')

# Plot accept_rate values
plt.scatter(index, accept_rate_values, color='red', label='Accept Rate')

# Adding titles and labels
plt.title('Beta vs. Accept Rate')
plt.xlabel('Index')  # Adjust as per your actual x-axis label
plt.ylabel('Values')
plt.legend()

# Show grid
plt.grid(True)

# Display the plot
plt.show()
