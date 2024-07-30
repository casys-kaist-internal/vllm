import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the data from prob.txt
data = []
with open('prob.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) == 3:
            label = parts[1].strip()
            prob = float(parts[2].strip())
            data.append((label, prob))

# Step 2: Separate the data into accept and reject categories
accept_probs = [prob for label, prob in data if label == 'accept']
reject_probs = [prob for label, prob in data if label == 'reject']

# Step 3: Prepare data for seaborn violin plot
category = ['Reject'] * len(reject_probs) + ['Accept'] * len(accept_probs)
probabilities = reject_probs + accept_probs

# Step 4: Plot the data using a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=category, y=probabilities, palette={
               'Accept': 'green', 'Reject': 'red'})

# Customize the plot
plt.xlabel('Category')
plt.ylabel('Predicted Probability')
plt.title('Violin Plot of Predicted Probabilities for Acceptance and Rejection')
plt.grid(True)
plt.tight_layout()

# Save and show the plot
plt.savefig('accept_reject_violinplot.png')
plt.show()
