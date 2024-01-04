import plotly.express as px
import pandas as pd
import numpy as np

# Read data from profile_results.csv
with open('output.csv', 'r') as f:
    data = f.read()

# Skip the first line
data = data[data.find('\n') + 1:]

# Split the data by newline to get each row
rows = data.split('\n')

# For each row, split by comma and strip whitespace, then skip the first element
data = [row.split(',')[1:] for row in rows if row]

# Remove row with length that is not 3
data = [row for row in data if len(row) == 2]
# data = [row for row in data if len(row) == 3]

# Convert to DataFrame
df = pd.DataFrame(
    data, columns=['Number of Input Tokens', 'Latency'])
df['Number of Input Tokens'] = df['Number of Input Tokens'].astype(int)
df['Latency'] = df['Latency'].astype(float)


# dataframe remove outlier that is above 3 sigma
df = df[np.abs(df.Latency - df.Latency.mean()) <= (3 * df.Latency.std())]
df = df[df.Latency - df.Latency.mean() <= (3 * df.Latency.std())]

# Create a 2D scatter plot with Number of Input Tokens and Latency but with

fig = px.scatter(df, x='Number of Input Tokens', y='Latency', color='Latency')

# Update the title
fig.update_layout(title='Latency vs. Number of Input Tokens')

# Show the plot
fig.write_html("profile_input_tokens_prompt.html")

# # Convert to DataFrame
# df = pd.DataFrame(
#     data, columns=['Number of Input Tokens', 'Sum of Context Length', 'Latency'])
# df['Number of Input Tokens'] = df['Number of Input Tokens'].astype(int)
# df['Sum of Context Length'] = df['Sum of Context Length'].astype(int)
# df['Latency'] = df['Latency'].astype(float)

# # dataframe remove outlier that is above 3 sigma
# # df = df[np.abs(df.Latency - df.Latency.mean()) <= (3 * df.Latency.std())]
# # df = df[df.Latency - df.Latency.mean() <= (3 * df.Latency.std())]

# # remove that is not monotonic increase in number of input tokens
# df = df[df['Number of Input Tokens'].diff() >= 0]

# # Create a 3D scatter plot using Plotly
# fig = px.scatter_3d(df, x='Number of Input Tokens',
#                     y='Sum of Context Length', z='Latency', color='Latency')
# fig.update_traces(marker_size=1)

# # Update the title
# fig.update_layout(
#     title='Latency vs. Number of Input Tokens and Sum of Context Length')

# # Show the plot
# fig.write_html("profile_total.html")

# # Create a 2D scatter plot with Number of Input Tokens and Latency but with

# fig = px.scatter(df, x='Number of Input Tokens', y='Latency', color='Latency')

# # Update the title
# fig.update_layout(title='Latency vs. Number of Input Tokens')

# # Show the plot
# fig.write_html("profile_input_tokens.html")

# # Create a 2D scatter plot with Sum of Context Length and Latency
# fig = px.scatter(df, x='Sum of Context Length',
#                  y='Latency', color='Latency')

# # Update the title
# fig.update_layout(title='Latency vs. Sum of Context Length')

# # Show the plot
# fig.write_html("profile_context_length.html")
