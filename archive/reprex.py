import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a sample DataFrame
data = pd.DataFrame({
    'dollar_value': [100, 200, 300],
    'conv_rate': [0.5, 0.6, 0.7],
    'growth_rate': [0.2, 0.4, 0.6]
})

# Reshape the data for the heatmap
heatmap_data = data.pivot("conv_rate", "growth_rate", "dollar_value")

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlGnBu')

# Set the labels for each value
for i in range(len(data)):
    plt.text(i + 0.5, 0.5, f"${data['dollar_value'][i]}\nCR: {data['conv_rate'][i]}\nGR: {data['growth_rate'][i]}",
             horizontalalignment='center', verticalalignment='center', fontsize=10)

# Set the axis labels and title
plt.xlabel('Growth Rate')
plt.ylabel('Conversion Rate')
plt.title('Heatmap with Dollar Value, Conversion Rate, and Growth Rate')

# Display the heatmap
plt.show()
