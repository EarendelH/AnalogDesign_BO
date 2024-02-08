from joblib import load
import pandas as pd

# Load data from joblib file
aggregated_metrics = load("/Users/hanwu/Downloads/sum.joblib")

# Convert dictionaries to DataFrame for easier manipulation
df = pd.DataFrame(aggregated_metrics)

# Remove rows where phaseMargin and gainBandWidth are 0, or pwr is 1
df_cleaned = df[(df['phaseMargin'] != 0) & (df['gainBandWidth'] != 0) & (df['pwr'] != 1)]

# Count the number of rows (data groups) before cleaning
num_original_data_groups = df.shape[0]

# Count the number of rows (data groups) after cleaning
num_cleaned_data_groups = df_cleaned.shape[0]

# Print the number of original and cleaned data groups for comparison
print(f"Original data groups: {num_original_data_groups}")
print(f"Cleaned data groups: {num_cleaned_data_groups}")

import matplotlib.pyplot as plt

# 创建一个图形框架，并设置子图的布局
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 绘制每个参数的直方图
axes[0].hist(df_cleaned['phaseMargin'], bins=10, color='skyblue', edgecolor='black')
axes[0].set_title('Phase Margin Distribution')

axes[1].hist(df_cleaned['gainBandWidth'], bins=10, color='lightgreen', edgecolor='black')
axes[1].set_title('Gain BandWidth Distribution')

axes[2].hist(df_cleaned['pwr'], bins=10, color='salmon', edgecolor='black')
axes[2].set_title('Pwr Distribution')

# 设置每个直方图的标签和标题
for ax in axes:
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

# 显示图形
plt.tight_layout()
plt.show()