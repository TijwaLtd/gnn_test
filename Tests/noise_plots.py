import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your full dataset
df = pd.read_csv('noise_scores.csv')

# Create a figure with multiple subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Histogram of noise scores
ax1.hist(df['noise_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax1.set_xlabel('Noise Score')
ax1.set_ylabel('Number of Structures')
ax1.set_title('Distribution of Noise Scores')
ax1.grid(True, alpha=0.3)

# 2. Box plot
ax2.boxplot(df['noise_score'])
ax2.set_ylabel('Noise Score')
ax2.set_title('Box Plot of Noise Scores')
ax2.grid(True, alpha=0.3)

# 3. Scatter plot: Energy vs Force uncertainty
scatter = ax3.scatter(df['energy_std'], df['force_std'], 
                     c=df['noise_score'], cmap='viridis', alpha=0.6)
ax3.set_xlabel('Energy Standard Deviation')
ax3.set_ylabel('Force Standard Deviation')
ax3.set_title('Energy vs Force Uncertainty')
plt.colorbar(scatter, ax=ax3, label='Noise Score')

# 4. Cumulative distribution
sorted_scores = np.sort(df['noise_score'])
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
ax4.plot(sorted_scores, cumulative, linewidth=2, color='red')
ax4.set_xlabel('Noise Score')
ax4.set_ylabel('Cumulative Probability')
ax4.set_title('Cumulative Distribution of Noise Scores')
ax4.grid(True, alpha=0.3)

# Add statistics text
stats_text = f"""Dataset Statistics:
Total structures: {len(df)}
Mean noise score: {df['noise_score'].mean():.4f}
Median noise score: {df['noise_score'].median():.4f}
95th percentile: {df['noise_score'].quantile(0.95):.4f}
Max noise score: {df['noise_score'].max():.4f}"""

fig.text(0.02, 0.02, stats_text, fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

plt.tight_layout()
plt.savefig('noise_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print outlier analysis
q1, q3 = df['noise_score'].quantile([0.25, 0.75])
iqr = q3 - q1
outlier_threshold = q3 + 1.5 * iqr
outliers = df[df['noise_score'] > outlier_threshold]

print(f"\nOutlier Analysis:")
print(f"IQR outlier threshold: {outlier_threshold:.6f}")
print(f"Number of outliers: {len(outliers)}")
print(f"Percentage of outliers: {len(outliers)/len(df)*100:.2f}%")

if len(outliers) > 0:
    print(f"\nTop 5 noisiest structures:")
    print(outliers.nlargest(5, 'noise_score')[['structure_id', 'noise_score']])
