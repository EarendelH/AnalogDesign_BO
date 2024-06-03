# Import required libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Check if the Excel file path is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python script.py <excel_file_path>")
    sys.exit(1)

# Get the Excel file path from the command-line argument
excel_file_path = sys.argv[1]

# Read the input data from the Excel file
data = pd.read_excel(excel_file_path)

# Separate input and output features
X = data.iloc[:, 22:]  # Input features
y = data.iloc[:, :22]  # Output features

# Create instances of the three algorithms
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
multi_output_rf = MultiOutputRegressor(rf)

pls = PLSRegression(n_components=10)
multi_output_pls = MultiOutputRegressor(pls)

krr = KernelRidge(alpha=1.0)
multi_output_krr = MultiOutputRegressor(krr)

# Perform cross-validation for each algorithm
cv_scores_rf = cross_validate(multi_output_rf, X, y, cv=5, scoring='r2')
cv_scores_pls = cross_validate(multi_output_pls, X, y, cv=5, scoring='r2')
cv_scores_krr = cross_validate(multi_output_krr, X, y, cv=5, scoring='r2')

# Train the multi-output models on the input and output features
multi_output_rf.fit(X, y)
multi_output_pls.fit(X, y)
multi_output_krr.fit(X, y)

# Get the directory path of the Excel file
excel_dir = os.path.dirname(excel_file_path)

# Compute the overall feature importances for each algorithm
overall_feature_importances_rf = np.mean([estimator.feature_importances_ for estimator in multi_output_rf.estimators_], axis=0)
overall_feature_importances_pls = np.mean(np.abs(multi_output_pls.estimators_[0].coef_), axis=0)
overall_feature_importances_krr = np.mean(np.abs(multi_output_krr.estimators_[0].dual_coef_), axis=0)

# Sort the input features based on their overall importance for each algorithm
sorted_features_rf = sorted(zip(overall_feature_importances_rf, X.columns), reverse=True)
sorted_features_pls = sorted(zip(overall_feature_importances_pls, X.columns), reverse=True)
sorted_features_krr = sorted(zip(overall_feature_importances_krr, X.columns), reverse=True)

# Function to group features based on importance
def group_features(sorted_features):
    groups = []
    current_group = []

    for importance, feature in sorted_features:
        if current_group and abs(importance - current_group[0][0]) > 0.05:
            groups.append(current_group)
            current_group = []
        current_group.append((importance, feature))

    if current_group:
        groups.append(current_group)

    groups = [[feature for _, feature in group] for group in groups]
    return groups

# Group features for each algorithm
groups_rf = group_features(sorted_features_rf)
groups_pls = group_features(sorted_features_pls)
groups_krr = group_features(sorted_features_krr)

# Print the cross-validation scores and feature groupings for each algorithm
print("Random Forest Regression:")
print("Cross-validation R^2 scores:", cv_scores_rf['test_score'])
print("Feature Grouping:")
for i, group in enumerate(groups_rf):
    print(f"Group {i+1}: {', '.join(group)}")
print()

print("Partial Least Squares Regression:")
print("Cross-validation R^2 scores:", cv_scores_pls['test_score'])
print("Feature Grouping:")
for i, group in enumerate(groups_pls):
    print(f"Group {i+1}: {', '.join(group)}")
print()

print("Kernel Ridge Regression:")
print("Cross-validation R^2 scores:", cv_scores_krr['test_score'])
print("Feature Grouping:")
for i, group in enumerate(groups_krr):
    print(f"Group {i+1}: {', '.join(group)}")
print()

# Save the cross-validation scores and feature groupings as a table
with open(os.path.join(excel_dir, 'results.txt'), 'w') as file:
    file.write("Random Forest Regression:\n")
    file.write("Cross-validation R^2 scores: {}\n".format(cv_scores_rf['test_score']))
    file.write("Feature Grouping:\n")
    for i, group in enumerate(groups_rf):
        file.write(f"Group {i+1}: {', '.join(group)}\n")
    file.write("\n")

    file.write("Partial Least Squares Regression:\n")
    file.write("Cross-validation R^2 scores: {}\n".format(cv_scores_pls['test_score']))
    file.write("Feature Grouping:\n")
    for i, group in enumerate(groups_pls):
        file.write(f"Group {i+1}: {', '.join(group)}\n")
    file.write("\n")

    file.write("Kernel Ridge Regression:\n")
    file.write("Cross-validation R^2 scores: {}\n".format(cv_scores_krr['test_score']))
    file.write("Feature Grouping:\n")
    for i, group in enumerate(groups_krr):
        file.write(f"Group {i+1}: {', '.join(group)}\n")
    file.write("\n")

# Plot the overall feature importances for each algorithm
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

for ax, sorted_features, title in zip(axs, [sorted_features_rf, sorted_features_pls, sorted_features_krr],
                                      ['Random Forest', 'Partial Least Squares', 'Kernel Ridge']):
    x_pos = range(len(sorted_features))
    importances, labels = zip(*sorted_features)
    ax.bar(x_pos, importances, align='center')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title(f'{title} Overall Feature Importances')
    ax.tick_params(axis='x', labelsize=8)

plt.tight_layout()
plot_file_path = os.path.join(excel_dir, 'overall_feature_importances.png')
plt.savefig(plot_file_path)