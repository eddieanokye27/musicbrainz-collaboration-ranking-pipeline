import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# 1. Load graph
G = nx.read_gml("../graph_with_listeners.gml")

# 2. Compute metrics
print("Calculating metrics...")
degree = dict(G.degree())
strength = dict(G.degree(weight='weight'))

# 3. Build DataFrame
nodes_data = []
for node, attrs in G.nodes(data=True):
    nodes_data.append({
        'artist': attrs.get('label'),
        'popularity': attrs.get('monthly_listeners'),
        'degree': degree[node],
        'strength': strength[node]
    })

df = pd.DataFrame(nodes_data)

# 4. Clean data
df = df.dropna(subset=['popularity'])
df = df[df['popularity'] > 0]

# 5. Log-transform popularity (CRITICAL)
df['log_popularity'] = np.log10(df['popularity'] + 1)

# -------------------------------
# PART A: Correlation Comparison
# -------------------------------
corr_deg, p_deg = spearmanr(df['degree'], df['log_popularity'])
corr_str, p_str = spearmanr(df['strength'], df['log_popularity'])

corr_results = pd.DataFrame([
    {"Metric": "Degree", "Spearman_Rho": round(corr_deg, 4), "P_Value": round(p_deg, 6)},
    {"Metric": "Strength", "Spearman_Rho": round(corr_str, 4), "P_Value": round(p_str, 6)}
])

print("\n--- Correlation Comparison ---")
print(corr_results)

# -------------------------------
# PART B: Multicollinearity Check
# -------------------------------
corr_matrix = df[['degree', 'strength']].corr()
print("\n--- Degree vs Strength Correlation ---")
print(corr_matrix)

# -------------------------------
# PART C: Multiple Regression
# -------------------------------
X = df[['degree', 'strength']]
X = sm.add_constant(X)
y = df['log_popularity']

model = sm.OLS(y, X).fit()

print("\n--- Regression Results (Raw) ---")
print(model.summary())

# -------------------------------
# PART D: Standardized Regression
# (for fair coefficient comparison)
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['degree', 'strength']])
X_scaled = sm.add_constant(X_scaled)

model_std = sm.OLS(y, X_scaled).fit()

print("\n--- Regression Results (Standardized) ---")
print(model_std.summary())

# -------------------------------
# PART E: Extract Key Results
# -------------------------------
results_summary = pd.DataFrame({
    "Variable": ["Degree", "Strength"],
    "Coefficient (Standardized)": [
        round(model_std.params[1], 4),
        round(model_std.params[2], 4)
    ],
    "P_Value": [
        round(model_std.pvalues[1], 6),
        round(model_std.pvalues[2], 6)
    ],
    "Significant (p<0.05)": [
        model_std.pvalues[1] < 0.05,
        model_std.pvalues[2] < 0.05
    ]
})

print("\n--- H2 Summary ---")
print(results_summary)

# -------------------------------
# Save outputs
# -------------------------------
df.to_csv("h2_cleaned_data.csv", index=False)
corr_results.to_csv("h2_correlation_results.csv", index=False)
results_summary.to_csv("h2_regression_summary.csv", index=False)

print("\nResults saved to CSV files.")