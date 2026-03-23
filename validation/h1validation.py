import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# 1. Load your GML
G = nx.read_gml("../graph_with_listeners.gml")

# 2. Compute Metrics
print("Calculating network metrics...")
degree = dict(G.degree())
strength = dict(G.degree(weight='weight'))
betweenness = nx.betweenness_centrality(G, weight='weight')
pagerank = nx.pagerank(G, weight='weight')
closeness = nx.closeness_centrality(G)

# 3. Build DataFrame
nodes_data = []
for node, attrs in G.nodes(data=True):
    nodes_data.append({
        'artist': attrs.get('label'),
        'popularity': attrs.get('monthly_listeners'),
        'degree': degree[node],
        'strength': strength[node],
        'betweenness': betweenness[node],
        'pagerank': pagerank[node],
        'closeness': closeness[node]
    })

df = pd.DataFrame(nodes_data)

# 4. Clean Data (IMPORTANT)
df = df.dropna(subset=['popularity'])   # remove missing
df = df[df['popularity'] > 0]           # remove invalid zeros

# 5. Log Transform Popularity (CRITICAL FIX)
df['log_popularity'] = np.log10(df['popularity'] + 1)

# 6. Correlation Function
def interpret_corr(rho):
    if abs(rho) < 0.2:
        return "very weak"
    elif abs(rho) < 0.4:
        return "weak"
    elif abs(rho) < 0.6:
        return "moderate"
    elif abs(rho) < 0.8:
        return "strong"
    else:
        return "very strong"

# 7. Compute Correlations
metrics = ['degree', 'strength', 'betweenness', 'pagerank', 'closeness']
results = []
p_values = []

for m in metrics:
    corr, p_value = spearmanr(df[m], df['log_popularity'], nan_policy='omit')
    
    results.append({
        'Metric': m,
        'Spearman_Rho': round(corr, 4),
        'P_Value': p_value
    })
    
    p_values.append(p_value)

# 8. Multiple Testing Correction (optional but strong)
_, corrected_p, _, _ = multipletests(p_values, method='bonferroni')

# 9. Final Results Formatting
for i, r in enumerate(results):
    r['P_Value'] = round(r['P_Value'], 10)
    r['Corrected_P'] = round(corrected_p[i], 10)
    r['Significant'] = corrected_p[i] < 0.05
    r['Effect_Size'] = interpret_corr(r['Spearman_Rho'])

results_df = pd.DataFrame(results)

# 10. Save Outputs
df.to_csv("h1_artist_network_metrics_cleaned.csv", index=False)
results_df.to_csv("h1_hypothesis_validation_results.csv", index=False)

# 11. Print Results
print("\n--- Correlation Results (log popularity) ---")
print(results_df)

plt.scatter(df['degree'], df['log_popularity'])
plt.xlabel("Degree")
plt.ylabel("Log Popularity")
plt.title("Degree vs Artist Popularity")
plt.show()