import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Read the data from a CSV file or any other source as needed
data = pd.read_csv(sys.argv[1], delimiter=' ', header=None)
num_features = int(sys.argv[2])

# Perform hierarchical clustering with "centroid" linkage
linked = linkage(data, 'complete')  # Use 'centroid' linkage method

# Create and display the dendrogram
# plt.figure(figsize=(12, 8))
dendrogram(linked, orientation='top', labels=list(range(1, len(data) + 1)), distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Centroid Linkage)')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.savefig('Dendogram.png')
plt.show()