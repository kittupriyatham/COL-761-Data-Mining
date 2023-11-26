import numpy as np
import time
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def generate_random_points_total(n, m, d):
    points = np.round(np.random.uniform(0, 1, size=(n + m, d)), 5)
    return points[:n], points[n:]


# Example usage:
n = 1000000  # Number of points
m = 100  # Number of query points
dim = [1, 2, 4, 8, 16, 32, 64]  # Number of dimensions
l1, l2, linf = [], [], []

c = 1
for d in dim:
    start = time.time()
    points, query_points = generate_random_points_total(n, m, d)
    # print(type(points))
    # ratios = {1: [], 2: [], np.inf: []}
    distances_l1 = cdist(query_points, points, metric='minkowski')
    farthest_distances_l1 = np.max(distances_l1, axis=1)
    distances_l1_copy = np.copy(distances_l1)
    distances_l1_copy[distances_l1_copy == 0] = np.inf
    nearest_distances_l1 = np.min(distances_l1_copy, axis=1)
    ratio_l1 = farthest_distances_l1 / nearest_distances_l1
    l1.append(sum(ratio_l1) / len(ratio_l1))

    distances_l2 = cdist(query_points, points, metric='euclidean')
    farthest_distances_l2 = np.max(distances_l2, axis=1)
    distances_l2_copy = np.copy(distances_l2)
    distances_l2_copy[distances_l2_copy == 0] = np.inf
    nearest_distances_l2 = np.min(distances_l2_copy, axis=1)
    ratio_l2 = farthest_distances_l2 / nearest_distances_l2
    l2.append(sum(ratio_l2) / len(ratio_l2))

    distances_linf = cdist(query_points, points, metric='chebyshev')
    farthest_distances_linf = np.max(distances_linf, axis=1)
    distances_linf_copy = np.copy(distances_linf)
    distances_linf_copy[distances_linf_copy == 0] = np.inf
    nearest_distances_linf = np.min(distances_linf_copy, axis=1)
    ratio_linf = farthest_distances_linf / nearest_distances_linf
    linf.append(sum(ratio_linf) / len(ratio_linf))

    print(d, time.time() - start)

x_axis = dim
x_axis = np.log2(x_axis)
log_l1 = np.log10(l1)
log_l2 = np.log10(l2)
log_linf = np.log10(linf)

# Plot for l1
plt.figure(figsize=(6, 5))
plt.plot(dim, l1, label='L1', linewidth=0.8)
plt.plot(dim, l2, label='L2', linewidth=0.8)
plt.plot(dim, linf, label='L∞', linewidth=0.8)
plt.xlabel('Points dimension')
plt.ylabel('Average ratio of Distances')
plt.title('Average distances\' ratio ')
plt.legend()
plt.savefig('./plots/Ratio_1.png')
plt.close()

# Plot for l2
plt.figure(figsize=(8, 6))
plt.plot(x_axis, l1, label='L1', linewidth=0.8)
plt.plot(x_axis, l2, label='L2', linewidth=0.8)
plt.plot(x_axis, linf, label='L∞', linewidth=0.8)
plt.xlabel('Points dimension')
plt.ylabel('Average ratio of Distances')
plt.title('Average distances\' ratio')
plt.legend()
plt.savefig('./plots/Ratio_2.png')
plt.close()

# Plot for l1
plt.figure(figsize=(6, 4))
plt.plot(dim, l1, label='L1')
plt.xlabel('Points dimension')
plt.ylabel('Average ratio of Distances')
plt.title('Average distances\' ratio (L1)')
plt.legend()
plt.savefig('./plots/plot_l1.png')
plt.close()

# Plot for l2
plt.figure(figsize=(6, 5))
plt.plot(dim, l2, label='L2')
plt.xlabel('Points dimension')
plt.ylabel('Average ratio of Distances')
plt.title('Average distances\' ratio (L2)')
plt.legend()
plt.savefig('./plots/plot_l2.png')
plt.close()

# Plot for linf
plt.figure(figsize=(6, 5))
plt.plot(dim, linf, label='L∞')
plt.xlabel('Points dimension')
plt.ylabel('Average ratio of Distances')
plt.title('Average distances\' ratio (L∞)')
plt.legend()
plt.savefig('./plots/plot_linf.png')
plt.close()

# Plot for l1
plt.figure(figsize=(7, 5))
plt.plot(x_axis, l1, label='L1')
plt.xlabel('Log(Points dimension)')
plt.ylabel('Average ratio of Distances')
plt.title('Average distances\' ratio (L1)')
plt.legend()
plt.savefig('./plots/plot_l1_log.png')
plt.close()

# Plot for l2
plt.figure(figsize=(7, 5))
plt.plot(x_axis, l2, label='L2')
plt.xlabel('Log(Points dimension)')
plt.ylabel('Average ratio of Distances')
plt.title('Average distances\' ratio (L2)')
plt.legend()
plt.savefig('./plots/plot_l2_log.png')
plt.close()

# Plot for linf
plt.figure(figsize=(7, 5))
plt.plot(x_axis, linf, label='L∞')
plt.xlabel('Log(Points dimension)')
plt.ylabel('Average ratio of Distances')
plt.title('Average distances\' ratio (L∞)')
plt.legend()
plt.savefig('./plots/plot_linf_log.png')
plt.close()

dim = [round(x, 5) for x in dim]
l1 = [round(x, 5) for x in l1]
l2 = [round(x, 5) for x in l2]
linf = [round(x, 5) for x in linf]

# Print table header
print(f"{'Dimension':<15}{'L1':<15}{'L2':<15}{'L∞':<15}")

# Print table rows
for i in range(len(dim)):
    print(f"{dim[i]:<15}{l1[i]:<15}{l2[i]:<15}{linf[i]:<15}")
