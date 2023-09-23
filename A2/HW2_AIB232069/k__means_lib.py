import sys
from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

data=[]
dimension = int(sys.argv[2])
# inputPath=sys.argv[1]
# column_names = [f'feature{i}' for i in range(1, dimension + 1)]
# data = pd.read_csv(sys.argv[1], delimiter=' ', header=None, names=column_names)
data = pd.read_csv(sys.argv[1], delimiter=' ', header=None)
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data)
# print(data)

mean_distances = []

for i in range(1,16):
    kmeans = KMeans(n_clusters=i,n_init=20, random_state=42)
    kmeans.fit(data)
    # kmeans.fit(scaled_data)
    distances = kmeans.transform(data)
    # distances = kmeans.transform(scaled_data)
    mean_distance = distances.min(axis=1).mean()
    mean_distances.append(mean_distance)

elbow_point = None
for i in range(1, 15):
    if (mean_distances[i] - mean_distances[i+1]) < 0.1 * (mean_distances[i-1] - mean_distances[i]):
        elbow_point = i+1
        break
    

plt.plot(range(1,16), mean_distances, marker='o')
plt.title('K vs Mean Distance from the cluster\'s centre\n(For the data '+str(sys.argv[1])+')')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Mean Distance \n(Of the points to their cluster\'s centre)')

if elbow_point is not None:
    plt.annotate(f'Elbow Point (at Optimal k = {elbow_point})', (elbow_point, mean_distances[elbow_point - 1]), textcoords="offset points", xytext=(-15, 10), ha='center')
    print(f"The optimal k (elbow point) is at k = {elbow_point}")
    print("Mean distance from cluster=",mean_distances[elbow_point-1])
    # plt.annotate(f'Optimal k (Elbow Point) = {elbow_point}', (elbow_point, 0), textcoords="offset points", xytext=(0, 30), ha='center')
else:
    print("The elbow point could not be determined.")

# plt.savefig('K_means'+str(dimension)+'D_plot.png')
plt.savefig(sys.argv[3])
# plt.show()

