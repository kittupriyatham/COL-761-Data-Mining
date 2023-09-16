import sys
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# def centroid:
#     print()


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

transaction=[]
items=[]
dimensions=int(sys.argv[2])
column_names = [f'col{i}' for i in range(1, dimensions + 1)]
df = pd.read_csv(sys.argv[1], delimiter=' ', header=None, names=column_names)
# print(type(df['col1'].iloc[0]))
plot_points=np.empty((0,2))


for k in range(1,16):
    centres=df.sample(k)
    # print(centres)
    closest_points={}
    for i in range(20):
        centres.reset_index(drop=True, inplace=True)
        # Create a dictionary to store data points closest to each random point
        closest_points = {i: [] for i in range(k)}
        for index, row in df.iterrows():
            closest_centre = None
            min_distance = float('inf')

            # Calculate the distance to each centre and find the closest one
            for i, centre in centres.iterrows():
                distance = euclidean_distance(row.values, centre.values)
                if distance < min_distance:
                    min_distance = distance
                    closest_centre = i
        
        # Append a dictionary containing both the closest_centre and the data point
            closest_points[closest_centre].append({'closest_centre': centres.loc[closest_centre], 'data_point': row})
        
        # Update the centres to the centroids of the points in closest_points
        for centre_index, points in closest_points.items():
            if len(points) > 0:
                centroid = pd.DataFrame([point['data_point'] for point in points]).mean()
                centres.loc[centre_index] = centroid
        # Print the dictionary where keys are random points, and values are lists of dictionaries containing closest_centre and data_point
        # if(k==5): print(centres)

    # Calculate the mean distance for all points collectively
    total_distance = 0
    total_points = 0
    for centre_index, points in closest_points.items():
        if len(points) > 0:
            total_distance += sum([euclidean_distance(point['data_point'].values, centres.loc[centre_index].values) for point in points])
            total_points += len(points)

    mean_distance = total_distance / total_points
    new_point=np.array([k,mean_distance])
    plot_points = np.append(plot_points, [new_point], axis=0)
 

 
    # print(centres)
# print(items)

# print(transaction,sys)

plt.scatter(plot_points[:, 0], plot_points[:, 1], marker='o', color='b', label='Points')
plt.xlabel('Number of clusters')
plt.ylabel('Mean distance from centroid')
plt.legend()
plt.show()