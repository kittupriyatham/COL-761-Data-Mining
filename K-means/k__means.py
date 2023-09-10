import sys
import pandas as pd
import random
import numpy as np


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



for n in range(5,6):
    centres=df.sample(n)
    print(centres)

    for i in range(5):
        centres.reset_index(drop=True, inplace=True)
        # Create a dictionary to store data points closest to each random point
        closest_points = {i: [] for i in range(n)}
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
        print(centres)
    
    # print(centres)
# print(items)






# print(transaction,sys)