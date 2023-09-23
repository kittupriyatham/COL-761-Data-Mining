import sys
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import time


# def centroid:
#     print()

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))


start=time.time()
epsilon=1e-10
transaction=[]
items=[]
dimensions=int(sys.argv[2])
column_names = [f'col{i}' for i in range(1, dimensions + 1)]
df = pd.read_csv(sys.argv[1], delimiter=' ', header=None, names=column_names)
# print(type(df['col1'].iloc[0]))

min__mean_distance=float('inf')
plot_points=np.empty((0,2))
for k in range(1,16):
    check=1
    while check==1:
        centres=df.sample(k)
        # print(centres)
        closest_points={}
        centre_diff=0.001
        centre_change=float('inf')
        x=0
        while centre_change>centre_diff and x<15:
            x+=1
            centres.reset_index(drop=True, inplace=True)
            prev_centres=centres.copy()
            closest_points = {i: [] for i in range(k)}
            for index, row in df.iterrows():
                closest_centre = None
                min_distance = float('inf')

                for i, centre in centres.iterrows():
                    distance = euclidean_distance(row.values, centre.values)
                    if distance < min_distance:
                        min_distance = distance
                        closest_centre = i
            
                closest_points[closest_centre].append({'closest_centre': centres.loc[closest_centre], 'data_point': row})
            
            for centre_index, points in closest_points.items():
                if len(points) > 0:
                    centroid = pd.DataFrame([point['data_point'] for point in points]).mean()
                    centres.loc[centre_index] = centroid
            # if(k==5): print(centres)
            
            center_change = np.max(np.abs((centres.values - prev_centres.values)/(prev_centres.values+epsilon)))
            

        total_distance = 0
        total_points = 0
        for centre_index, points in closest_points.items():
            if len(points) > 0:
                total_distance += sum([euclidean_distance(point['data_point'].values, centres.loc[centre_index].values) for point in points])
                total_points += len(points)
        mean_distance = total_distance / total_points
        if min__mean_distance>mean_distance:
            min__mean_distance=mean_distance
            check=0
            new_point=np.array([k,mean_distance])
            plot_points = np.append(plot_points, [new_point], axis=0)
 

print(plot_points)
    # print(centres)
# print(items)
        
# print(transaction,sys)

plt.plot(plot_points[:, 0], plot_points[:, 1], marker='o', color='b', label='Points')
plt.title('K vs Mean Distance \n ( For data '+str(sys.argv[1])+')')
plt.xlabel('Number of clusters')
plt.ylabel('Mean distance from centroid')
plt.legend()

print(time.time()-start)
plt.show()
# print(plot_points)
