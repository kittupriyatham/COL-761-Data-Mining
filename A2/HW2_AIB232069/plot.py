#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('execution_times.csv')
algo_data = {}
for i, row in data.iterrows():
    algo, support, time = row['Algorithm Name'], row['Support Value'], row['Execution Time (seconds)']
    if algo not in algo_data:
        algo_data[algo] = {"support": [], "time": []}
    algo_data[algo]["support"].append(support)
    algo_data[algo]["time"].append(time)
for algo, data in algo_data.items():
    plt.plot(data["support"], data["time"], marker='o', label=algo)
plt.xlabel("Support Value")
plt.ylabel("Execution Time (sec)")
plt.legend()
plt.title("Min Support vs Execution Time for Graph mining Algos")
plt.grid(True)
plt.savefig('output_plot.png')
plt.close()