import pandas as pd
import numpy as np

data = pd.read_csv("Data/KDD.csv",delimiter=",", header=0).to_numpy()

list = []
for i in range(9):
    #remove stationId and utc_time
    station_record = data[:, i*13+2: (i+1)*13]
    list.append(station_record)
data = np.stack(list, axis=1).reshape(data.shape[0], -1)

means, stds = [], []
for j in range(data.shape[1]):
    data_j = []
    for i in range(data.shape[0]):
        if data[i,j] == -200:
            continue
        data_j.append(data[i,j])
    data_j = np.array(data_j)
    mean_j = np.mean(data_j)
    std_j = np.std(data_j)

    for i in range(data.shape[0]):
        if data[i,j] == -200:
            continue
        data[i,j] = (data[i,j] - mean_j) / std_j
    means.append(mean_j)
    stds.append(std_j)
    
np.savetxt("Data/KDD_norm.csv",data, delimiter=",",fmt="%6f")