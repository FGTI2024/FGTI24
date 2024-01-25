import numpy as np


# Download traffic_speed_data.csv in to "./Data" from https://zenodo.org/record/1205229


data = np.loadtxt("Data/traffic_speed_data.csv", delimiter=",")
miss_p = np.where(data == -200)
no_missing = []
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i,j] == -200:
            continue
        no_missing.append(data[i,j])

no_missing = np.array(no_missing)

# all attribute from same domain
mean = np.mean(no_missing)
std = np.std(no_missing)
data = (data-mean) / std
data[miss_p] = -200

np.savetxt("Data/Guangzhou_norm.csv",data, delimiter=",",fmt="%6f")