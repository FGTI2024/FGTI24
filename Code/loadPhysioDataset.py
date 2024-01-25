import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset
from sklearn.preprocessing import StandardScaler

data = load_specific_dataset('physionet_2012')

X = data['X']
y = data["y"]

num_samples = len(X['RecordID'].unique())


train_set_idx = []
test_set_idx = []
labels = []
for num in range(num_samples):
    data = X[num*48: (num+1)*48]
    
    dataid = data["RecordID"].iloc[0]

    label = y.loc[dataid]["In-hospital_death"]
    
    labels.append(label)

X = X.drop(['RecordID', 'Time'], axis = 1)
X = StandardScaler().fit_transform(X.to_numpy())
X[np.isnan(X)] = -200

np.savetxt("Data/Physio_norm.csv", X, fmt="%.6f", delimiter=",")
