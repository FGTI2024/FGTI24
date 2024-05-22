
import torch
import numpy as np
import random

random.seed(3407)
np.random.seed(3407)
missing_p = 0.4
a = np.loadtxt("Data/Physio_norm.csv", delimiter=",")

mask_org = np.ones_like(a)
mask_org[np.where(a==-200)] = 0

x = a.shape[0]
y = a.shape[1]
print(np.sum(mask_org) / (x*y))

mask_target = mask_org.copy()

def ready():
    all_missing = x*y - np.sum(mask_target)
    target_missing = x*y - np.sum(mask_org)
    if (all_missing - target_missing) / (x*y) < missing_p:
        return True
    else:
        return False


while ready() == True:
    i = random.randint(0, x-1)
    j = random.randint(0, y-1)    
    mask_target[i,j] = 0

print(np.sum(mask_target) / (x*y))
np.savetxt("Data/mask/physio/physio_" + str(missing_p) + ".csv", mask_target, fmt="%d", delimiter=",")
