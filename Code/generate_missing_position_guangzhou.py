
import torch
import numpy as np
import random

missing_ps = [0.1, 0.2, 0.3, 0.4]
seeds = [3407,3408,3409,3410,3411]

for missing_p in missing_ps:
    for seed in seeds:

        random.seed(seed)
        np.random.seed(seed)
        a = np.loadtxt("Data/Guangzhou_norm.csv", delimiter=",")

        mask_org = np.ones_like(a)
        mask_org[np.where(a==-200)] = 0

        x = a.shape[0]
        y = a.shape[1]

        mask_target = mask_org.copy()

        missing_sum = 0
        missing_target_sum = np.sum(mask_org) * missing_p
        while missing_sum <= missing_target_sum:
            
            i = random.randint(0, x-1)
            j = random.randint(0, y-1)    
            if mask_target[i,j] == 0:
                continue
            mask_target[i,j] = 0
            missing_sum += 1

        np.savetxt("Data/mask/guangzhou/guangzhou_" + str(missing_p) + "_" + str(seed) + ".csv", mask_target, fmt="%d", delimiter=",")
