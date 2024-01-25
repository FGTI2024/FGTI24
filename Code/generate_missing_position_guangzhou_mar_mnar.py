
import torch
import numpy as np
import random

missing_p = 0.1
seeds = [3407,3408,3409,3410,3411]

def get_MAR_mask_flag(org_data, seed):
    random.seed(seed)
    np.random.seed(seed)

    mask_flag = np.ones_like(org_data)
    mask_flag[np.where(org_data == -200)] = 0

    missing_sum_target = 0.1 * np.sum(mask_flag)
    time_step_num = org_data.shape[0]
    
    # based on first sensor
    attribute_data = org_data[:, 0]
    index = np.argsort(attribute_data) 
    rank = np.argsort(index) + 1 
    rank_sum = np.sum(rank)
    probability = rank / rank_sum
    
    missing_sum = 0
    while missing_sum <= missing_sum_target:
        attr = random.randint(0, org_data.shape[1] - 1)

        x = np.random.choice(range(time_step_num), p = probability.ravel())

        if mask_flag[x, attr] == 0:
            continue

        mask_flag[x, attr] = 0
        missing_sum += 1
    
    return mask_flag

def get_MNAR_mask_flag(org_data, seed):
    random.seed(seed)
    np.random.seed(seed)

    mask_flag = np.ones_like(org_data)
    mask_flag[np.where(org_data == -200)] = 0

    missing_sum_target = 0.1 * np.sum(mask_flag)
    time_step_num = org_data.shape[0]

    missing_sum = 0
    while missing_sum <= missing_sum_target:
        attr = random.randint(0, org_data.shape[1] - 1)

        attribute_data = org_data[:,attr]
        index = np.argsort(attribute_data) 
        rank = np.argsort(index) + 1 
        rank_sum = np.sum(rank)
        probability = rank / rank_sum

        x = np.random.choice(range(time_step_num), p = probability.ravel())

        if mask_flag[x, attr] == 0:
            continue

        mask_flag[x, attr] = 0
        missing_sum += 1

    return mask_flag

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    a = np.loadtxt("Data/Guangzhou_norm.csv", delimiter=",")

    mar_mask = get_MAR_mask_flag(org_data=a, seed=seed)
    mnar_mask = get_MNAR_mask_flag(org_data=a, seed=seed)

    np.savetxt("Data/mask/guangzhou/guangzhoumar_" + str(missing_p) + "_" + str(seed) + ".csv", mar_mask, fmt="%d", delimiter=",")
    np.savetxt("Data/mask/guangzhou/guangzhoumnar_" + str(missing_p) + "_" + str(seed) + ".csv", mnar_mask, fmt="%d", delimiter=",")

    
