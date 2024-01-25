
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
import torchcde

class KDD_DATASET(Dataset):
    def __init__(self, configs):
        super(KDD_DATASET, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/KDD_norm.csv", delimiter=",")[0:8016].reshape(-1, self.configs.seq_len, self.configs.enc_in)
        if self.configs.missing_rate == 0:
            self.mask = np.ones_like(self.data)
            self.mask[np.where(self.data == -200)] = 0
        else:
            self.mask = np.loadtxt("Data/mask/kdd/kdd_"+str(self.configs.missing_rate)+ "_"  + str(self.configs.seed) +  ".csv", delimiter=",")[0:8016].reshape(-1, self.configs.seq_len, self.configs.enc_in)
        self.mask_gt = np.ones_like(self.data)
        self.mask_gt[np.where(self.data == -200)] = 0

        self.dataf = np.array(self.data)
        self.dataf[np.where(self.mask == 0)] = np.nan
        self.dataf = torch.from_numpy(self.dataf).float()
        self.dataf = torchcde.linear_interpolation_coeffs(self.dataf)
        self.maxdataf = self.dataf.clone().detach()

        temp_list = []
        for j in range(self.dataf.shape[2]):
            x = self.dataf[:, :, j]
            xf = torch.fft.rfft(x)
            pass_f = torch.abs((torch.fft.rfftfreq(x.shape[1]))) > self.configs.flimit
            rx = torch.fft.irfft(xf * pass_f, n=x.shape[1])

            temp_list.append(rx)
        self.dataf = torch.stack(temp_list, dim=2)

        temp_list = []
        for j in range(self.maxdataf.shape[2]):
            x_j = self.maxdataf[:, :, j]
            xf = torch.fft.rfft(x_j)
            freq = abs(xf)
            __, toplist = torch.topk(freq, self.configs.topf)

            for x in range(xf.shape[0]):
                for y in range(xf.shape[1]):
                    if y not in toplist[x]:
                        xf[x,y] = 0
            
            rx = torch.fft.irfft(xf, n=x_j.shape[1])
            temp_list.append(rx)
        self.maxdataf = torch.stack(temp_list, dim=2)

        self.dataf = torch.stack([self.dataf, self.maxdataf], dim=-1).reshape(-1, self.configs.seq_len, self.configs.enc_in*2)

    def __len__(self):
        # Needs to be divisible
        return self.data.shape[0]

    def __getitem__(self, index):
        data_res = self.data[index]
        dataf_res = self.dataf[index]
        mask_res = self.mask[index]
        observed_tp = np.arange(self.configs.seq_len)
        mask_gt = self.mask_gt[index]

        data_res = torch.from_numpy(data_res).float()
        mask_res = torch.from_numpy(mask_res).float()    
        observed_tp = torch.from_numpy(observed_tp).float()
        mask_gt = torch.from_numpy(mask_gt).float()

        return data_res, dataf_res, mask_res, observed_tp, mask_gt

class GUANGZHOU_DATASET(Dataset):
    def __init__(self, configs):
        super(GUANGZHOU_DATASET, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/Guangzhou_norm.csv", delimiter=",")[0: 8784 // 48 * 48].reshape(-1, self.configs.seq_len, self.configs.enc_in)
        self.mask = np.loadtxt("Data/mask/guangzhou/guangzhou_"+str(self.configs.missing_rate)+ "_"  + str(self.configs.seed) +  ".csv", delimiter=",")[0: 8784 // 48 * 48].reshape(-1, self.configs.seq_len, self.configs.enc_in)
        self.mask_gt = np.ones_like(self.data)
        self.mask_gt[np.where(self.data == -200)] = 0

        self.dataf = np.array(self.data)
        self.dataf[np.where(self.mask == 0)] = np.nan
        self.dataf = torch.from_numpy(self.dataf).float()
        self.dataf = torchcde.linear_interpolation_coeffs(self.dataf)
        self.maxdataf = self.dataf.clone().detach()

        temp_list = []
        for j in range(self.dataf.shape[2]):
            x = self.dataf[:, :, j]
            xf = torch.fft.rfft(x)
            pass_f = torch.abs((torch.fft.rfftfreq(x.shape[1]))) > self.configs.flimit
            rx = torch.fft.irfft(xf * pass_f, n=x.shape[1])

            temp_list.append(rx)
        self.dataf = torch.stack(temp_list, dim=2)

        temp_list = []
        for j in range(self.maxdataf.shape[2]):
            x_j = self.maxdataf[:, :, j]
            xf = torch.fft.rfft(x_j)
            freq = abs(xf)
            __, toplist = torch.topk(freq, self.configs.topf)

            for x in range(xf.shape[0]):
                for y in range(xf.shape[1]):
                    if y not in toplist[x]:
                        xf[x,y] = 0
            
            rx = torch.fft.irfft(xf, n=x_j.shape[1])
            temp_list.append(rx)
        self.maxdataf = torch.stack(temp_list, dim=2)

        self.dataf = torch.stack([self.dataf, self.maxdataf], dim=-1).reshape(-1, self.configs.seq_len, self.configs.enc_in*2)

    def __len__(self):
        # Needs to be divisible
        return self.data.shape[0]

    def __getitem__(self, index):
        data_res = self.data[index]
        dataf_res = self.dataf[index]
        mask_res = self.mask[index]
        observed_tp = np.arange(self.configs.seq_len)
        mask_gt = self.mask_gt[index]

        data_res = torch.from_numpy(data_res).float()
        mask_res = torch.from_numpy(mask_res).float()    
        observed_tp = torch.from_numpy(observed_tp).float()
        mask_gt = torch.from_numpy(mask_gt).float()

        return data_res, dataf_res, mask_res, observed_tp, mask_gt

class PHYSIO_DATASET(Dataset):
    def __init__(self, configs):
        super(PHYSIO_DATASET, self).__init__()
        self.configs = configs
        self.data = np.loadtxt("Data/Physio_norm.csv", delimiter=",").reshape(-1, self.configs.seq_len, self.configs.enc_in)

        self.mask_gt = np.ones_like(self.data)
        self.mask_gt[np.where(self.data == -200)] = 0
        self.mask = self.mask_gt

        self.dataf = np.array(self.data)
        self.dataf[np.where(self.mask == 0)] = np.nan
        self.dataf = torch.from_numpy(self.dataf).float()
        self.dataf = torchcde.linear_interpolation_coeffs(self.dataf)
        self.maxdataf = self.dataf.clone().detach()

        temp_list = []
        for j in range(self.dataf.shape[2]):
            x = self.dataf[:, :, j]
            xf = torch.fft.rfft(x)
            pass_f = torch.abs((torch.fft.rfftfreq(x.shape[1]))) > self.configs.flimit
            rx = torch.fft.irfft(xf * pass_f, n=x.shape[1])

            temp_list.append(rx)
        self.dataf = torch.stack(temp_list, dim=2)

        temp_list = []
        for j in range(self.maxdataf.shape[2]):
            x_j = self.maxdataf[:, :, j]
            xf = torch.fft.rfft(x_j)
            freq = abs(xf)
            __, toplist = torch.topk(freq, self.configs.topf)

            for x in range(xf.shape[0]):
                for y in range(xf.shape[1]):
                    if y not in toplist[x]:
                        xf[x,y] = 0
            
            rx = torch.fft.irfft(xf, n=x_j.shape[1])
            temp_list.append(rx)
        self.maxdataf = torch.stack(temp_list, dim=2)

        self.dataf = torch.stack([self.dataf, self.maxdataf], dim=-1).reshape(-1, self.configs.seq_len, self.configs.enc_in*2)

    def __len__(self):
        # Needs to be divisible
        return self.data.shape[0]

    def __getitem__(self, index):
        data_res = self.data[index]
        dataf_res = self.dataf[index]
        mask_res = self.mask[index]
        observed_tp = np.arange(self.configs.seq_len)
        mask_gt = self.mask_gt[index]

        data_res = torch.from_numpy(data_res).float()
        mask_res = torch.from_numpy(mask_res).float()    
        observed_tp = torch.from_numpy(observed_tp).float()
        mask_gt = torch.from_numpy(mask_gt).float()

        return data_res, dataf_res, mask_res, observed_tp, mask_gt

def get_physio_dataset(configs):
    dataset = PHYSIO_DATASET(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader

def get_kdd_dataset(configs):
    dataset = KDD_DATASET(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader

def get_guangzhou_dataset(configs):
    dataset = GUANGZHOU_DATASET(configs)
    train_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    return train_loader, test_loader


def get_dataset(configs):
    if configs.dataset == "kdd":
        return get_kdd_dataset(configs)
    if configs.dataset == "physio":
        return get_physio_dataset(configs)
    if configs.dataset == "guangzhou":
        return get_guangzhou_dataset(configs)