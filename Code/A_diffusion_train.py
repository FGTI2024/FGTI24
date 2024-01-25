import torch
import numpy as np
import A_train
import argparse

parser = argparse.ArgumentParser(description='FGTI')

parser.add_argument('--device', type=str, default="cuda", help='input sequence length')
parser.add_argument('--batch', type=int, default=16, help='input batch size')
parser.add_argument('--dataset', type=str, default="kdd", help='data set name')
parser.add_argument('--missing_rate', type=float, default=0.1, help='missing percent for experiment')
parser.add_argument('--seed', type=int, default=3407, help='random seed')

# input data enc_in c_out setting: kdd:99 guangzhou:214 physio:37
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--enc_in', type=int, default=99, help='encoder input size')
parser.add_argument('--c_out', type=int, default=99, help='decoder output size')

# encoder model setting 
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')

#Diffusion learning setting
parser.add_argument('--diffusion_step_num', type=int, default=50, help='total number of diffusion step')
parser.add_argument('--timeemb', type=int, default=128, help='side information timeemb dimension')
parser.add_argument('--featureemb', type=int, default=16, help='side information featureemb dimension')
parser.add_argument('--nheads', type=int, default=8, help='number of head for attention')
parser.add_argument('--channel', type=int, default=128, help='channel dimension of diffusion')
parser.add_argument('--proj_t', type=int, default=128, help='proj_t for feature self-attention')
parser.add_argument('--residual_layers', type=int, default=4, help='number of residual layers in diffusion model')
parser.add_argument('--schedule', type=str, default='quad', help='beta increase schedule')
parser.add_argument('--beta_start', type=float, default=0.0001)
parser.add_argument('--beta_end', type=float, default=0.2)
parser.add_argument('--epoch_diff', type=int, default=200, help='training epoch for diffusion training')
parser.add_argument('--learning_rate_diff', type=float, default=1e-3, help='learning rate of diffusion training')

#FGTI
parser.add_argument('--flimit', type=float, default=0.3,  help='cutoff frequency in high-freq filer') 
parser.add_argument('--topf', type=int, default=10, help='number of dominant freq for dominant-freq filer')

if __name__ == '__main__':
    configs = parser.parse_args()

    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    torch.cuda.manual_seed(configs.seed)

    model = A_train.diffusion_train(configs)
    print("TEST")
    A_train.diffusion_test(configs, model)