import numpy as np
import torch
import torch.nn as nn
from models.diffusion import diff_FGTI

class FGTI(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.target_dim = configs.enc_in

        self.emb_time_dim = configs.timeemb
        self.emb_feature_dim = configs.featureemb

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.configs.side_dim = self.emb_total_dim
        self.embed_layer = nn.Embedding(
            num_embeddings=self.configs.enc_in, embedding_dim=self.emb_feature_dim
        )
        self.diffmodel = diff_FGTI(self.configs)

        # parameters for diffusion models
        self.num_steps = configs.diffusion_step_num
        if configs.schedule == "quad":
            self.beta = np.linspace(
                configs.beta_start ** 0.5, configs.beta_end ** 0.5, self.num_steps
            ) ** 2
        elif configs.schedule == "linear":
            self.beta = np.linspace(
                configs.beta_start, configs.beta_end, self.num_steps
            )
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def process_data(self, observed_data, observed_dataf, observed_mask,observed_tp,gt_mask):
        observed_data = observed_data.to(self.device).float().permute(0,2,1)
        observed_dataf = observed_dataf.to(self.device).float().permute(0,2,1)
        observed_mask = observed_mask.to(self.device).float().permute(0,2,1)
        observed_tp = observed_tp.to(self.device).float()
        gt_mask = gt_mask.to(self.device).float().permute(0,2,1)

        return (observed_data, observed_dataf, observed_mask, observed_tp, gt_mask)
    
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask


    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        return side_info

    def set_input_to_diffmodel(self, noisy_data, observed_data, observed_dataf, cond_mask):
 
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)

        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        return total_input

    def calc_loss(self, observed_data, observed_dataf, cond_mask, observed_mask, side_info):
        B, L, K = observed_data.shape
        #Generate Time Step
        diffusion_step = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[diffusion_step]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        # forward diffusion
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        
        cond_obs = cond_mask * observed_data
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, observed_dataf, cond_mask)

        predicted = self.diffmodel(total_input, side_info, observed_dataf, diffusion_step)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def impute(self, observed_data, observed_dataf, cond_mask, side_info, n_samples=100):
        B, K, L = observed_data.shape
        observed_data = observed_data * cond_mask
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                 
                total_input = self.set_input_to_diffmodel(current_sample, observed_data, observed_dataf, cond_mask)

                predicted = self.diffmodel(total_input, side_info, observed_dataf, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, observed_data, observed_dataf, observed_mask, observed_tp, gt_mask):
        (observed_data, observed_dataf, observed_mask, observed_tp, gt_mask) = self.process_data(observed_data, observed_dataf, observed_mask, observed_tp, gt_mask)
        observed_data = observed_data * observed_mask
   
        cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        return self.calc_loss(observed_data, observed_dataf, cond_mask, observed_mask, side_info)
    
    def evaluate(self, observed_data, observed_dataf, observed_mask, observed_tp, gt_mask, n_samples=100):
        (observed_data, observed_dataf, observed_mask, observed_tp, gt_mask) = self.process_data(observed_data, observed_dataf, observed_mask, observed_tp, gt_mask)
        with torch.no_grad():
            cond_mask = observed_mask
            side_info = self.get_side_info(observed_tp, cond_mask)

            imputed_samples = self.impute(observed_data, observed_dataf, cond_mask, side_info, n_samples)

            return imputed_samples, observed_data, gt_mask-observed_mask, observed_mask, observed_tp
        