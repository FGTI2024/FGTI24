import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.fft
from layers.Embed import DataEmbedding
from layers.Diff_layers import *
from models.ts_transformer import TSTransformerEncoder

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class diff_FGTI(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.channel = configs.channel
        self.residual_layers = self.configs.residual_layers
        
        self.encoder = TSTransformerEncoder(configs)

        #condition:
        self.condition_projection = Conv1d_with_init(self.configs.d_model, self.configs.enc_in * self.channel, 1)
        self.side_projection = Conv1d_with_init(self.configs.side_dim, self.channel, 1)
        self.condition_projection2 = Conv1d_with_init(self.channel, self.channel, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=configs.diffusion_step_num,
            embedding_dim=configs.d_model
        )

        self.input_projection = Conv1d_with_init(2, self.channel, 1)
        self.output_projection1 = Conv1d_with_init(self.channel, self.channel, 1)
        self.output_projection2 = Conv1d_with_init(self.channel, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=self.configs.side_dim,
                    channels=self.channel,
                    diffusion_embedding_dim=self.configs.d_model,
                    nheads=self.configs.nheads,
                    target_dim = self.configs.enc_in,
                    proj_t = self.configs.proj_t
                )
                for _ in range(self.residual_layers)
            ]
        )

    def forward(self, total_input, side_info, cond_obs, diffusion_step):
        B, inputdim, K, L = total_input.shape

        x = total_input.reshape(B, inputdim, K * L)

        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channel, K, L)

        #condition guide
        conditions = self.encoder.encoder(cond_obs.permute(0,2,1)).permute(0,2,1) #[B, d, L]
        conditions = self.condition_projection(conditions) # [B, d*K, L]
        sides = side_info.reshape(B, -1, K * L)
        sides = self.side_projection(sides)
        
        conditions = conditions.reshape(B, -1, K*L)
        conditions = conditions + sides
        conditions = self.condition_projection2(conditions)
        conditions = F.relu(conditions)


        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, side_info, diffusion_emb, conditions)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channel, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x
    
class NoiseProject(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, target_dim, proj_t, is_cross_t=True, is_cross_s=True):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.forward_time = TemporalLearning(channels=channels, nheads=nheads, is_cross=is_cross_t)
        self.forward_feature = FeatureLearning(channels=channels, nheads=nheads, 
                                               target_dim = target_dim, proj_t=proj_t, is_cross=is_cross_s)

    def forward(self, x, side_info, diffusion_emb, guide_info):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape, guide_info)
        y = self.forward_feature(y, base_shape, guide_info)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip