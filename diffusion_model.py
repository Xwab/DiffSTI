import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from S4Model import S4Layer
from spatial_conv import SpatialConvOrderK
from gcrnn import GCGRUCell




def Conv1d(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def attention_layer(channels=64, heads=8, layers=1):
    encoder = nn.TransformerEncoderLayer(d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu")
    return nn.TransformerEncoder(encoder, num_layers=layers)


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
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
        steps = torch.arange(num_steps).unsqueeze(1)  
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  
        table = steps * frequencies  
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  
        return table


class Diff_STI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.channels = config["channels"]

        self.input_projection = Conv1d(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d(self.channels, 1, 1)
        self.spa_conv = SpatialConvOrderK(1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step, observed_data, cond_mask, adj, A_m, k=2):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        support = self.spa_conv.compute_support_orderK(adj, k)
        observed_data = torch.unsqueeze(observed_data, 1)
        spa_x = 0
    
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb, cond_mask, spa_x, support, A_m, adj)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d(side_dim, 2 * channels, 1)
        self.spa_cond_projection = Conv1d(1, 2 * channels, 1) 
        self.spa_conv = SpatialConvOrderK(channels, channels)
        self.afternn_conv = Conv1d(channels, channels, 1)
        self.mid_projection = Conv1d(channels, 2 * channels, 1)
        self.output_projection = Conv1d(channels, 2 * channels, 1)
        self.S41 = S4Layer(features=channels,
                          lmax=200,
                          N=64,
                          dropout=0.0,
                          bidirectional=1,
                          layer_norm=1)

        self.S42 = S4Layer(features=2 * channels,
                          lmax=200,
                          N=64,
                          dropout=0.0,
                          bidirectional=1,
                          layer_norm=1)
        self.gcrnn = nn.ModuleList()
        for _ in range(1):
            self.gcrnn.append(GCGRUCell(d_in=channels + 1, num_units=channels, support_len=4, order=2))
        self.time_layer = attention_layer(heads=nheads, layers=1, channels=channels)
        self.feature_layer = attention_layer(heads=nheads, layers=1, channels=channels)

    def temporal_attn_forward(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def feature_attn_forward(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def update_state(self, x, h, adj, m_in):
        rnn_in = x
        for layer, cell in enumerate(self.gcrnn):
            rnn_in = h[layer] = cell(rnn_in, h[layer], adj, m_in)
        return rnn_in, h

    def init_hidden_states(self, x):
        return [torch.zeros(size=(x.shape[0], x.shape[1], x.shape[2])).to(x.device) for _ in range(1)]

    def forward(self, x, cond_info, diffusion_emb, cond_mask ,spa_x, support, A_m, adj):
        #observed_mask: B K L
        #cond_mask: B K L

        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)       

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb   #(B, channel, K * L)

        y = self.temporal_attn_forward(y, base_shape)
        y = self.feature_attn_forward(y, base_shape)
    
        #S4-1
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.S41(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)

        #spatial
        y = y.reshape(B, channel, K, L)
        
        rnn_hidden = self.init_hidden_states(y)
        rnn_out = []
        #GCRNN
        for step in range(L):
            m_s = cond_mask[...,step].unsqueeze(1)
            y_s = y[..., step]
            m_in = cond_mask[...,step].unsqueeze(1).repeat(1,K,1).unsqueeze(-1)
            y_in = torch.cat([y_s, m_s], dim=1)
            out, rnn_hidden = self.update_state(y_in, rnn_hidden, support, m_in)
            rnn_out.append(out)
            
        y = torch.stack(rnn_out, dim=-1)

        y = self.afternn_conv(y.reshape(B, channel, K*L))

        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)

        y = y + cond_info

        #S4-2
        y = y.reshape(B, 2 * channel, K, L).permute(0, 2, 1, 3).reshape(B * K, 2 * channel, L)
        y = self.S42(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, 2 * channel, L).permute(0, 2, 1, 3).reshape(B, 2 * channel, K * L)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip