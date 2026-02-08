import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from layers.Embed import DataEmbedding

class Mamba_model(nn.Module):
    
    def __init__(self, pred_len,enc_in,d_model,d_conv,d_ff,expand,c_out,e_layers=1):
        super(Mamba_model, self).__init__()
        self.pred_len = pred_len

        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16) # TODO implement "auto"
        
        self.embedding = DataEmbedding(enc_in, d_model)

        self.mamba = Mamba(
            d_model = d_model,
            d_state = d_ff,
            d_conv = d_conv,
            expand = expand,
        )

        self.out_layer = nn.Linear(d_model, c_out, bias=False)

    def forecast(self, x_enc, x_mark_enc=None):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x)
        x_out = self.out_layer(x)

        x_out = x_out * std_enc + mean_enc
        return x_out

    def forward(self, x_enc, x_mark_enc=None):
        x_out = self.forecast(x_enc, x_mark_enc)
        return x_out[:, -self.pred_len:, :]

        # other tasks not implemented