import torch
import torch.nn as nn
from layers.Pyraformer_EncDec import Encoder


class Pyraformer(nn.Module):
    """ 
    Pyraformer: Pyramidal attention to reduce complexity
    Paper link: https://openreview.net/pdf?id=0EXmFzUn5I
    """

    def __init__(self,pred_len=4,seq_len=48,n_heads=4,e_layers=1,d_model=128,enc_in=1, window_size=[2,2], inner_size=5,prob=False):
        """
        window_size: list, the downsample window size in pyramidal attention.
        inner_size: int, the size of neighbour attention
        """
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_ff=4*d_model
        # short term forecast
        # window_size = [2,2]
        self.encoder = Encoder(d_model,seq_len,self.d_ff,n_heads,e_layers,enc_in, window_size, inner_size)
        self.prob = prob
        if not self.prob:
            self.projection = nn.Linear(
                (len(window_size)+1)*self.d_model, self.pred_len * enc_in)
        else:
            self.projection = nn.Linear(
                (len(window_size) + 1) * self.d_model, self.pred_len * 7)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)
        return dec_out
    
    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        enc_out = self.encoder(x_enc, x_mark_enc)
        enc_out=enc_out[:, -1, :]
        if not self.prob:
            dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)
            dec_out = dec_out * std_enc + mean_enc
        else:
            dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)
            dec_out = dec_out * std_enc[:,:,0,None] + mean_enc[:,:,0,None]
        return dec_out
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):

        dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if not self.prob:
            dec_out=dec_out[:, -self.pred_len:, 0]
        return dec_out # [B, L, D]

