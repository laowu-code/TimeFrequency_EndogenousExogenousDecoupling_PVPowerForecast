from models import iTransformer_block, CrossAttention
import torch.nn as nn
from models.KAN import KAN
from utils.loss_kan import Dense_Gauss, SDERLayer
from models.TimeSter import TimeSter
import torch
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class dlinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """
    def __init__(self, seq_len,pred_len,moving_avg=3,enc_in=1, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(dlinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decompsition = series_decomp(moving_avg)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # if self.task_name == 'classification':
        #     self.projection = nn.Linear(
        #         configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class GTUFusion(nn.Module):
    def __init__(self, dim_t, dim_f, dim_h=None):
        super(GTUFusion, self).__init__()
        dim_h=dim_t if dim_h is None else dim_h
        self.linear_main = nn.Linear(dim_t, dim_h)
        self.linear_gate = nn.Linear(dim_f, dim_h)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, main_feature, gate_feature):
        gate_feature=  F.max_pool1d(gate_feature.transpose(2,1), kernel_size=4).transpose(2,1)
        gated_output = self.tanh(self.linear_main(main_feature)) * self.sigmoid(self.linear_gate(gate_feature))
        return gated_output
class Proposed_prob(nn.Module):
    def __init__(self, enc_in=5, pred_len=1, d_lstm=128, layers_lstm=1,
                 seq_len=48, d_model=128, e_layers=4, n_heads=6, prob=True, prob_type='QR',beta=0.4):
        super(Proposed_prob, self).__init__()
        self.model1 = iTransformer_block(num_variates=enc_in-1, lookback_len=seq_len,
                                         pred_length=pred_len, dim=d_model, depth=e_layers, heads=n_heads,
                                         num_tokens_per_variate=1, use_reversible_instance_norm=True)
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=d_lstm,
                            num_layers=layers_lstm,
                            batch_first=True,
                            bidirectional=False)

        # self.dlinear=dlinear(seq_len=seq_len,pred_len=d_model,enc_in=enc_in-1)

        # self.cross = CrossAttention(dim=d_lstm, lenth=d_model)
        self.cross=GTUFusion(dim_t=d_lstm,dim_f=d_model,dim_h=d_lstm)
        self.prob = prob
        self.prob_type = prob_type
        self.pred_len = pred_len
        self.beta=beta
        self.enc_in = enc_in
        self.time_enc= TimeSter(n_variate=1, n_out=1,time_dim=5, seq_len=seq_len, pred_len=pred_len,ksize=3)
        if not self.prob:
            self.kan = KAN([d_lstm, pred_len])
        else:
            if prob_type == 'SDER':
                self.kan = SDERLayer(d_lstm, pred_len)
                #
            elif prob_type == 'QL':
                self.kan = KAN([d_lstm, pred_len * 7])
            elif prob_type == 'Gauss' or prob_type == 'Laplace':
                self.kan = Dense_Gauss(d_lstm, pred_len)
            else:
                raise ValueError('Invalid prob_type')

    def forward(self, x):
        x2, _ = self.lstm(x[:, :, 0,None])
        # x2=self.dlinear(x[:, :, 1:self.enc_in]).transpose(2,1)
        x2=x2[:,-1,None,:]
        x1 = self.model1(x[:, :,1:self.enc_in])
        if x.shape[-1]>self.enc_in:
            x_mark = x[:, :, self.enc_in:]
        else:
            x_mark = None
        x1 = self.cross(x2, x1)
        if not self.prob:
            x_o = self.kan(x1)
        else:
            if self.prob_type == 'QL':
                x_o = self.kan(x1).view(-1, self.pred_len, 7)
            else:
                x_o = self.kan(x1)
        if x_mark is not None:
            x_time = self.time_enc(x_mark)
            if self.prob_type == 'QL':
                x_o = (1-self.beta)*x_o + self.beta*x_time
                # x_o=x_o
            else:
                x_o[:,:,0] = (1-self.beta)*x_o[:,:,0] + self.beta*x_time.squeeze(-1)
                # x_o=x_o
        return x_o
