import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np
from models.KAN import KAN
from utils.loss_kan import Dense_Gauss, SDERLayer


class Frets(nn.Module):
    """
    """

    def __init__(self, embed_size=128, hidden_size=256, pred_len=4, enc_in=5, seq_len=48):
        super(Frets, self).__init__()
        self.embed_size = embed_size  # embed_size
        self.hidden_size = hidden_size  # hidden_size
        self.pred_len = pred_len
        self.feature_size = enc_in  # channels
        self.seq_len = seq_len
        self.channel_independence = 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            # nn.Linear(self.hidden_size, self.pred_len)
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forecast(self, x_enc):
        # x: [Batch, Input length, Channel]
        B, T, N = x_enc.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_enc)
        bias = x
        # [B, N, T, D]
        if self.channel_independence == '0':
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        dec_out = self.forecast(x_enc)
        return dec_out  # [B, L, D]
        # else:
        #     raise ValueError('Only forecast tasks implemented yet')


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Timexer(nn.Module):

    def __init__(self, seq_len, pred_len, patch_len, e_layers, enc_in, d_model, n_heads, drop=0.1, use_norm=True):
        super(Timexer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm
        self.patch_len = patch_len
        self.patch_num = int(seq_len // patch_len)
        self.n_heads = n_heads
        self.d_ff = 2 * d_model
        self.n_vars = 1
        self.enc_in = enc_in
        self.drop = drop
        self.e_layers = e_layers
        self.d_model = d_model
        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, d_model, self.patch_len, self.drop)

        self.ex_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, )

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=self.drop,
                                      output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=self.drop,
                                      output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.drop,
                    activation='gelu',
                )
                for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.head_nf = self.d_model * (self.patch_num + 1)
        self.head = FlattenHead(self.enc_in, self.head_nf, self.pred_len,
                                head_dropout=self.drop)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # if self.use_norm:
        #     # Normalization from Non-stationary Transformer
        #     means = x_enc.mean(1, keepdim=True).detach()
        #     x_enc = x_enc - means
        #     stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #     x_enc /= stdev

        _, _, N = x_enc.shape

        en_embed, n_vars = self.en_embedding(x_enc[:, :, 0].unsqueeze(-1).permute(0, 2, 1))

        # if x_enc.shape[-1]==1:
        # x_enc[:, :, 1:] = torch.zeros_like(x_enc[:, :, 1:])
        ex_embed = self.ex_embedding(x_enc[:, :, 2:], x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        enc_out = enc_out.flatten(-2)
        return enc_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out  # [B, L, D]
        # else:
        #     return None


class GTUFusion(nn.Module):
    def __init__(self, dim_t, dim_f, dim_h):
        super(GTUFusion, self).__init__()
        self.linear_main = nn.Linear(dim_t, dim_h)
        self.linear_gate = nn.Linear(dim_f, dim_h)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, main_feature, gate_feature):
        gated_output = self.tanh(self.linear_main(main_feature)) * self.sigmoid(self.linear_gate(gate_feature))
        return gated_output


class CrossAttention(nn.Module):
    def __init__(self, dim_x, lenth, dim):
        super(CrossAttention, self).__init__()
        self.query_layer = nn.Linear(dim_x, dim)
        self.key_layer = nn.Linear(lenth, dim)
        self.value_layer = nn.Linear(lenth, dim)
        self.scale = dim ** 0.5

    def forward(self, input1, input2):
        # 生成查询、键、值
        query = self.query_layer(input1)  # (b, c1, d)
        key = self.key_layer(input2)  # (b, c, l)
        value = self.value_layer(input2)  # (b, c, l)
        # 计算注意力权重
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # (b, c1, c)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # 归一化
        # 使用注意力权重加权值
        attended_output = torch.matmul(attention_weights, value)  # (b, c1, d)
        return attended_output


class Proposed(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, e_layers, enc_in, d_model, n_heads, embed_size, hidden_size,
                 dim_f=None, prob=False, prob_type='DER', gate=False):
        super(Proposed, self).__init__()
        self.frets = Frets(embed_size=embed_size, hidden_size=hidden_size, pred_len=pred_len, enc_in=1, seq_len=seq_len)
        self.timexer = Timexer(seq_len=seq_len, patch_len=patch_len, pred_len=pred_len, d_model=d_model,
                               e_layers=e_layers, enc_in=enc_in, n_heads=n_heads)
        self.patch_num = int(seq_len // patch_len) + 1
        if gate:
            self.gate = GTUFusion(d_model * self.patch_num, hidden_size, d_model)
        else:
            self.gate = CrossAttention(d_model * self.patch_num, hidden_size, d_model)
        self.prob = prob
        self.prob_type = prob_type
        self.pred_len = pred_len
        if not self.prob:
            self.kan = KAN([d_model, pred_len])
        else:
            if prob_type == 'SDER':
                self.kan = SDERLayer(d_model, pred_len)
            elif prob_type == 'QL':
                self.kan = KAN([d_model, pred_len * 7])
            elif prob_type == 'Gauss_Laplace':
                self.kan = Dense_Gauss(d_model, pred_len)
            else:
                raise ValueError('Invalid prob_type')

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x_t = self.timexer(x_enc)
        x_f = self.frets(x_enc[:, :, 0, None]).transpose(2, 1)
        x_g = self.gate(x_t, x_f).flatten(1)
        if not self.prob:
            x_o = self.kan(x_g)
        else:
            if self.prob_type == 'QL':
                x_o = self.kan(x_g).view(-1, self.pred_len, 7)
            else:
                x_o = self.kan(x_g)
        return x_o
