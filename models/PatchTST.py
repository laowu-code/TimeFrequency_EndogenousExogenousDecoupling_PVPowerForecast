import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


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


class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, seq_len=48, pred_len=6, d_model=256, enc_in=1, n_heads=6, e_layers=3, patch_len=4,
                 stride_flag='full', prob=False):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super(PatchTST, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        stride = patch_len if stride_flag == 'full' else patch_len // 2
        padding = stride
        dropout = 0.1
        self.d_ff = 4 * d_model
        self.prob = prob

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    self.d_ff,
                    dropout=dropout,
                    activation='gelu'
                ) for l in range(e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        )

        # Prediction Head
        self.head_nf = d_model * \
                       int((seq_len - patch_len) / stride + 2)
        if not self.prob:
            self.head = FlattenHead(enc_in, self.head_nf, pred_len, head_dropout=dropout)
        else:
            self.head = FlattenHead(enc_in, self.head_nf, pred_len * 7, head_dropout=dropout)
        # elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
        #     self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
        #                             head_dropout=configs.dropout)
        # elif self.task_name == 'classification':
        #     self.flatten = nn.Flatten(start_dim=-2)
        #     self.dropout = nn.Dropout(configs.dropout)
        #     self.projection = nn.Linear(
        #         self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    # def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
    #     # Normalization from Non-stationary Transformer
    #     means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
    #     means = means.unsqueeze(1).detach()
    #     x_enc = x_enc - means
    #     x_enc = x_enc.masked_fill(mask == 0, 0)
    #     stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
    #                        torch.sum(mask == 1, dim=1) + 1e-5)
    #     stdev = stdev.unsqueeze(1).detach()
    #     x_enc /= stdev
    #
    #     # do patching and embedding
    #     x_enc = x_enc.permute(0, 2, 1)
    #     # u: [bs * nvars x patch_num x d_model]
    #     enc_out, n_vars = self.patch_embedding(x_enc)
    #
    #     # Encoder
    #     # z: [bs * nvars x patch_num x d_model]
    #     enc_out, attns = self.encoder(enc_out)
    #     # z: [bs x nvars x patch_num x d_model]
    #     enc_out = torch.reshape(
    #         enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
    #     # z: [bs x nvars x d_model x patch_num]
    #     enc_out = enc_out.permute(0, 1, 3, 2)
    #
    #     # Decoder
    #     dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
    #     dec_out = dec_out.permute(0, 2, 1)
    #
    #     # De-Normalization from Non-stationary Transformer
    #     dec_out = dec_out * \
    #               (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
    #     dec_out = dec_out + \
    #               (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
    #     return dec_out
    #
    # def anomaly_detection(self, x_enc):
    #     # Normalization from Non-stationary Transformer
    #     means = x_enc.mean(1, keepdim=True).detach()
    #     x_enc = x_enc - means
    #     stdev = torch.sqrt(
    #         torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    #     x_enc /= stdev
    #
    #     # do patching and embedding
    #     x_enc = x_enc.permute(0, 2, 1)
    #     # u: [bs * nvars x patch_num x d_model]
    #     enc_out, n_vars = self.patch_embedding(x_enc)
    #
    #     # Encoder
    #     # z: [bs * nvars x patch_num x d_model]
    #     enc_out, attns = self.encoder(enc_out)
    #     # z: [bs x nvars x patch_num x d_model]
    #     enc_out = torch.reshape(
    #         enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
    #     # z: [bs x nvars x d_model x patch_num]
    #     enc_out = enc_out.permute(0, 1, 3, 2)
    #
    #     # Decoder
    #     dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
    #     dec_out = dec_out.permute(0, 2, 1)
    #
    #     # De-Normalization from Non-stationary Transformer
    #     dec_out = dec_out * \
    #               (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
    #     dec_out = dec_out + \
    #               (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
    #     return dec_out
    #
    # def classification(self, x_enc, x_mark_enc):
    #     # Normalization from Non-stationary Transformer
    #     means = x_enc.mean(1, keepdim=True).detach()
    #     x_enc = x_enc - means
    #     stdev = torch.sqrt(
    #         torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    #     x_enc /= stdev
    #
    #     # do patching and embedding
    #     x_enc = x_enc.permute(0, 2, 1)
    #     # u: [bs * nvars x patch_num x d_model]
    #     enc_out, n_vars = self.patch_embedding(x_enc)
    #
    #     # Encoder
    #     # z: [bs * nvars x patch_num x d_model]
    #     enc_out, attns = self.encoder(enc_out)
    #     # z: [bs x nvars x patch_num x d_model]
    #     enc_out = torch.reshape(
    #         enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
    #     # z: [bs x nvars x d_model x patch_num]
    #     enc_out = enc_out.permute(0, 1, 3, 2)
    #
    #     # Decoder
    #     output = self.flatten(enc_out)
    #     output = self.dropout(output)
    #     output = output.reshape(output.shape[0], -1)
    #     output = self.projection(output)  # (batch_size, num_classes)
    #     return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        out = dec_out[:, :, 0]
        if self.prob:
            out = out.reshape(out.shape[0], -1, 7)
        return out  # [B, L, D]
