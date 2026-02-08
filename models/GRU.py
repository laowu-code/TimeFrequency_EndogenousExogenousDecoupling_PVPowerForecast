# import torch.nn as nn
# class GRU(nn.Module):
#     """Gat e Recurrent Unit"""
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(GRU, self).__init__()
#
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.output_size = output_size
#
#         self.gru = nn.GRU(input_size=input_size,
#                           hidden_size=hidden_size,
#                           num_layers=num_layers,
#                           batch_first=True)
#
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         out, _ = self.gru(x)
#         out = out[:, -1, :]
#         out = self.fc(out)
#
#         return out
import torch.nn as nn
class GRU(nn.Module):
    """Long Short Term Memory"""

    def __init__(self, enc_in, d_model, e_layers, pred_len,seq_len=None,prob=False):
        super(GRU, self).__init__()

        self.input_size = enc_in
        self.hidden_size = d_model
        self.num_layers = e_layers
        self.output_size = pred_len
        self.prob = prob

        self.gru = nn.GRU(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,)
        if not self.prob:
            self.fc = nn.Linear(self.hidden_size, self.output_size)
        else:
            self.fc = nn.Linear(self.hidden_size, self.output_size * 7)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        if self.prob:
            out = out.view(-1, self.output_size, 7)
        return out