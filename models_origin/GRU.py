import torch.nn as nn
class Model(nn.Module):
    """Long Short Term Memory"""

    def __init__(self, args, bidirectional=False):
        super(Model, self).__init__()

        self.input_size = args.enc_in
        self.hidden_size = args.d_model
        self.num_layers = args.e_layers
        self.output_size = args.c_out
        self.bidirectional = bidirectional

        self.lstm = nn.GRU(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        if self.bidirectional:
            self.fc = nn.Linear(self.hidden_size * 2, self.output_size)
        else:
            self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out