import torch
from torch import nn
from Util.ModelHelper import build_layers


class GRU(nn.Module):
    def __init__(self, out_dim, in_dim=200, dims=None):
        super().__init__()

        if dims is None:
            dims = [200, 200]
        self.in_dim = in_dim
        self.layers = build_layers(in_dim, out_dim, dims, 'gru')

    def forward(self, x):
        sequence_len = len(x)
        x = torch.reshape(x, (sequence_len, 1, -1))
        for layer in self.layers:
            if layer == 'gru':
                x, _ = self.layers[layer](x)
            else:
                x = self.layers[layer](x)
        return x[-1]


