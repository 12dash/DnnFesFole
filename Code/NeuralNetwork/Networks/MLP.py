from torch import nn
from Util.ModelHelper import build_layers


class MLP(nn.Module):
    def __init__(self, out_dim, in_dim=200, dims=None):
        super().__init__()

        if dims is None:
            dims = [100, 100, 100]
        self.layers = build_layers(in_dim, out_dim, dims)

    def forward(self, x):
        for layer in self.layers:
            x = self.layers[layer](x)
        return x
