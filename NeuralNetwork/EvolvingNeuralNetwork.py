import torch
from torch import nn
import torch.nn.functional as F

from NeuralNetwork.Evolving.DynamicLayer import DynamicLayer
from NeuralNetwork.Networks.MLP import MLP
from NeuralNetwork.Networks.LSTM import LSTM
from NeuralNetwork.Networks.GRU import GRU


class EvolvingNN(nn.Module):
    def __init__(self, out_dim, in_dim=200, dims=None, sub_model='MLP'):
        super().__init__()

        self.out_dim = out_dim
        self.in_dim = in_dim

        self.child_model_type = 'seq' if sub_model in ['LSTM', 'GRU'] else 'non_seq'

        self.fc1 = DynamicLayer(self.in_dim)
        self.fc2 = self.initialize_child_model(sub_model, dims)

    def initialize_child_model(self, sub_model, dims):
        fc2 = None
        if sub_model == 'MLP':
            fc2 = MLP(out_dim=self.out_dim, in_dim=self.in_dim, dims=dims)
        elif sub_model == 'LSTM':
            fc2 = LSTM(out_dim=self.out_dim, in_dim=self.in_dim, dims=dims)
        elif sub_model == 'GRU':
            fc2 = GRU(out_dim=self.out_dim, in_dim=self.in_dim, dims=dims)
        return fc2

    def forward(self, x):
        # x = [(val, id), (val, id)...]
        def build_sequential_input(x):
            seq = []
            for i in x[:-1]:
                seq.append(self.fc1.forward(i).view(1, -1))
                self.fc1.clear_grad()
            seq.append(self.fc1.forward(x[-1]).view(1, -1))
            x = torch.cat(seq, dim=0)
            return x

        if self.child_model_type == 'seq':
            x = build_sequential_input(x)
        else:
            x = F.relu(self.fc1.forward(x))
        x = self.fc2(x)
        return x

    def backward_compute(self):
        self.fc1.backward_propagation()

    def merge_nodes(self, node_list, merged_node):
        self.fc1.merge_nodes(node_list, merged_node)
