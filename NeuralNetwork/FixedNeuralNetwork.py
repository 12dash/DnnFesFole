from NeuralNetwork.Networks.MLP import MLP
from NeuralNetwork.Networks.LSTM import LSTM
from NeuralNetwork.Networks.GRU import GRU
from Util.ModelHelper import *
from Util.VisualizationHelper import plot_output
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class StaticModel(nn.Module):
    def __init__(self, inp_dim, out_dim, in_fc2_dim=200, dims=None, sub_model='MLP'):
        super().__init__()

        self.out_dim = out_dim
        self.in_fc2_dim = in_fc2_dim

        self.child_model_type = 'seq' if sub_model in ['LSTM', 'GRU'] else 'non_seq'

        self.fc1 = nn.Linear(inp_dim, in_fc2_dim)
        self.fc2 = self.initialize_child_model(sub_model, dims)

    def initialize_child_model(self, sub_model, dims):
        fc2 = None
        if sub_model == 'MLP':
            fc2 = MLP(out_dim=self.out_dim, in_dim=self.in_fc2_dim, dims=dims)
        elif sub_model == 'LSTM':
            fc2 = LSTM(out_dim=self.out_dim, in_dim=self.in_fc2_dim, dims=dims)
        elif sub_model == 'GRU':
            fc2 = GRU(out_dim=self.out_dim, in_dim=self.in_fc2_dim, dims=dims)
        return fc2

    def forward(self, x):
        # x = [(val, id), (val, id)...]
        x = F.relu(self.fc1.forward(x))
        x = self.fc2(x)
        return x

class FixedNeuralNetwork:
    def __init__(self, in_dim, last_node,
                 in_fc2_dim=200,
                 problem_type='Classification',
                 sequence_length=10,
                 dim=[200, 200, 200], sub_model='LSTM'):

        self.in_dim = in_dim
        self.seq = True if sub_model in ['LSTM', 'GRU'] else False
        self.loss_measure = None
        self.metric = None
        self.define_parameters(problem_type)
        self.sequence_length = sequence_length
        self.problem_type = problem_type

        self.model = nn.Module()
        self.generate_model(in_dim, last_node, in_fc2_dim, dim, sub_model)
        self.optim = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.temp_x = []
        self.y_true, self.y_pred = [], []

    def define_parameters(self, problem_type):
        if problem_type == "Classification":
            self.loss_measure = nn.CrossEntropyLoss()
            self.metric = check_accuracy
        else:
            self.loss_measure = nn.L1Loss()
            self.metric = return_error

    def generate_model(self, in_dim, out_dim, inp_fc2_dim, dims, sub_model):
        self.model = StaticModel(in_dim, out_dim,
                                 in_fc2_dim=inp_fc2_dim, dims=dims,
                                 sub_model=sub_model)

    def get_pred(self, y_pred):
        assert self.problem_type == "Classification"
        y = []
        for row in y_pred:
            r = row.tolist()
            y.append(r.index(max(r)))
        return y

    @staticmethod
    def build_sequential_input(x):
        seq = []
        for i in x[:-1]:
            seq.append(i)
        seq.append(x[-1])
        x = torch.cat(seq, dim=0)
        return x

    def train(self, x, y):
        x = np.array(x)
        x = torch.from_numpy(x.astype(np.float32)).view(1, self.in_dim)
        self.temp_x.append(x)
        y = torch.tensor(y).view(1)
        y = y.float() if self.problem_type == 'Regression' else y.long()

        if self.seq:
            sequence_length = min(10, len(self.temp_x))
            x_t = self.temp_x[-1 * sequence_length:]
            x = self.build_sequential_input(x_t)

        self.optim.zero_grad()
        out = self.model(x).view(1)
        self.y_pred.append(out.item()) if self.problem_type == 'Regression' else self.y_pred.append(self.get_pred(out))
        self.y_true.append(y)

        loss = self.loss_measure(out, y)
        loss.backward()
        self.optim.step()

    def display_result(self, last_percent=1):
        assert last_percent <= 1
        last_ind = int(len(self.y_true)*(1-last_percent))

        if self.problem_type == 'Regression':
            return_error(self.y_true[last_ind:], self.y_pred[last_ind:])
            #plot_output(self.y_true[last_ind:], self.y_pred[last_ind:])
        else:
            self.metric(self.y_true[last_ind:], self.y_pred[last_ind:])











