import torch

from FuzzySets.Fuzzify import Fuzzify
from NeuralNetwork.EvolvingNeuralNetwork import EvolvingNN
from Util.ModelHelper import *
from Util.Util import *


class EvolvingStructure:
    def __init__(self, col, last_nodes=1,
                 problem_type="Classification",
                 sub_model='LSTM',
                 sequence_length=10,
                 in_dim=100,
                 dims=[100, 100, 100]):

        self.fuzzy_feature = {}
        self.problem_type = problem_type
        self.last_nodes = last_nodes

        self.inp_col, self.target_col = col
        self.training_fuzzy_input = []

        self.build_fuzzy_feature()

        self.loss_measure = None
        self.metric = None
        self.define_parameters(problem_type)

        self.sub_model = sub_model
        self.sequence_length = sequence_length

        self.model = EvolvingNN(last_nodes,
                                sub_model=self.sub_model,
                                in_dim=in_dim,
                                dims=dims)
        self.optim = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def define_parameters(self, problem_type):
        if problem_type == "Classification":
            self.loss_measure = nn.CrossEntropyLoss()
            self.metric = check_accuracy
        else:
            self.loss_measure = nn.MSELoss()
            self.metric = return_error

    def clear_seq_data(self):
        self.training_fuzzy_input = []

    def build_fuzzy_feature(self):
        for feature in self.inp_col:
            self.fuzzy_feature[feature] = Fuzzify(feature)

    def build_input(self, x_row, add_point=True):
        x = []
        for feature in self.inp_col:
            point = x_row[feature]
            fuzzifier = self.fuzzy_feature[feature]
            if add_point:
                fuzzifier.add_point(point)
                if fuzzifier.merged_nodes:
                    self.model.merge_nodes(fuzzifier.merged_nodes[0], fuzzifier.merged_nodes[1])
                fuzzifier.merged_nodes = []
            x.append(fuzzifier.get_membership_val(point))
        x = flatten_list(x)
        self.training_fuzzy_input.append(x)
        return x

    def forward_pass(self, x):
        def get_seq():
            sequence_length = min(self.sequence_length, len(self.training_fuzzy_input))
            return self.training_fuzzy_input[-1 * sequence_length:]

        if self.problem_type == 'Classification':
            if self.sub_model != 'MLP':
                x = get_seq()
            out = self.model(x)
            out = out.view(1, self.last_nodes)
        else:
            if self.sub_model != 'MLP':
                x = get_seq()
            out = self.model(x)
            out = out.view(1)

        return out

    def update_model(self, x, y):
        self.optim.zero_grad()
        y = torch.tensor(y).view(1)

        y = y.float() if self.problem_type == 'Regression' else y.long()
        out = self.forward_pass(x)

        loss = self.loss_measure(out, y)
        loss.backward()

        self.optim.step()
        self.model.backward_compute()

        if self.problem_type == "Classification":
            out = get_pred(out)

        return out

    def update(self, x_row, y):
        x = self.build_input(x_row)
        out = self.update_model(x, y)
        if torch.is_tensor(out):
            out = out.item()
        return out

    def forward(self, x, y):
        x = self.build_input(x, add_point=False)
        out = self.forward_pass(x)
        if torch.is_tensor(out):
            out = out.item()
        return out

    def train(self, x, y):
        return self.update(x, y)

    def return_result(self, y_true, y_pred, last_percent=1):
        assert last_percent <= 1
        last_ind = int(len(y_true)*(1-last_percent))
        data = {}

        if self.problem_type == 'Regression':
            error = return_error(y_true[last_ind:], y_pred[last_ind:])
        else:
            error = self.metric(y_true[last_ind:], y_pred[last_ind:])
            y_pred = [i[0] for i in y_pred]
        data['score'] = error
        data['true'] = y_true
        data['pred'] = y_pred
        return data

    def plot_membership(self, dataset):
        print('Plotting Membership')
        for i in self.fuzzy_feature:
            self.fuzzy_feature[i].plot_membership(dataset)
