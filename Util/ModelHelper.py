from collections import OrderedDict
import torch
from torch import nn
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from Util.VisualizationHelper import plot_output


def get_pred(self, y_pred):
    assert self.problem_type == "Classification"
    y = []
    for row in y_pred:
        r = row.tolist()
        y.append(r.index(max(r)))
    return y


def check_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy : ", accuracy)
    return accuracy


def calculate_mse(model, inp, out):
    with torch.no_grad():
        pred = [model(i).item() for i in inp]
    _ = return_error(out, pred)


def return_error(out, pred, mse=True):
    if not mse:
        error = mean_absolute_error(out, pred)
        print("Mean Absolute Error : ", error)
    else:
        error = mean_squared_error(out, pred)
        print("Mean Squared Error : ", error)
    return error


def print_model(model):
    for name, param in model.named_parameters():
        print(name, param)


def build_layers(in_dim, out_dim, dims, layer_type='linear'):
    layers = OrderedDict()
    if layer_type == 'lstm':
        layers['lstm_1'] = nn.LSTM(in_dim, dims[0])
    elif layer_type == 'gru':
        layers['gru'] = nn.GRU(in_dim, dims[0])
    else:
        layers['linear'] = nn.Linear(in_dim, dims[0])
        layers[f'act_{0}'] = nn.ReLU()
    k = 1
    for i in range(len(dims) - 1):
        layers[f'layer_{k}'] = nn.Linear(dims[i], dims[i + 1])
        layers[f'act_{k}'] = nn.ReLU()
        k = k + 1
    layers[f'layer_{k}'] = nn.Linear(dims[-1], out_dim)
    return nn.ModuleDict(layers)


def get_results(y_true, y_pred, last_percent=1, problem_type='Regression', path=None):
    assert last_percent <= 1
    last_ind = int(len(y_true) * (1 - last_percent))

    if problem_type == 'Regression':
        return_error(y_true[last_ind:], y_pred[last_ind:])
        plot_output(y_true[last_ind:], y_pred[last_ind:], path)
    else:
        check_accuracy(y_true[last_ind:], y_pred[last_ind:])