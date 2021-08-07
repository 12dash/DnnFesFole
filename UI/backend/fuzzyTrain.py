from pre_process import read_data
from Util.Preprocess import pre_process
from Util.Util import convert_label
from EvolvingStructure import EvolvingStructure
from NeuralNetwork.FixedNeuralNetwork import FixedNeuralNetwork


def get_structure(dataset, dataframe):
    df, problem_type = read_data(dataset, dataframe)
    processed_data = pre_process(df)
    x, y = processed_data["data"]

    sub_model = 'LSTM'
    in_fc2_dim = 200
    dims = [100, 100, 100]
    num_out_nodes = 1

    if problem_type != "Regression":
        num_out_nodes = len(set(y))
        y, le = convert_label(y)

    es = EvolvingStructure(col=processed_data["col"],
                           last_nodes=num_out_nodes,
                           problem_type=problem_type,
                           sub_model=sub_model,
                           in_dim=in_fc2_dim,
                           sequence_length=10,
                           dims=dims)

    return es, x, y


def train(dataset, dataframe):
    df, problem_type = read_data(dataset, dataframe)
    processed_data = pre_process(df)
    x, y = processed_data["data"]

    sub_model = 'LSTM'
    in_fc2_dim = 200
    dims = [100, 100, 100]
    num_out_nodes = 1

    if problem_type != "Regression":
        num_out_nodes = len(set(y))
        y, le = convert_label(y)

    es = EvolvingStructure(col=processed_data["col"],
                           last_nodes=num_out_nodes,
                           problem_type=problem_type,
                           sub_model=sub_model,
                           in_dim=in_fc2_dim,
                           sequence_length=10,
                           dims=dims)
    
    y_true, y_pred =[], []    
    for i in range(len(x)):
        y_pred.append(es.train(x[i], y[i]))
        y_true.append(y[i])

    data = es.return_result(y_true, y_pred)
    return data
