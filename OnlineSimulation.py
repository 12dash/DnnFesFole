from Util.Preprocess import pre_process
from Util.Util import convert_label
from Util.ModelHelper import get_results
from EvolvingStructure import EvolvingStructure
from NeuralNetwork.FixedNeuralNetwork import FixedNeuralNetwork


def simulate(dataframe, problem_type='Regression', data_fuzzify=True, dataset=None):
    processed_data = pre_process(dataframe)
    x, y = processed_data["data"]

    le = None
    num_out_nodes = 1
    sub_model = ['MLP', 'LSTM', 'GRU']

    in_fc2_dim = 50
    dims = [200, 200]

    if problem_type != "Regression":
        num_out_nodes = len(set(y))
        y, le = convert_label(y)

    if data_fuzzify:
        for s in sub_model:
            path = f"./Runs/{dataset}/out_{s}.png"
            print(s, end="  ")
            es = EvolvingStructure(col=processed_data["col"],
                                   last_nodes=num_out_nodes,
                                   problem_type=problem_type,
                                   sub_model=s,
                                   in_dim=in_fc2_dim,
                                   sequence_length=10,
                                   dims=dims)
            y_pred, y_true = [], []
            for i in range(len(x)):
                y_true.append(y[i])
                out = es.train(x[i], y[i])
                y_pred.append(out)

            print("Fuzzifying the inputs", end=" ")
            get_results(y_true=y_true,
                        y_pred=y_pred,
                        last_percent=1,
                        problem_type=problem_type,
                        path=path)
            es.plot_membership(dataset)
    else:
        print("-"*30)
        for s in sub_model:
            print(s, end="  ")
            fn = FixedNeuralNetwork(in_dim=len(processed_data["col"][0]),
                                    last_node=num_out_nodes,
                                    in_fc2_dim=in_fc2_dim,
                                    problem_type=problem_type,
                                    dim=dims,
                                    sequence_length=10,
                                    sub_model=s)

            for i in range(len(x)):
                x_row = [x[i][j] for j in processed_data['col'][0]]
                fn.train(x_row, y[i])

            print("Without fuzzification the inputs", end=" ")
            fn.display_result(last_percent=1)
