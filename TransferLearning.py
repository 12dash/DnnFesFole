from Util.Preprocess import pre_process
from Util.Util import convert_label
from Util.ModelHelper import get_results
from EvolvingStructure import EvolvingStructure


def transfer_learning(df_train, df_test, problem_type='Regression'):
    processed_data = pre_process(df_train)
    x, y = processed_data["data"]

    processed_data_test = pre_process(df_test)
    x_test, y_test = processed_data_test["data"]

    le = None
    num_out_nodes = 1
    sub_model = ['LSTM']

    in_fc2_dim = 200
    dims = [100, 100]

    if problem_type != "Regression":
        num_out_nodes = len(set(y))
        y, le = convert_label(y)
        y_test = convert_label(y_test, le)

    for s in sub_model:
        print(s, end="  ")
        es = EvolvingStructure(col=processed_data["col"],
                               last_nodes=num_out_nodes,
                               problem_type=problem_type,
                               sub_model=s,
                               in_dim=in_fc2_dim,
                               sequence_length=10,
                               dims=dims)

        y_true, y_pred = [], []
        for i in range(len(x)):
            out = es.train(x[i], y[i])
            y_pred.append(out)
            y_true.append(y[i])

        print("Train Output", end=" ")
        get_results(y_true=y_true,
                    y_pred=y_pred,
                    problem_type=problem_type)

        es.clear_seq_data()

        y_true, y_pred = [], []
        for i in range(len(x_test)):
            out = es.train(x_test[i], y_test[i])
            y_pred.append(out)
            y_true.append(y_test[i])

        print("Test Output", end=" ")
        get_results(y_true=y_true,
                    y_pred=y_pred,
                    problem_type=problem_type)


