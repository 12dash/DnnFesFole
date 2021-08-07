from Util.FileHelper import *

from OnlineSimulation import simulate
from TransferLearning import transfer_learning

if __name__ == "__main__":
    dataset = "Traffic"
    make_dir(f"Runs/{dataset}")
    problem_type = "Classification"
    df = None
    if dataset == "Iris":
        column_list = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
        path = "./Data/Iris/iris.data"
        df = read_csv(path, column_list)
    elif dataset == "Diabetes":
        path = "./Data/MedicalData/Diabetes.csv"
        df = read_csv(path, sampling=1)
    elif dataset == "Traffic":
        path = "./Data/Traffic/D1_05_40_1c(s).txt"
        df = read_file(path, cols=['Lane_1', 'Lane_2', 'Lane_3', 'Lane_4', 'CLass'],
                       sampling=None, skiprows=1,
                       delimiter='\t')
        problem_type = "Regression"

    tl = False
    if tl:
        path1 = "./Data/Traffic/D3_45_40_3t(s).txt"
        path2 = "./Data/Traffic/D1_05_40_1c(s).txt"

        df_train = read_file(path1, cols=['Lane_1', 'Lane_2', 'Lane_3', 'Lane_4', 'CLass'],
                             sampling=None, skiprows=1,
                             delimiter='\t')
        df_test = read_file(path2, cols=['Lane_1', 'Lane_2', 'Lane_3', 'Lane_4', 'CLass'],
                            sampling=None, skiprows=1,
                            delimiter='\t')
        problem_type = "Regression"
        transfer_learning(df_train, df_test, problem_type)
    else:
        simulate(df, problem_type=problem_type, data_fuzzify=True, dataset=dataset)
