import matplotlib.pyplot as plt
import matplotlib.animation as ani


def get_list_from_dic(dic):
    x, y = [], []
    for i in sorted(dic.keys()):
        x.append(i)
        y.append(dic[i])

    return x, y


def show_plot(a, feature, return_plot=False):
    plt.xlabel(feature)
    plt.ylabel("Membership Grade")
    for i in a:
        x, y = get_list_from_dic(i)
        plt.plot(x, y)
    if return_plot:
        return plt
    else:
        #plt.show()
        pass


def save_plot(a, feature, path):
    plt = show_plot(a, feature, return_plot=True)
    plt.savefig(path)
    return


def plot_output(y_true, y_pred, path):
    x = [i for i in range(len(y_pred))]
    plt.figure(figsize=(24, 12))
    plt.plot(x, y_true, '-o', linewidth=0.4, markersize=2)
    plt.plot(x, y_pred, '-o', linewidth=0.4, markersize=2)

    plt.legend(['Y True', 'Y Pred'])
    plt.savefig(path)

