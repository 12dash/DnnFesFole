from sklearn import preprocessing


def convert_label(y, le=None):
    if le is None:
        le = preprocessing.LabelEncoder()
        le.fit(y)
    y_labels = le.transform(y)

    return y_labels, le


def flatten_list(l):
    l = [j for i in l for j in i]
    return l
