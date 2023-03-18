def pre_process(dataframe, target_col=None):
    cols = list(dataframe.columns)
    if target_col is None:
        target_col = cols[-1]
    inp_col = cols[:-1]
    x = dataframe[inp_col].to_dict('records')
    y = dataframe[target_col].to_list()
    return {"data": (x, y), "col": (inp_col, target_col)}

