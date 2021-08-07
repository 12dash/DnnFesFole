import sys
sys.path.append("../../Code")

from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
import os
import pandas as pd

from fuzzyTrain import train, get_structure
app = Flask(__name__)
CORS(app, support_credentials=True)

es=None
x,y = None,None

@app.route('/get_dirs')
def get_option():    
    data_dir = "../../Code/Data/"
    path = request.args.get("dataset")
    l = []
    for i in os.listdir(f"{data_dir}/{path}"):
        l.append(i)
    paths = {'paths': l}
    return paths

@app.route('/begin_train')
def begin_train():
    dataset = request.args.get("dataset")
    dataframe = request.args.get("dataframe")
    data = train(dataset, dataframe)
    return data

@app.route('/train_online')
def online_train():
    global es
    global x
    global y
    dataset = request.args.get("dataset")
    dataframe = request.args.get("dataframe")
    i = int(request.args.get("i"))
    if es is None:
        es, x, y = get_structure(dataset, dataframe)

    if i < len(x):
        es.train(x[i],y[i])
        data = es.return_result()
        return data
        
    return -1

@app.route('/')
def main():
    cont = "Backend Server for FYP"
    return cont

if __name__ == "__main__":
    app.run(debug=True)