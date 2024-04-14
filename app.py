import flask
from flask import Flask,render_template,request
import numpy as np
import os
from werkzeug.utils import secure_filename
from pre_process import *
from distil_model import * 
from universal_mlp import *
from fetch_news import *
from lstm import *
import json

app = Flask(__name__)

classes=["World", "Sports", "Business","Tech"]

@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/classify",methods=['POST'])
def classify_text():
    txt = request.form['txt']
    txt_clean = clean_text(txt)
    y_pred_mlp = classify_mlp(txt_clean)
    y_pred_bert = distil_model_single(txt)
    y_pred_lstm = classify_LSTM(txt_clean)
    #print(f"MLP Predicted: {classes[y_pred_mlp]}")
    #print(f"BERT Predicted: {classes[y_pred_bert]}")
    
    to_render = {
        "txt":txt,
        "MLP": classes[y_pred_mlp],
        #"MLP": "Sports",
        "LSTM Neural Network": classes[y_pred_lstm],
        #"LSTM Neural Network": "Sports",
        "BERT": classes[y_pred_bert]
        #"BERT": "Sports"
        
    }
    
    return render_template("display.html",to_render = to_render)

@app.route("/fetch",methods=['GET'])
def fetch_news():
    all_news_desc,all_news_links,all_news_tups = call_news()
    preds = distil_model_multiple(all_news_desc)
    print(preds.tolist())
    pred_classes = [classes[i] for i in preds.tolist()]
    to_render_dict = {} 
    for i in range(len(pred_classes)):
        if pred_classes[i] not in to_render_dict.keys(): 
            to_render_dict[pred_classes[i]]=[]
        to_render_dict[pred_classes[i]].append({"title":all_news_tups[i][0],"link":all_news_tups[i][2]})
    print(to_render_dict)
    return render_template("all_news.html",to_render = to_render_dict)