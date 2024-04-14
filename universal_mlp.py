import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pickle

def classify_mlp(txt):
    mlp = pickle.load(open("static\\weights\\mlp.sav", 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    
    print ("module %s loaded" % module_url)
    vec =   model([txt])[0]
    y_pred = mlp.predict([vec])
    
    return y_pred[0]
    
    