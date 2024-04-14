import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

import pandas as pd
import numpy as np 

def classify_LSTM(text):
    df_train = pd.read_csv("static\\weights\\train\\train.csv")
    #df_test = pd.read_csv("/content/drive/MyDrive/NUS/Hands-on-Analytics/Project/Data/test.csv")
    df_train = df_train.drop(["Unnamed: 0"],axis=1)
    #df_test = df_test.drop(["Unnamed: 0"],axis=1)
    X_train, y_train = df_train["cleaned_text"],df_train["label"]
    
    tokenizer = Tokenizer()
    # build the vocabulary based on train dataset
    tokenizer.fit_on_texts(X_train)
    # tokenize the train and test dataset
    X_train = tokenizer.texts_to_sequences(X_train)
    text = tokenizer.texts_to_sequences([text])

    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(seq) for seq in X_train)


    X_train = pad_sequences(X_train, maxlen = max_length)
    text = pad_sequences(text, maxlen = max_length)
    #print(X_train.shape)
    #print(text.shape)
    output_dim = 200
    
    #model.load_weights('static\\weights\\LSTM_weights.h5')
    model = tf.keras.models.load_model('static\\weights\\LSTM_final.h5')
    y_pred = model.predict(text)[0]
    #print(y_pred)
    return np.argmax(y_pred)

#classify_LSTM("Man City defeate Liverpool 1-0 and secure PL")