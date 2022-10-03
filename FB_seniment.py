import streamlit as st
import filetype
import numpy as np
from io import StringIO
import pandas as pd
import sys, os, re, csv, math, codecs, numpy as np, pandas as pd
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D
from keras.layers import MaxPool1D, Flatten, Conv1D, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.layers import concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras import initializers, regularizers, constraints, optimizers, layers
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
import tweepy
from tweepy import OAuthHandler
from facebook_scraper import get_posts
tokenizer = Tokenizer()

from keras.models import load_model
model = load_model('comment_sentiment_type_model.h5')
class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
count=10
input_ = st.text_input('Input username', '#India')
st.write('Your input', input_)
if st.button('Predict_sentiment'):
    listposts = []
    for post in get_posts("MetaIndia", pages=1):#366190054572553
        txt=post['text'][:100]
        listposts.append(post)
        seq = tokenizer.texts_to_sequences(txt)
        padded = pad_sequences(seq, maxlen=500)

        start_time = time.time()
        pred = model.predict(padded)
        st.write('Message: ' + str(txt))
        st.write('🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🤖🤖🤖🤖🤖🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠')
        st.write('Predicted: {} ({:.2f} seconds)'.format(class_names[np.argmax(pred)], (time.time() - start_time)))