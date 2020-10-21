# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:41:06 2020

@author: mirco marastoni
"""

import pandas as pd
import numpy as np
import os
import sklearn.datasets as skd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from cleantext import clean

#keras
from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten
from keras.preprocessing import text
from keras import layers
from keras.utils import to_categorical

#matplotlib
import matplotlib.pyplot as plt



# creating a dataset with colums: ['comments','target']
# target = 0 means 'others' target label
# target = 1 means 'concerns' targte label

def clean_text(df):
    commentList = df.to_list()
    # if '"' in commentList[i]:
    #     commentList[i] = commentList[i].replace('"','')
    # if ';' in commentList[i]:
    #     commentList[i] = commentList[i].replace(';','')
    for i in range(len(commentList)):
        commentList[i] = clean(commentList[i],
                fix_unicode=True,               # fix various unicode errors
                to_ascii=True,                  # transliterate to closest ASCII representation
                lower=True,                     # lowercase text
                no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
                no_urls=True,                  # replace all URLs with a special token
                no_emails=True,                # replace all email addresses with a special token
                no_phone_numbers=True,         # replace all phone numbers with a special token
                no_numbers=True,               # replace all numbers with a special token
                no_digits=True,                # replace all digits with a special token
                no_currency_symbols=True,      # replace all currency symbols with a special token
                no_punct=True,                 # remove punctuations
                replace_with_punct="",          # instead of removing punctuations you may replace them
                replace_with_url="<URL>",
                replace_with_email="<EMAIL>",
                replace_with_phone_number="<PHONE>",
                replace_with_number="<NUMBER>",
                replace_with_digit="0",
                replace_with_currency_symbol="<CUR>",
                lang="en"                       # set to 'de' for German special handling
                )
    return commentList


content_concerns = []
content_others = []
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        file_path = os.path.join(root, name)
        print(file_path)
        categories = ['usefull','misc']
        if 'concerns' in file_path:
            with open(file_path,'r+') as f:
                content_concerns.extend(f.readlines())
        a1 = pd.Series(content_concerns,name='comments')
        a = np.ones(shape=len(a1))
        a2 = pd.Series(a,name='labels')
        dataset_1 = pd.concat([a1, a2], axis=1)
        if 'others' in file_path:
            with open(file_path,'r+') as f:
                content_others.extend(f.readlines())
        b1 = pd.Series(content_others,name='comments')
        b = np.zeros(shape=len(b1))
        b2 = pd.Series(b,name='labels')
        dataset_2 = pd.concat([b1,b2],axis=1)
        
df = pd.concat([dataset_1,dataset_2],axis=0,ignore_index=True)
#random mix
df = df.sample(frac=1).reset_index(drop=True)



df_x = df['comments'] 
df_y = df['labels']

#clean_text
# df_x = clean_text(df_x)

#split train test
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)

#vectorization

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = max(len(x) for x in x_train) # longest text in train set

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

#model
embedding_dim = 32

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Fit model
history = model.fit(x_train, y_train,
                    epochs=5,
                    verbose=True,
                    validation_data=(x_test, y_test),
                    batch_size=50)
loss, accuracy = model.evaluate(x_train, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))               
# pred = model.predict_classes(x_test)
# pred = np.where(pred==1, 'usefull', pred) 
# pred = np.where(pred==0, 'misc', pred) 

model.save('model.h5')

#visualize accuracy

# plt.style.use('ggplot')
# def plot_history(history):
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     x = range(1, len(acc) + 1)
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(x, acc, 'b', label='Training acc')
#     plt.plot(x, val_acc, 'r', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(x, loss, 'b', label='Training loss')
#     plt.plot(x, val_loss, 'r', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()
# plot_history(history)

#save & load model
# from keras.models import load_model

# model.save('my_model.h5')
# model_load = load_model('my_model.h5')









