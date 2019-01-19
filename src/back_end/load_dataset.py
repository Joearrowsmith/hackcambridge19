from py_isear.isear_loader import IsearLoader
import numpy as np
import itertools
#from preprocess import clean_data
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import pandas as pd

data = ['TEMPER', 'TROPHO']
target = ['EMOT']
loader = IsearLoader(data,target)
dataset = loader.load_isear('isear.csv')

#load the text data and clean it
text_data_set = dataset.get_freetext_content()
print(text_data_set[0])

#load the target emotion classes 
target_set = np.asarray(dataset.get_target())
target_chain = itertools.chain(*target_set)
target_data = np.asarray(list(target_chain))

# One-hot encoding of target values
encoded_target_data = np_utils.to_categorical(target_data)

#count_vect = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
stopWords = ["a", "and", "the", "an", "but", "how", "what"]

X_train, X_test, y_train, y_test = train_test_split(text_data_set, encoded_target_data, test_size = 0.2)

vectorizer = CountVectorizer(stop_words = stopWords, lowercase = True)
X_train_onehot = vectorizer.fit_transform(X_train)

from keras.models import Sequential
from keras.layers import Dense
 
model = Sequential()
 
model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=8, activation='softmax'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train_onehot[:-100], y_train[:-100], 
          epochs=2, batch_size=128, verbose=1, 
          validation_data=(X_train_onehot[-100:], y_train[-100:]))

scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
print("Accuracy:", scores[1])  # Accuracy: 0.875
 
