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
dataset = loader.load_isear('data/isear.csv')

#load the text data and clean it
text_data_set = dataset.get_freetext_content()
print(text_data_set[0])

#load the target emotion classes 
target_set = np.asarray(dataset.get_target())
target_chain = itertools.chain(*target_set)
target_data = np.asarray(list(target_chain))

# One-hot encoding of target values
encoded_target_data = np_utils.to_categorical(target_data)

stopWords = ["a", "and", "the", "an", "but", "how", "what"]

X_train, X_test, y_train, y_test = train_test_split(text_data_set, encoded_target_data, test_size = 0.2)

vectorizer = CountVectorizer(strip_accents = 'ascii', stop_words = stopWords, lowercase = True)
X_train_onehot = vectorizer.fit_transform(X_train)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
 
from keras.models import Sequential
from keras.layers import Dense 
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import keras.backend as K
 
model = Sequential()
model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=8, activation='softmax'))
 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])
model.summary()

checkpointer = ModelCheckpoint(filepath='MLP_NLP_v1.hdf5', verbose=1, save_best_only=True)
#model.load_weights('MLP_NLP_v1.hdf5')

history = model.fit(X_train_onehot[:-100], y_train[:-100], 
                    epochs=10, batch_size=128, verbose=1, 
                    validation_data=(X_train_onehot[-100:], y_train[-100:]), shuffle = True,
                    callbacks = [checkpointer])

scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
print("Accuracy:", scores[1])  # Accuracy: 0.875
 
def plot_train_history_accuracy(history):
    # summarize history for loss
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'f1', 'val_f1'], loc='upper right')
    plt.show()

plot_train_history_accuracy(history)
