from py_isear.isear_loader import IsearLoader
import numpy as np
import itertools
from preprocess import clean_data
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.utils import np_utils
from nltk.corpus import stopwords

data = ['TEMPER', 'TROPHO']
target = ['EMOT']
loader = IsearLoader(data,target)
dataset = loader.load_isear('data/isear.csv')

#load the text data and clean it
text_data_set = dataset.get_freetext_content()
#text_data_set = clean_data(text_data_set)

#load the target emotion classes 
target_set = np.asarray(dataset.get_target())
target_chain = itertools.chain(*target_set)
target_data = np.asarray(list(target_chain))

# One-hot encoding of target values
target_data = np_utils.to_categorical(target_data)
target_data = target_data[:,1:]

X_train, X_test, y_train, y_test = train_test_split(text_data_set, target_data, test_size = 0.2, shuffle = True)

vectorizer = CountVectorizer(strip_accents = 'ascii', lowercase = True, max_features = 7500)
#vectorizer = TfidfVectorizer(strip_accents = 'ascii', lowercase = True, max_features = 7900)
X_train_onehot = vectorizer.fit_transform(X_train)

word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
tokenize = vectorizer.build_tokenizer()
preprocess = vectorizer.build_preprocessor()
 
def to_sequence(tokenizer, preprocessor, index, text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes
 
print(to_sequence(tokenize, preprocess, word2idx, "test"))
X_train_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_train]

# Compute the max length of a text
MAX_SEQ_LENGTH = len(max(X_train_sequences, key=len))
print("MAX_SEQ_LENGTH=", MAX_SEQ_LENGTH)
 
from keras.preprocessing.sequence import pad_sequences
N_FEATURES = len(vectorizer.get_feature_names())
X_train_sequences = pad_sequences(X_train_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)
print(X_train_sequences[0])

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

"""
import spacy

nlp = spacy.load('en_core_web_md')
 
EMBEDDINGS_LEN = len(nlp.vocab['apple'].vector)
print("EMBEDDINGS_LEN=", EMBEDDINGS_LEN)  # 300
 
embeddings_index = np.zeros((len(vectorizer.get_feature_names()) + 1, EMBEDDINGS_LEN))
for word, idx in word2idx.items():
    try:
        embedding = nlp.vocab[word].vector
        embeddings_index[idx] = embedding
    except:
        pass
"""

GLOVE_PATH = 'data/ntua_twitter_affect_310.txt'
GLOVE_VECTOR_LENGTH = 310

def read_glove_vectors(path, lenght):
    embeddings = {}
    with open(path, encoding="utf8") as glove_f:
        for line in glove_f:
            chunks = line.split()
            assert len(chunks) == lenght + 1
            embeddings[chunks[0]] = np.array(chunks[1:], dtype='float32')
 
    return embeddings
 
GLOVE_INDEX = read_glove_vectors(GLOVE_PATH, GLOVE_VECTOR_LENGTH)
 
# Init the embeddings layer with GloVe embeddings
embeddings_index = np.zeros((len(vectorizer.get_feature_names()) + 1, GLOVE_VECTOR_LENGTH))
for word, idx in word2idx.items():
    try:
        embedding = GLOVE_INDEX[word]
        embeddings_index[idx] = embedding
    except:
        pass
 
from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, LSTM, Embedding, Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
 
model = Sequential()
"""
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    EMBEDDINGS_LEN,
                    weights = [embeddings_index],
                    input_length=MAX_SEQ_LENGTH,
                    trainable = False))
"""
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    GLOVE_VECTOR_LENGTH,  # Embedding size
                    weights=[embeddings_index],
                    input_length=MAX_SEQ_LENGTH,
                    trainable=False))
#model.add(CuDNNLSTM(EMBEDDINGS_LEN))
model.add(CuDNNLSTM(GLOVE_VECTOR_LENGTH))
model.add(Dense(units=7, activation='softmax'))
 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])
model.summary()

checkpointer = ModelCheckpoint(filepath='models/LSTM_NLP_v2_TWITTER.hdf5', verbose=1, save_best_only=True)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              #patience=3, min_lr=0.001)
early_stop = EarlyStopping(monitor='val_loss', patience = 3)

history = model.fit(X_train_sequences[:-300], y_train[:-300], 
              epochs=10, batch_size=32, verbose=1, 
              validation_data=(X_train_sequences[-300:], y_train[-300:]),
              shuffle = True,
              callbacks = [checkpointer, early_stop])

X_test_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_test]
X_test_sequences = pad_sequences(X_test_sequences, maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)

model.load_weights('models/LSTM_NLP_v2_TWITTER.hdf5')
scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1])  

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test_sequences)
matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

def plot_train_history_accuracy(history):
    # summarize history for loss
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'f1', 'val_f1', 'loss', 'val_loss'], loc='upper right')
    plt.show()

#plot_train_history_accuracy(history)
