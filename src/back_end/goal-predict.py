import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

X = pd.read_csv('data/health-data/health_data.csv', delimiter = ';')
y = pd.read_csv('data/health-data/goal_achieved.csv', delimiter = ';')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)

model = Sequential()
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

checkpointer = ModelCheckpoint(filepath='models/MLP_goal_prob.hdf5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=1, min_lr=0.001)
early_stop = EarlyStopping(monitor='val_loss', patience = 2)

history = model.fit(X_train_sequences[:-1000], y_train[:-1000], 
              epochs=10, batch_size=32, verbose=1, 
              validation_data=(X_train_sequences[-1000:], y_train[-1000:]),
              shuffle = True,
              callbacks = [checkpointer, reduce_lr, early_stop])

def plot_train_history_accuracy(history):
    # summarize history for loss
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'loss', 'val_loss'], loc='upper right')
    plt.show()

plot_train_history_accuracy(history)

model.load_weights('models/MLP_goal_prob.hdf5')
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy:", scores[1])

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(matrix, classes = list(X), normalize=True,
                      title='Normalized confusion matrix')

plt.show()

