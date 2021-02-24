import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from tensorflow.keras import Sequential
from sklearn.model_selection import KFold
import keras.backend as K

num_folds = 5
np.random.seed(0)
learning_rate = 0.001
activation_function1 = 'tanh'
activation_function2 = 'softmax'
adam = Adam(lr=learning_rate)
batch_size = 50
num_epochs = 5


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def reshape(X):
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    return X


trainX = reshape(pd.read_csv('trainX.csv').to_numpy())
testX = reshape(pd.read_csv('testX.csv').to_numpy())
trainY = reshape(pd.read_csv('trainY.csv').to_numpy())
testY = reshape(pd.read_csv('testY.csv').to_numpy())

inputs = np.concatenate((trainX, testX), axis=0)
targets = np.concatenate((trainY, testY), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=False)

# K-fold Cross Validation model evaluation
fold_no = 1
acc_per_fold = []
loss_per_fold = []

for train, test in kfold.split(inputs, targets):
    # Define the model architecture

    model = Sequential()
    model.add(LSTM(96, activation=activation_function1, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(LSTM(96, activation=activation_function2))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(1, activation=activation_function2))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy', f1_score])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(inputs[train], targets[train],
                        batch_size=batch_size,
                        epochs=5,
                        verbose=1,
                        shuffle=False,
                        validation_split=0.33)

    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(scores)
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

plt.plot(acc_per_fold)
plt.plot(loss_per_fold)
