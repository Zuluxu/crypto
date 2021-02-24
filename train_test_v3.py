import pandas as pd
import numpy as np

np.random.seed(0)
dataframe = pd.read_csv("data_sample_BR17.csv")
timeframeh = 4
timeframem = timeframeh
forecast_deep = 3 * timeframem
nb_obs_day = 1
lookback = 24 * timeframem * nb_obs_day
tt_ratio = 0.7
dataset_close = dataframe['Close'].round(decimals=4)
dataset_close = pd.to_numeric(dataset_close, downcast='float')


def create_dataset(dataset):
    dX, dY = [], []
    n = 0
    for i in range(lookback + 1, len(dataset) - forecast_deep):
        a = dataset[i - 1 - lookback:i - 1]
        a2 = np.round(np.log(a) - np.log(dataset_close.loc[i]), decimals=6)

        dX.append(a2)

        # b = dataset[i + 1:i + forecast_deep].values
        b = dataset[i + 2*forecast_deep-1]
        b2 = np.round(np.log(b) - np.log(dataset_close.loc[i]), decimals=6)
        b2 = np.where(b2 >= 0, 1, b2)
        b2 = np.where(b2 < 0, 0, b2)
        dY.append(b2)

        j = i
        if j >= n:
            print(j)
            n += 1000

    return np.array(dX), np.array(dY)


def split_data(ratio, array):
    train = array[lookback + 1:int(np.round(len(array) * ratio))]
    test = array[int(np.round(len(array) * ratio) + 1):len(array) - forecast_deep - 1]

    return train, test


dataX, dataY = create_dataset(dataset_close)
trainX, testX = split_data(tt_ratio, dataX)
trainY, testY = split_data(tt_ratio, dataY)

np.savetxt('trainX.csv', trainX, delimiter=",")
np.savetxt('testX.csv', testX, delimiter=",")
np.savetxt('trainY.csv', trainY, delimiter=",")
np.savetxt('testY.csv', testY, delimiter=",")
