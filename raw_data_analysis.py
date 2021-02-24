# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 13:35:47 2021

@author: SusuRog
"""
import pandas as pd
import numpy as np

timeframeh = 4
lookbacks_µ = [21, 50, 200]
lookbacks_ema = [12, 26]
np.random.seed(0)

raw_data = pd.read_csv("bitfinex_tBTCUSD_1m.csv")
dataframe_BR17 = raw_data.iloc[235:len(raw_data) - 1, [0, 1, 5]]

pd.DataFrame(dataframe_BR17).to_csv("raw_dataframe.csv")
dataframe_BR17['Time'] = pd.to_datetime(dataframe_BR17['Date'] + ' ' + dataframe_BR17['Time'])
dataframe_BR17 = dataframe_BR17.set_index('Time')
dataframe_BR17 = dataframe_BR17.iloc[:, [1]]
dataframe_BR17 = dataframe_BR17.resample("1H").bfill()

pd.DataFrame(dataframe_BR17).to_csv("data_sample.csv")

dataframe_BR17['MA21'] = dataframe_BR17.iloc[:, 0].rolling(window=lookbacks_µ[0] * timeframeh).mean()
dataframe_BR17['MA50'] = dataframe_BR17.iloc[:, 0].rolling(window=lookbacks_µ[1] * timeframeh).mean()
dataframe_BR17['MA200'] = dataframe_BR17.iloc[:, 0].rolling(window=lookbacks_µ[2] * timeframeh).mean()

dataframe_BR17['Sigma21'] = dataframe_BR17.iloc[:, 0].rolling(window=lookbacks_µ[0] * timeframeh).std()
dataframe_BR17['Sigma50'] = dataframe_BR17.iloc[:, 0].rolling(window=lookbacks_µ[1] * timeframeh).std()
dataframe_BR17['Sigma200'] = dataframe_BR17.iloc[:, 0].rolling(window=lookbacks_µ[2] * timeframeh).std()

dataframe_BR17['Bollinger High 21'] = dataframe_BR17['MA21'] + (dataframe_BR17['Sigma21'] * 2)
dataframe_BR17['Bollinger Low 21'] = dataframe_BR17['MA21'] - (dataframe_BR17['Sigma21'] * 2)
dataframe_BR17['Bollinger High 50'] = dataframe_BR17['MA50'] + (dataframe_BR17['Sigma50'] * 2)
dataframe_BR17['Bollinger Low 50'] = dataframe_BR17['MA50'] - (dataframe_BR17['Sigma50'] * 2)
dataframe_BR17['Bollinger High 200'] = dataframe_BR17['MA200'] + (dataframe_BR17['Sigma200'] * 2)
dataframe_BR17['Bollinger Low 200'] = dataframe_BR17['MA200'] - (dataframe_BR17['Sigma200'] * 2)

dataframe_BR17['Delta_EMA_12_26'] = dataframe_BR17.iloc[:, 0].ewm(
    span=lookbacks_µ[1] * timeframeh).mean() - dataframe_BR17.iloc[:, 0].ewm(span=lookbacks_µ[0] * timeframeh).mean()

print(dataframe_BR17['log_ret_12h'].value_counts())

pd.DataFrame(dataframe_BR17).to_csv("data_sample_BR17.csv", index=False)
