import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

seed = 52

def process_outliers(x):  
  q1 = x.quantile(0.25)
  q3 = x.quantile(0.75)
  iqr = q3 - q1
  new_x = x.clip(lower=(q1 - 1.5*iqr), upper=(q3 + 1.5*iqr)) 
  return new_x

def scale_data(df):
  sc = MinMaxScaler()
  df_scaled = pd.DataFrame(sc.fit_transform(df), columns=df.columns)
  return df_scaled, sc 

def get_data(data_link, start_date = None, end_date = None):
  #read data
  in_data = pd.read_csv(data_link)
  in_data['time'] = pd.to_datetime(in_data['time'])
  if start_date is not None and end_date is not None:
    in_data = in_data.loc[(in_data['time'] > start_date) & (in_data['time'] < end_date)]
  in_data.set_index('time', inplace=True)
  
  # in_data = in_data.drop('time', axis=1)
  #fill NA
#   in_data.interpolate(method='ffill', limit_direction='forward', axis=0, inplace=True)
  #process outlier
  in_data['PM2.5'] = process_outliers(in_data['PM2.5'])
  in_data['humidity'] = process_outliers(in_data['humidity'])
  in_data['temperature'] = process_outliers(in_data['temperature'])
  #make data set
  data_in, sc_in = scale_data(in_data)
  return data_in, sc_in
  
def get_train_valid_test(X, Y):
  # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, shuffle = False, random_state=seed)
  # x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size = 0.5, shuffle = False, random_state=seed)
  x_train = X[:int(X.shape[0]*0.6)]
  y_train = Y[:int(Y.shape[0]*0.6)]
  x_valid = X[int(X.shape[0]*0.6):int(X.shape[0]*0.8)]
  y_valid = Y[int(Y.shape[0]*0.6):int(Y.shape[0]*0.8)]
  x_test = X[int(X.shape[0]*0.8):]
  y_test = Y[int(Y.shape[0]*0.8):]
  return x_train, x_valid, x_test, y_train, y_valid, y_test 

def make_data_set(data_link, seq_len,output_len = 1, start_date = None, end_date = None):
    data_in, sc = get_data(data_link, start_date, end_date)
    X = torch.FloatTensor(np.array([data_in[i:i+seq_len] for i in range(len(data_in) - seq_len - output_len)]))
    Y = torch.FloatTensor(np.array([data_in['PM2.5'][i+seq_len:i+seq_len + output_len] for i in range(len(data_in) - seq_len - output_len)]))
    x_train, x_valid, x_test, y_train, y_valid, y_test = get_train_valid_test(X, Y)
    return x_train, x_valid, x_test, y_train, y_valid, y_test, sc
