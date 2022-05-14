import torch.nn as nn
from torch import optim
import torch
import numpy as np
import random
import math
from tqdm import tqdm 
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

def median_absolute_percentage_error(actual, predicted):
  return np.median((np.abs(np.subtract(actual, predicted)/ actual))) * 100

class Encoder(nn.Module):
  def __init__(self, device, num_stations, input_size, hidden_size, sc, epochs = 30, batch_size = 32, learning_rate = 0.001, num_layers=1, dropout=0):
    super(Encoder,self).__init__()
    self.device = device
    self.num_stations = num_stations
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.sc = sc
    self.epochs = epochs

    self.lstm1 = nn.LSTM(input_size*num_stations, hidden_size*num_stations, num_layers, batch_first=True, dropout=dropout)
    self.lstm2 = nn.LSTM(hidden_size*num_stations, 256*num_stations, batch_first=True, dropout=dropout)
    self.ln = nn.Linear(256, 1)
  
  def forward(self, x):
    out, (h, c) = self.lstm1(x)
    out, (h, c) = self.lstm2(out)
    out = torch.reshape(-1, self.num_stations, 256)    
    out = self.ln(out) 
    return out[:, -1, :], (h,c)
  
  def train(self, x_train, y_train, x_valid, y_valid):
    optimizer = optim.Adam(self.parameters(),lr=self.learning_rate)
    criterion = nn.MSELoss()
    for epoch in tqdm(range(self.epochs)):
      epoch_train_loss = 0
      n = int(len(x_train)/self.batch_size)
      for _ in range(n):
        index = random.randrange(0, n-self.batch_size)
        # x = x_train[index : index + self.batch_size].to(self.device)
        # y = y_train[index : index + self.batch_size].to(self.device)
        x = x_train[index : index + self.batch_size]
        y = y_train[index : index + self.batch_size]
        if self.device:
          x = x.cuda()
          y = y.cuda()

        batch_loss = 0 
        optimizer.zero_grad()
    
        outputs, _ = self.forward(x)
        batch_loss = criterion(outputs, y)
        batch_loss.backward()
        optimizer.step()
        epoch_train_loss += batch_loss.item()

      # print(f'\nEpoch: {epoch+1:02}')
      if epoch % 10 == 0:
        

        epoch_val_loss = 0
        with torch.no_grad():
          n_val = int((len(x_valid) - self.batch_size))
          for index in range(0, n_val, self.batch_size): 
            # x = x_valid[index : index + self.batch_size].to(self.device)
            # y = y_valid[index : index + self.batch_size].to(self.device)
            x = x_valid[index : index + self.batch_size]
            y = y_valid[index : index + self.batch_size]
            if self.device:
              x = x.cuda()
              y = y.cuda()
            batch_loss =  0
            outputs, _ = self.forward(x)
            batch_loss = criterion(outputs, y)
            epoch_val_loss += batch_loss.item()
        print(f'Train loss: {epoch_train_loss / n:.10f} \t Val loss: {epoch_val_loss / n_val:.10f}')

  def test(self, x_test, y_test):
    y_original = []
    y_predict = []
    with torch.no_grad():
      n_test = int((len(x_test) - self.batch_size))
      for index in range(0, n_test, self.batch_size):
        # x = x_test[index : index + self.batch_size].to(self.device)
        # y = y_test[index : index + self.batch_size].to(self.device)
        x = x_test[index : index + self.batch_size]
        y = y_test[index : index + self.batch_size]
        if self.device:
            x = x.cuda()
            y = y.cuda()
        outputs, _ = self.forward(x)
        
        y_predict += outputs.view(-1, 1).squeeze(-1).tolist()
        y_original += y.view(-1, 1).squeeze(-1).tolist()

    # y_predict = np.reshape(y_predict, (-1))
    # y_original = np.reshape(y_original, (-1))

    # rescale to true values
    # import pdb
    # pdb.set_trace()
    y_predict -= self.sc.min_[0]
    y_predict /= self.sc.scale_[0]
    y_original -= self.sc.min_[0]
    y_original /= self.sc.scale_[0]
    # y_pred_ = np.expand_dims(y_predict, 1)
    # y_preds = np.repeat(y_pred_, self.input_size, 1)

    # y_inv = self.sc.inverse_transform(y_preds)
    # y_pred_true = y_inv[:, 0]

    # y_orig_ = np.expand_dims(y_original, 1)
    # y_origs = np.repeat(y_orig_, self.input_size, 1)

    # y_inv_ori = self.sc.inverse_transform(y_origs)
    # y_orig_true = y_inv_ori[:, 0]

    # y_predict = y_pred_true
    # y_original = y_orig_true

    loss_mae = mean_absolute_error(y_original, y_predict)
    loss_rmse = mean_squared_error(y_original, y_predict, squared=False)
    loss_mape = mean_absolute_percentage_error(y_original, y_predict)*100
    r2 = r2_score(y_original, y_predict)
    loss_mdape = median_absolute_percentage_error(y_original, y_predict)

    return loss_mae, loss_rmse, loss_mape, r2,loss_mdape, y_predict, y_original

