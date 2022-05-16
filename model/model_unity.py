import torch.nn as nn
from torch import optim
import torch
import numpy as np
import random
import os
from tqdm import tqdm 
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

def median_absolute_percentage_error(actual, predicted):
  return np.median((np.abs(np.subtract(actual, predicted)/ actual))) * 100

class Encoder(nn.Module):
  def __init__(self, log, device, num_stations, input_size, hidden_size, sc, epochs = 30, batch_size = 32, learning_rate = 0.001, num_layers=1, dropout=0):
    super(Encoder,self).__init__()
    self.log = log
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
    out, _ = self.lstm1(x)
    out, _ = self.lstm2(out)
    out = out[:, -1, :]
    out = torch.reshape(out, (-1, self.num_stations, 256))
    out = self.ln(out) 
    return out
  
  def train(self, x_train, y_train, x_valid, y_valid, station_name = None):
    optimizer = optim.Adam(self.parameters(),lr=self.learning_rate)
    criterion = nn.MSELoss()
    for epoch in tqdm(range(self.epochs)):
      epoch_train_loss = 0
      n = int(len(x_train)/self.batch_size)
      for _ in range(n):
        index = random.randrange(0, n-self.batch_size)
        x = x_train[index : index + self.batch_size]
        y = y_train[index : index + self.batch_size]
        if self.device:
          x = x.cuda()
          y = y.cuda()

        batch_loss = 0 
        optimizer.zero_grad()
        outputs = self.forward(x)
        batch_loss = criterion(outputs, y)
        batch_loss.backward()
        optimizer.step()
        epoch_train_loss += batch_loss.item()

      if epoch % 10 == 0:
        epoch_val_loss = 0
        with torch.no_grad():
          n_val = int((len(x_valid) - self.batch_size))
          for index in range(0, n_val, self.batch_size): 
            x = x_valid[index : index + self.batch_size]
            y = y_valid[index : index + self.batch_size]
            if self.device:
              x = x.cuda()
              y = y.cuda()
            batch_loss =  0
            outputs = self.forward(x)
            batch_loss = criterion(outputs, y)
            epoch_val_loss += batch_loss.item()
        print(f'Train loss: {epoch_train_loss / n:.10f} \t Val loss: {epoch_val_loss / n_val:.10f}')
    torch.save(self.state_dict(), os.path.join(self.log, 'last.pt'))

  def test(self, x_test, y_test, station_name = None):
    self.load_state_dict(torch.load(os.path.join(self.log, 'last.pt')))
    # self.eval()
    y_original = []
    y_predict = []
    with torch.no_grad():
      n_test = int((len(x_test) - self.batch_size))
      for index in range(0, n_test, self.batch_size):
        x = x_test[index : index + self.batch_size]
        y = y_test[index : index + self.batch_size]
        if self.device:
            x = x.cuda()
            y = y.cuda()
        outputs = self.forward(x)
        y_predict.append(outputs)
        y_original.append(y)
    y_predict = torch.cat(y_predict, dim=0).cpu().detach().numpy()
    y_original = torch.cat(y_original, dim=0).cpu().detach().numpy()

    loss_mae = []
    loss_rmse = []
    loss_mape = []
    r2 = []
    loss_mdape = []
    for i in range(self.num_stations):
      _p = y_predict[:, i]
      _y = y_original[:, i]
      _p -= self.sc[i].min_[0]
      _p /= self.sc[i].scale_[0]
      _y -= self.sc[i].min_[0]
      _y /= self.sc[i].scale_[0]
      loss_mae.append(mean_absolute_error(_y, _p))
      loss_rmse.append(mean_squared_error(_y, _p, squared=False))
      loss_mape.append(mean_absolute_percentage_error(_y, _p)*100)
      r2.append(r2_score(_y, _p))
      loss_mdape.append(median_absolute_percentage_error(_y, _p))

    return loss_mae, loss_rmse, loss_mape, r2,loss_mdape, y_predict, y_original

