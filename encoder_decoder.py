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
  def __init__(self,input_seq_len, input_size, hidden_size, num_layers=1, dropout=0):
    super(Encoder,self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
    # self.ln = nn.Linear(hidden_size, 1) # (batch_size, 1)
  
  def forward(self, x):
    # x = (batch_size, input_seq_len, input_size)
    # (h0, c0) : default filled with 0, h0, c0 : (num_layers, batch_size, hidden_size)
    out, state = self.lstm(x)     
    # out = self.ln(out) 
    # state: (num_layers, batch_size, hidden_size)
    # out: (batch_size, input_seq_len, hidden_size)
    
    return out, state   


class Decoder(nn.Module):
  def __init__(self, output_size, hidden_size, num_layers=1, dropout=0):
    super(Decoder, self).__init__()
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    
    self.lstm = nn.LSTM(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x, state):
    # hidden = h_out
    # cell = c_out
    # x: (batch_size, 1, output_size)   -> 1 input at a time, seq_len=1, initialize to 0
    # state: (num_layers, batch_size, hidden_size)
    output, hidden_state = self.lstm(x, (state))   # state: (num_layers, batch_size, hidden_size)
    # output: (batch_size, seq_length=1, hidden_size)
    out = self.fc(output)   
    # out: (batch_size, seq_length=1, output_size)
    return out, hidden_state

class EncoderDecoder(nn.Module):
  def __init__(self, input_seq_len=24, output_seq_len=5, input_size=14, output_size=1, hidden_size=12, num_layers=1, dropout=0):
    super(EncoderDecoder, self).__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.input_seq_len = input_seq_len
    self.output_seq_len = output_seq_len
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.encoder = Encoder(self.input_seq_len, self.input_size, self.hidden_size, self.num_layers, self.dropout)
    self.decoder = Decoder(self.output_size, self.hidden_size, self.num_layers, self.dropout)

  def forward(self, x, batch_size):
    outputs = torch.zeros(batch_size, self.output_seq_len, self.output_size)
      
    encoder_output, encoder_hidden = self.encoder(x)

    decoder_input = torch.zeros(batch_size, 1, self.output_size)
    decoder_hidden = encoder_hidden

    for t in range(self.output_seq_len):
      decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
      outputs[:,t,:] = decoder_output.squeeze(1)
      decoder_input = decoder_output
    return outputs

  def train(self, x_train, y_train, x_valid, y_valid, learning_rate=0.001, num_epochs=50, batch_size=32):
    # train_iterator: (batch_size, input_seq_len, input_size)
    optimizer = optim.Adam(self.parameters(),lr=learning_rate)
    criterion = nn.MSELoss()

    print(self.input_seq_len, self.output_seq_len)
    
    for epoch in tqdm(range(num_epochs)):
        # self.train()
        # train
        epoch_train_loss = 0
        n = int(len(x_train)/self.batch_size)
        for _ in range(n):
            index = random.randrange(0, n-self.batch_size)
            x = x_train[index : index + self.batch_size].to(self.device)
            y = y_train[index : index + self.batch_size].to(self.device)

            batch_loss = 0 
            optimizer.zero_grad()
        
            outputs = self.forward(x, batch_size)
            batch_loss = criterion(outputs,  y)
            batch_loss.backward()
            optimizer.step()
            epoch_train_loss  += batch_loss.item()
        print(f'\n\tLearning rate: {learning_rate}')
        print(f'Epoch: {epoch+1:02}')
        print(f'\t Train loss: {epoch_train_loss / len(n):.4f}')
        #validation
        # self.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            n_val = int((len(x_valid) - self.batch_size))
            for index in range(0, n_val, self.batch_size): 
                x = x_valid[index : index + self.batch_size].to(self.device)
                y = y_valid[index : index + self.batch_size].to(self.device) 
                batch_loss =  0
                outputs = self.forward(x, batch_size)
                batch_loss = criterion(outputs, y)
                epoch_val_loss += batch_loss.item()
            # val_loss = epoch_val_loss / len(valid_iterator)
        print(f'\t Val loss: {epoch_val_loss / len(n_val):.4f}')

  def test(self, iterator, num_epochs, batch_size, sc_test):
    # iterator: (batch_size, input_seq_len, input_size)
    y_original = []
    y_predict = []
    # self.load_state_dict(torch.load(path_model + 'checkpoint.pt')["model_dict"])
    with torch.no_grad():
      for x,y in iterator:          
        # output tensor
        outputs = self.forward(x, batch_size)

        y_predict.append(np.append(outputs.detach().numpy()[:, 0, :].reshape(-1), outputs.detach().numpy()[-1, 1:, :].reshape(-1)))
        y_original.append(np.append(y.detach().numpy()[:, 0, :].reshape(-1), y.detach().numpy()[-1, 1:, :].reshape(-1)))

    y_predict = np.reshape(y_predict, (-1))
    y_original = np.reshape(y_original, (-1))

    # rescale to true values
    y_pred_ = np.expand_dims(y_predict, 1)
    y_preds = np.repeat(y_pred_, self.input_size, 1)

    y_inv = sc_test.inverse_transform(y_preds)
    y_pred_true = y_inv[:, 0]

    y_orig_ = np.expand_dims(y_original, 1)
    y_origs = np.repeat(y_orig_, self.input_size, 1)

    y_inv_ori = sc_test.inverse_transform(y_origs)
    y_orig_true = y_inv_ori[:, 0]

    y_predict = y_pred_true
    y_original = y_orig_true

    loss_mae = mean_absolute_error(y_original, y_predict)
    loss_rmse = mean_squared_error(y_original, y_predict, squared=False)
    loss_mape = mean_absolute_percentage_error(y_original, y_predict)*100
    r2 = r2_score(y_original, y_predict)
    loss_mdape = median_absolute_percentage_error(y_original, y_predict)

    return loss_mae, loss_rmse, loss_mape, r2,loss_mdape, y_predict, y_original
