from load_data import *
from model import Encoder
from utils import *
import os
import torch 

#hyper parameters
input_seq_len = 48
horizon = 1
input_size = 3
hidden_size = 64
epochs = 200
batch_size = 32
learning_rate = 0.0001
num_layers=1
dropout=0
cuda=False

log = './log/imputation'
if not os.path.exists(log):
    os.makedirs(log)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = './imputation_data/'
    list_station = os.listdir(data_path)
    save_results(["Target Station", "MAE", "RMSE", "MAPE", "R2_score", "MDAPE"], log)
    num_stations = 0
    x_train_full = []
    x_valid_full = []
    x_test_full = []
    y_train_full = []
    y_valid_full = []
    y_test_full = []
    for station in list_station:
        num_stations += 1
        data_link = data_path + station
        x_train, x_valid, x_test, y_train, y_valid, y_test, sc = make_data_set(data_link, seq_len = input_seq_len, output_len = 1, min = None, max = None)
        x_train_full.append(x_train)
        x_valid_full.append(x_valid)
        x_test_full.append(x_test)
        y_train_full.append(y_train)
        y_valid_full.append(y_valid)
        y_test_full.append(y_test)
    x_train_full = torch.stack(x_train_full)
    x_train_full = torch.reshape(x_train_full, (-1, input_seq_len, num_stations*input_size))
    y_train_full = torch.stack(y_train_full)
    y_train_full = torch.reshape(y_train_full, (-1, num_stations, horizon))
    x_valid_full = torch.stack(x_valid_full)
    x_valid_full = torch.reshape(x_valid_full, (-1, input_seq_len, num_stations*input_size))
    y_valid_full = torch.stack(y_valid_full)
    y_valid_full = torch.reshape(y_valid_full, (-1, num_stations, horizon))
    x_test_full = torch.stack(x_test_full)
    x_test_full = torch.reshape(x_test_full, (-1, input_seq_len, num_stations*input_size))
    y_test_full = torch.stack(y_test_full)
    y_test_full = torch.reshape(y_test_full, (-1, num_stations, horizon))

    model = Encoder(cuda, num_stations, input_size, hidden_size, sc, epochs = epochs, batch_size = batch_size, learning_rate = learning_rate, num_layers=num_layers, dropout=dropout)
    if cuda:
        model = model.cuda()
    model.train(x_train, y_train, x_valid, y_valid)
    loss_mae, loss_rmse, loss_mape, r2,loss_mdape, y_predict, y_original = model.test(x_test, y_test)
    save_results([station, loss_mae, loss_rmse, loss_mape, r2, loss_mdape], log)
    visualize(y_original, y_predict, log, "result_{}.png".format(station.split('.')[0]))