from load_data import *
from model import Encoder
from utils import *
import os
import torch 

#hyper parameters
input_seq_len = 24
input_size = 3
hidden_size = 128
epochs = 1
batch_size = 32
learning_rate = 0.0001
num_layers=1
dropout=0
cuda=True
train = 1
test = 1

log = './log/lstm_imputation'
if not os.path.exists(log):
    os.makedirs(log)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_link = './imputation_data/S0000099-Phu-Vien.csv'
    # data_path = './pam_air_data/pam_air/'
    data_path = './imputation_data/'
    list_station = os.listdir(data_path)
    save_results(["Target Station", "MAE", "RMSE", "MAPE", "R2_score", "MDAPE"], log)
    for station in list_station:
        # if station != "S0000099-Phu Vien.csv":
        #     continue
        data_link = data_path + station
        x_train, x_valid, x_test, y_train, y_valid, y_test, sc = make_data_set(data_link, seq_len = 24, output_len = 1, min = None, max = None)
        model = Encoder(log, cuda, input_size, hidden_size, sc, epochs = epochs, batch_size = batch_size, learning_rate = learning_rate, num_layers=num_layers, dropout=dropout)
        if cuda:
            model = model.cuda()
        if train:
            model.train(station, x_train, y_train, x_valid, y_valid)
        if test:
            loss_mae, loss_rmse, loss_mape, r2,loss_mdape, y_predict, y_original = model.test(station, x_test, y_test)
        save_results([station, loss_mae, loss_rmse, loss_mape, r2, loss_mdape], log)
        visualize(y_original, y_predict, log, "result_{}.png".format(station.split('.')[0]))