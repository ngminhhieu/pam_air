from load_data import *
from model.model import Encoder
from model.model_unity import Encoder as Encoder_unity
from model.encoder_decoder import EncoderDecoder
from utils import *
import os
import sys
from tqdm import tqdm
import torch 

#hyper parameters
horizon = 1
input_seq_len = 48
input_size = 3
hidden_size = 64
epochs = 100
batch_size = 32
learning_rate = 0.0001
num_layers=1
dropout=0
cuda=True
train = 0
test = 1
input_features = None
fill = False
start_date = '21/05/2021 10:00:00'
end_date = '15/12/2021 23:00:00'
len_data = 4400
# start_date = None
# end_date = None
# len_data = None

log = './log/lstm_unity'
if not os.path.exists(log):
    os.makedirs(log)

def run_model(model,x_train_full, y_train_full, x_valid_full, y_valid_full,x_test_full, y_test_full, list_station, cuda = cuda, train = train, test = test, station_name = None):
    if cuda:
        model = model.cuda()
    if train:
        model.train(x_train_full, y_train_full, x_valid_full, y_valid_full, station_name = station_name)
    if test:
        loss_mae, loss_rmse, loss_mape, r2,loss_mdape, y_predict, y_original = model.test(x_test_full, y_test_full, station_name = station_name)
        if station_name is None:
            for i, station in enumerate(list_station):
                save_results([station, loss_mae[i], loss_rmse[i], loss_mape[i], r2[i], loss_mdape[i]], log)
                visualize(y_original[:, i], y_predict[:, i], log, "result_{}.png".format(station.split('.')[0]))
        else:
            save_results([station_name, loss_mae, loss_rmse, loss_mape, r2, loss_mdape], log)
            visualize(y_original, y_predict, log, "result_{}.png".format(station_name))


if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <model>")
        exit()
    model = sys.argv[1]
    train = int(sys.argv[2])
    test = int(sys.argv[3])
    cuda = sys.argv[4]
    if len(sys.argv) > 5:
        input_features = sys.argv[5].split(',')
        fill = sys.argv[6]
    if cuda == 'True':
        cuda = True
    else:
        cuda = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_link = './imputation_data/S0000099-Phu-Vien.csv'
    # data_path = './pam_air_data/pam_air/'
    data_path = './pam_air_data_loc/'
    # data_path = './data/ref/'
    list_station = os.listdir(data_path)
    save_results(["Target Station", "MAE", "RMSE", "MAPE", "R2_score", "MDAPE"], log)

    if model == 'unity':
        log = './log/lstm_unity'
        num_stations = 0
        x_train_full = []
        x_valid_full = []
        x_test_full = []
        y_train_full = []
        y_valid_full = []
        y_test_full = []
        sc_full = []
        for station in tqdm(list_station, desc="Getting data"):
            num_stations += 1
            data_link = data_path + station
            x_train, x_valid, x_test, y_train, y_valid, y_test, sc = make_data_set(data_link, seq_len = input_seq_len, output_len = 1, start_date = start_date, end_date = end_date, len_data = len_data, input_feature = input_features, fill = fill)
            x_train_full.append(x_train)
            x_valid_full.append(x_valid)
            x_test_full.append(x_test)
            y_train_full.append(y_train)
            y_valid_full.append(y_valid)
            y_test_full.append(y_test)
            sc_full.append(sc)
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
        model = Encoder_unity(log, cuda, num_stations, input_size, hidden_size, sc_full, epochs = epochs, batch_size = batch_size, learning_rate = learning_rate, num_layers=num_layers, dropout=dropout)
        run_model(model, x_train_full, y_train_full, x_valid_full, y_valid_full, x_test_full, y_test_full, list_station, cuda = cuda, train = train, test = test)
 
    elif model == 'encoder':
        log = './log/lstm'
        for station in list_station:
            station_name = station.split('.')[0]
            data_link = data_path + station
            # data_link = './raw_hanoi_data.csv'
            # station = 'raw hanoi'
            # station_name = station
            x_train, x_valid, x_test, y_train, y_valid, y_test, sc = make_data_set(data_link, seq_len = input_seq_len, output_len = 1, start_date = start_date, end_date = end_date, len_data = len_data, input_feature = input_features, fill = fill)
            model = Encoder(log, cuda, input_size, hidden_size, sc, epochs = epochs, batch_size = batch_size, learning_rate = learning_rate, num_layers=num_layers, dropout=dropout)
            run_model(model, x_train, y_train, x_valid, y_valid, x_test, y_test, [station], cuda = cuda, train = train, test = test, station_name = station_name)
    elif model == 'enc_dec':
            # data_link = './raw_hanoi_data.csv'
            # station = 'raw hanoi'
            # station_name = station
        log = './log/enc_dec'
        if not os.path.exists(log):
            os.makedirs(log)
        for station in list_station:
            station_name = station.split('.')[0]
            data_link = data_path + station
            x_train, x_valid, x_test, y_train, y_valid, y_test, sc = make_data_set(data_link, seq_len = input_seq_len, output_len = 1, start_date = start_date, end_date = end_date, len_data = len_data, input_feature = input_features, fill = fill)
            model = EncoderDecoder(log, cuda, sc, input_seq_len=input_seq_len,batch_size = batch_size, epochs = epochs,learning_rate = learning_rate, output_seq_len=1, input_size=input_size, output_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
            run_model(model, x_train, y_train, x_valid, y_valid, x_test, y_test, [station], cuda = cuda, train = train, test = test, station_name = station_name)
