import matplotlib.pyplot as plt
import os
import csv

def plot(station, y_predict, y_original):
    # print(y_predict.shape, y_original.shape)
    plt.figure(figsize=(13,5))
    plt.title(station)
    plt.plot(y_original, label='Original')
    plt.plot(y_predict, label='Predict')
    plt.legend()
    plt.show()

def save_results(result_list, log_dir):
	path = os.path.join(log_dir, "metrics.csv")
	with open(path, 'a') as file:
		writer = csv.writer(file)
		writer.writerow(result_list)

def visualize(y_true, y_pred, log_dir, name):
	plt.plot(y_pred, label='preds')
	plt.plot(y_true, label='gt')
	plt.legend()
	plt.savefig(os.path.join(log_dir, name))
	plt.close()