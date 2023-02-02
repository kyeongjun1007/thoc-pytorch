import pandas as pd
import numpy as np
import torch
import logging
from pickle import load
from torch.utils.data import DataLoader

from THOCBase import THOC
from train import get_test_params
from common.utils import get_result_folder
from DataConfigurator import SlidingWindowTest as SlidingWindow

# model config
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
test_param_dict = get_test_params()

model_params = test_param_dict['model_params']
init_data = test_param_dict['init_data']
logger_params = test_param_dict['logger_params']

result_folder = get_result_folder(logger_params['log_file']['desc'],
                                  date_prefix=logger_params['log_file']['date_prefix'],
                                  result_dir=logger_params['log_file']['result_dir']
                                   )

model = THOC(**model_params, init_data=init_data).to(device) # init model randomly to load params

param_dict = torch.load(result_folder + "/param_dict.pt", map_location=device)
model.load_state_dict(param_dict)

# data config
header = test_param_dict['header']
index_col = test_param_dict['index_col']

data_name = test_param_dict['data_name']
test_df = pd.read_csv(result_folder + data_name, header=header, index_col=index_col)

scaler = load(open(result_folder + '/scaler.pkl', 'rb'))

test_X = scaler.transform(test_df.iloc[:, :-1])
test_y = test_df.iloc[:, -1]

test_X = torch.tensor((np.array(test_X, dtype="float32")))
test_y = torch.tensor((np.array(test_y, dtype="float32")))

# logging
logging.basicConfig(level=logging.INFO, filename=result_folder + "/test_log.log", filemode="w", format="%(message)s")

# DataLoader
batch_size = test_param_dict['batch_size']
window_size = test_param_dict['window_size']
shuffle = test_param_dict['shuffle']

test_dataset = SlidingWindow(X=test_X, y=test_y, window=window_size)
test_dl = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

# run test
threshold = test_param_dict['threshold']

total = 0
normal, abnormal = 0, 0
tp, tn, fp, fn = 0, 0, 0, 0

model.eval()
with torch.no_grad() :
    for i, (window, y) in enumerate(test_dl):
        window = window.to(device)
        y = y>0

        anomaly_scores, _, _ = model.forward(window)
        anomaly_score = anomaly_scores.sum(dim=0)

        pred = (anomaly_score < threshold).max()
        real = y.max()

        total += 1
        if real is False :
            normal += 1
        else :
            abnormal += 1

        if pred is True :
            if real is True :
                tp += 1
            else :
                fp += 1
        else :
            if real is True :
                fn += 1
            else :
                tn += 1
        print(f"TP : {tp},TN : {tn},FP : {fp},FN : {fn}.")

precision = tp / (tp+fp+1e-10)
recall = tp / (tp+fn+1e-10)
f1_score = 2*precision*recall / (precision+recall+1e-10)

logging.info(f"Test result of {data_name} dataset.\n")
logging.info(f"total : {total}, normal : {normal}, abnormal : {abnormal}.")
logging.info(f"TP : {tp},TN : {tn},FP : {fp},FN : {fn}.")
logging.info(f"precision : {precision}.")
logging.info(f"recall : {recall}.")
logging.info(f"f1-score : {f1_score}.")

print(f"Test result of {data_name} dataset.\n")
print(f"total : {total}, normal : {normal}, abnormal : {abnormal}.")
print(f"TP : {tp},TN : {tn},FP : {fp},FN : {fn}.")
print(f"precision : {precision}.")
print(f"recall : {recall}.")
print(f"f1-score : {f1_score}.")