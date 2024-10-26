import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.data.loaddata import load_data
import os
import matplotlib.pyplot as plt
import pickle
# 多个受试者的数据
USE_NANSEIZURE_DATA = True
SAVE_PROCESSED_DATA = True
USE_SAVED_DATA = True
UNDER_SAMPLING_RATIO = 0.4

# <读取含发作事件的数据> ----------------------------------------------------------
seizure_base_path = "data"
seizure_file_ids = [3, 5, 13, 21]

all_X_ws = []
all_y_ws = []

for file_id in seizure_file_ids:
    if USE_SAVED_DATA:
        data_file = f"{seizure_base_path}/data_subject_{file_id}.pkl"
        if os.path.exists(data_file):
            print(f"Loading data from {data_file} for subject {file_id}...")
            with open(data_file, 'rb') as f:
                X, y = pickle.load(f)
        else:
            print(f"Data file not found, loading and saving data for subject {file_id}...")
            X, y = load_data(file_id, seizure_base_path)
            if SAVE_PROCESSED_DATA:
                with open(data_file, 'wb') as f:
                    pickle.dump((X, y), f)
                print(f"Data saved to {data_file}.")
    else:
        X, y = load_data(file_id, seizure_base_path)
    all_X_ws.append(X)
    all_y_ws.append(y)
# <读取不含发作事件的数据> --------------------------------------------------------
nonseizure_base_path = "data_wos"
nonseizure_file_ids = [1, 2, 3, 5, 11]

all_X_wos = []
all_y_wos = []

for file_id in nonseizure_file_ids:
    if USE_NANSEIZURE_DATA:
        if USE_SAVED_DATA:
            data_file = f"{nonseizure_base_path}/data_subject_{file_id}.pkl"
            if os.path.exists(data_file):
                print(f"Loading data from {data_file} for subject {file_id}...")
                with open(data_file, 'rb') as f:
                    X, y = pickle.load(f)
            else:
                print(f"Data file not found, loading and saving data for subject {file_id}...")
                X, y = load_data(file_id, nonseizure_base_path)

                if SAVE_PROCESSED_DATA:
                    with open(data_file, 'wb') as f:
                        pickle.dump((X, y), f)
                    print(f"Data saved to {data_file}.")
        else:
            X, y = load_data(file_id, nonseizure_base_path)
        all_X_wos.append(X)
        all_y_wos.append(y)
# <合并数据> -----------------------------------------------
if USE_NANSEIZURE_DATA:
    all1 = all_X_wos
    all_X = np.vstack(all1)
    all_y = np.concatenate(all_y_ws + all_y_wos)
else:
    all_X = np.vstack(all_X_ws)
    all_y = np.concatenate(all_y_ws)


print(f"总数据量：{all_X.shape[0]}")
print(f"发作数据量：{np.sum(all_y == 1)}")
print(f"发作数据比例：{np.sum(all_y == 1) / all_X.shape[0]}")

# <划分欠采样部分与过采样部分> -------------------------------
over_X, under_X, over_y, under_y = train_test_split(
    all_X, all_y, test_size=1-OVER_SAMPLING_RATE, random_state=42, stratify=all_y)