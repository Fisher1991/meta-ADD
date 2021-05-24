import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# from matplotlib import pyplot as plt
# import cv2
import os
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
tqdm.pandas(desc="my bar!")


def preprocessingAbruptDriftData(feature_length, DATA_FILE):

    file_dir='input/Data/'+ DATA_FILE +'/abrupt'
    all_file_list=os.listdir(file_dir)

    for single_file in all_file_list:
        single_data_frame = pd.read_csv(os.path.join(file_dir, single_file), sep=',', header=0)
        # print(single_data_frame.iloc[:,[1]])
        single_data_frame['gap'] = ((single_data_frame.iloc[:,[1]].shift(-1) - single_data_frame.iloc[:,[1]])
                                    /single_data_frame.iloc[:,[1]])
        train_data = single_data_frame[['gap']][:feature_length].T
        # print(train_data)

        if single_file == all_file_list[0]:
            all_abrupt_drift_data_frame = train_data
        else:  #进行concat操作
            all_abrupt_drift_data_frame = pd.concat([all_abrupt_drift_data_frame, train_data], ignore_index=True)

    all_abrupt_drift_data_frame['label'] = 0

    # print(all_abrupt_drift_data_frame)
    return all_abrupt_drift_data_frame


def preprocessingGradualDriftData(feature_length, DATA_FILE):

    file_dir = 'input/Data/'+ DATA_FILE +'/gradual'
    all_file_list = os.listdir(file_dir)

    for single_file in all_file_list:
        single_data_frame = pd.read_csv(os.path.join(file_dir, single_file), sep=',', header=0)
        # print(single_data_frame.iloc[:,[1]])
        single_data_frame['gap'] = ((single_data_frame.iloc[:, [1]].shift(-1) - single_data_frame.iloc[:, [1]])
                                    / single_data_frame.iloc[:, [1]])
        train_data = single_data_frame[['gap']][:feature_length].T
        # print(train_data)

        if single_file == all_file_list[0]:
            all_gradual_drift_data_frame = train_data
        else:  # concat
            all_gradual_drift_data_frame = pd.concat([all_gradual_drift_data_frame, train_data], ignore_index=True)

    all_gradual_drift_data_frame['label'] = 1

    # print(all_abrupt_drift_data_frame)
    return all_gradual_drift_data_frame


def preprocessingIncrementalDriftData(feature_length, DATA_FILE):

    file_dir = 'input/Data/'+ DATA_FILE +'/incremental'
    all_file_list = os.listdir(file_dir)

    for single_file in all_file_list:
        single_data_frame = pd.read_csv(os.path.join(file_dir, single_file), sep=',', header=0)
        # print(single_data_frame.iloc[:,[1]])
        single_data_frame['gap'] = ((single_data_frame.iloc[:, [1]].shift(-1) - single_data_frame.iloc[:, [1]])
                                    / single_data_frame.iloc[:, [1]])
        train_data = single_data_frame[['gap']][:feature_length].T
        # print(train_data)

        if single_file == all_file_list[0]:
            all_incremental_drift_data_frame = train_data
        else:  # 进行concat操作
            all_incremental_drift_data_frame = pd.concat([all_incremental_drift_data_frame, train_data], ignore_index=True)

    all_incremental_drift_data_frame['label'] = 2

    # print(all_abrupt_drift_data_frame)
    return all_incremental_drift_data_frame


def preprocessingNoDriftData(feature_length, DATA_FILE):
    file_dir = 'input/Data/'+ DATA_FILE +'/normal'
    all_file_list = os.listdir(file_dir)

    for single_file in all_file_list:
        single_data_frame = pd.read_csv(os.path.join(file_dir, single_file), sep=',', header=0)
        # print(single_data_frame.iloc[:,[1]])
        single_data_frame['gap'] = ((single_data_frame.iloc[:, [1]].shift(-1) - single_data_frame.iloc[:, [1]])
                                    / single_data_frame.iloc[:, [1]])
        train_data = single_data_frame[['gap']][:feature_length].T
        # print(train_data)

        if single_file == all_file_list[0]:
            all_no_drift_data_frame = train_data
        else:  # concat
            all_no_drift_data_frame = pd.concat([all_no_drift_data_frame, train_data],
                                                         ignore_index=True)

    all_no_drift_data_frame['label'] = 3

    # print(all_abrupt_drift_data_frame)
    return all_no_drift_data_frame


def LoadDriftData(Data_Vector_Length, DATA_FILE):
    feature_length = Data_Vector_Length
    sample_num = 1250
    all_abrupt_drift_data_frame = preprocessingAbruptDriftData(feature_length, DATA_FILE)
    all_data_frame = all_abrupt_drift_data_frame.iloc[0:sample_num]
    all_gradual_drift_data_frame = preprocessingGradualDriftData(feature_length, DATA_FILE)
    all_data_frame = pd.concat([all_data_frame, all_gradual_drift_data_frame], ignore_index=True).iloc[0:2*sample_num]
    all_incremental_drift_data_frame = preprocessingIncrementalDriftData(feature_length, DATA_FILE)
    all_data_frame = pd.concat([all_data_frame, all_incremental_drift_data_frame], ignore_index=True).iloc[0:3*sample_num]
    all_no_drift_data_frame = preprocessingNoDriftData(feature_length, DATA_FILE)
    all_data_frame = pd.concat([all_data_frame, all_no_drift_data_frame], ignore_index=True).iloc[0:4*sample_num]

    columnList = []
    for i in range(feature_length):
        feature_name = "feature_" + str(i)
        columnList.append(feature_name)
    columnList.append('label')
    all_data_frame.columns = columnList
    return all_data_frame