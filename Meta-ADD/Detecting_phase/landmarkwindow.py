from tqdm import tqdm
import pandas as pd
from sklearn.naive_bayes import GaussianNB

import data_handler as dh
import stream_learning_lib as sl_lib





def exp4_SyntData_SyntDrift(learner, learner_param, target_ddi_t1):
    path = "Datasets/"
    # dataset_name_list = ['SEAa0', 'SEAg', 'HYPi', 'AGRa', 'AGRg', 'RBFi', 'RBFr', 'RTGn']
    # dataset_name_list = ['SEAa0']
    # dataset_name_list = ['AGRa', 'RTGn']
    # dataset_name_list = ['SEAa0', 'SEAg', 'HYPi', 'AGRg']
    # dataset_name_list = [ 'LEDa']
    dataset_name_list = ['SEAa0', 'HYPi', 'AGRa', 'RTGn', 'RBFi']
    # =====================================#
    # stream learning evaluation settings #
    # =====================================#
    train_size_min = 200
    # train_size_min = 1000
    df_result_list = []
    for dataset_name in tqdm(dataset_name_list):
        stream = dh.DriftDataset(path, dataset_name).np_data
        df_result = sl_lib.eval_stream_on_all_skmultiflow_ddm(dataset_name, stream[:, :-1], stream[:, -1],
                                                              train_size_min, learner, learner_param, target_ddi_t1)
        df_result_list.append(df_result)

    df_result_all = pd.concat(df_result_list, axis=0)
    df_result_all.reset_index(drop=True, inplace=True)

    return df_result_all

def exp4_RealData_UnknDrift(learner, learner_param, target_ddi_t1):

    path = "Datasets/"
    # dataset_name_list = ['elec', 'weat', 'spam', 'airl', 'covt-binary', 'poke-binary']
    # dataset_name_list = ['elec', 'weat', 'spam', 'airl', 'poke-binary']
    dataset_name_list = ['elec', 'weat', 'spam', 'airl']
    # dataset_name_list = ['elec', 'weat']
    # dataset_name_list = ['airl', 'poke-binary']
    # dataset_name_list = ['covt-binary']
    # =====================================#
    # stream learning evaluation settings #
    # =====================================#
    train_size_min = 200

    df_result_list = []
    for dataset_name in tqdm(dataset_name_list):
        stream = dh.DriftDataset(path, dataset_name).np_data
        df_result = sl_lib.eval_stream_on_all_skmultiflow_ddm(dataset_name, stream[:, :-1], stream[:, -1],
                                                              train_size_min, learner, learner_param, target_ddi_t1)
        df_result_list.append(df_result)

    df_result_all = pd.concat(df_result_list, axis=0)
    df_result_all.reset_index(drop=True, inplace=True)

    return df_result_all


if __name__ == "__main__":
    # the_k = 40
    # print("EI-kMeans", str(the_k))
    #
    # path = "Datasets/"
    # dataset_name_list = ['SEAa0', 'SEAg', 'HYPi', 'AGRa', 'AGRg', 'LEDa', 'LEDg', 'RBFi', 'RBFr', 'RTGn']

    #target_t1_list = [0.99, 0.95, 0.9, 0.85, 0.8]
    target_t1_list = [0.8]
    merged_result = []
    for t1 in target_t1_list:
        # df_result = exp4_SyntData_SyntDrift(GaussianNB, {}, target_ddi_t1=t1)
        # df_result = exp4_SyntData_SyntDrift(GaussianNB, {}, target_ddi_t1=t1)
        df_result = exp4_RealData_UnknDrift(GaussianNB, {}, target_ddi_t1=t1)
        df_result['TargetDDI'] = t1
        merged_result.append(df_result)
    merged_result_df = pd.concat(merged_result)
    #merged_result_df.to_csv('50-5-artificial-CNN.csv')
