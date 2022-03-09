# -*- coding: utf-8 -*-
import pandas as pd
import torch
import h5py
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
import sys
import glob

#sys.path.append('C:/Users/chand/Downloads/UserResponsePrediction/pk/deepctr_torch')

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *


def append_files():
    input_files = glob.glob('../../iPinYou-all/hdf/train_input*')
    output_files = glob.glob('../../iPinYou-all/hdf/train_output*.h5')
    lst = []
    #print(input_files)
    input_files.sort()
    output_files.sort()
    s = 0
    for i,j in zip(input_files,output_files):
        with pd.HDFStore(str(i), mode='r') as hdf_y:
            input_data = pd.read_hdf(hdf_y)
            input_len = len(input_data)

        with pd.HDFStore(str(j), mode='r') as hdf_y:
            output_data = pd.read_hdf(hdf_y)
            output_len = len(input_data)

            pos_neg = np.sum(output_data['click'] == 1)
            num_neg = np.sum(output_data['click'] == 0)
            #print ("Pos: %d, Neg: %d" % (pos_neg, num_neg))

        i = str(i).split("hdf")[1][1:]
        print(f"The file {str(i)} has size of {len(input_data)} and pos {pos_neg} and neg {num_neg}")
        s+=len(input_data)

        final_data = pd.concat([input_data, output_data], axis=1, join='inner')
        lst.append(final_data)
    df = pd.concat(lst)
    print(f"List length is {len(lst)}The total dataframe size is {len(df)} and count is {s}")
    return df



def random_under_sampling(df,sampling_ratio):
    value_0 = df[df['click'] == 0]
    value_1 = df[df['click'] == 1]

    value_0_sampled = value_0.sample(frac=sampling_ratio)
    newdf= pd.concat([value_1,value_0_sampled])
    print(f"Old df with length of {len(df)} New df has length of {len(newdf)}")
    print(newdf['click'].value_counts())
    return newdf


def random_under_over_sampler(df, sampling_ratio):
    X = df.loc[:, df.columns != 'click']
    y = df.loc[:, df.columns == 'click']
    rus = RandomUnderSampler(sampling_ratio)
    X_rus, y_rus= rus.fit_resample(X, y)
    finaldf = pd.concat([X_rus,y_rus],axis = 1)
    print(f"Old df with length of {len(df)} New df has length of {len(finaldf)}")
    print(finaldf['click'].value_counts())
    return finaldf


def tomeks_link(df):
    X = df.loc[:, df.columns != 'click']
    y = df.loc[:, df.columns == 'click']
    tomek = TomekLinks(sampling_strategy='auto')
    X_rus, y_rus, id_rus = tomek.fit_resample(X, y)
    finaldf = pd.concat([X_rus,y_rus],axis = 1)
    print(f"Old df with length of {len(df)} New df has length of {len(newdf)}")

    return finaldf    





if __name__ == "__main__":


    filename = "../data/ipinyou/iPinYou-all/hdf/train_input_part_0.h5"
    with pd.HDFStore(filename, mode='r') as hdf_y:
        input_data = pd.read_hdf(hdf_y)

    filename = "../data/ipinyou/iPinYou-all/hdf/train_output_part_0.h5"
    with pd.HDFStore(filename, mode='r') as hdf_y:
        tgt_data = pd.read_hdf(hdf_y)
        pos_neg = np.sum(tgt_data['click'] == 1)
        num_neg = np.sum(tgt_data['click'] == 0)
        print ("Pos: %d, Neg: %d" % (pos_neg, num_neg))

    final_data = pd.concat([input_data, tgt_data], axis=1, join='inner')

    '''
    Commented lines belpw appends all the h5 files
    '''
    #final_data = append_files()
    print(final_data['click'].value_counts())



    ##Sampling Strategy. Three function implemented above. Call one of those 3 for results

    sampling_ratio = 0.05
    num_epochs = 10
    # final_data = random_under_sampling(final_data, sampling_ratio=sampling_ratio)
    final_data = random_under_over_sampler(final_data, sampling_ratio=sampling_ratio)




    input_columns=['weekday', 'hour', 'IP', 'region', 'city', 'adexchange', 'domain',
       'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat',
       'creative', 'advertiser', 'useragent', 'slotprice']
    sparse_features = ['weekday', 'hour', 'region', 'city', 'adexchange', 'slotwidth', 'slotheight', \
            'slotvisibility', 'slotformat', 'creative', 'advertiser', 'useragent', 'slotprice', \
            'IP', 'domain', 'slotid']
    dense_features = []
    target= ['click']

    # Label Encoding for sparse features
    for feat in sparse_features:
        lbe = LabelEncoder()
        final_data[feat] = lbe.fit_transform(final_data[feat])

    # Count unique feautres for each sparse feature
    fixlen_feature_columns = [SparseFeat(feat, final_data[feat].nunique())
                              for feat in sparse_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # Split into train-test
    train, test = train_test_split(final_data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # Define model and run
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    # model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                task='binary',
    #                l2_reg_embedding=1e-5, device=device)
    model = PNN(dnn_feature_columns=dnn_feature_columns, dnn_hidden_units=(32,32),
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    history = model.fit(train_model_input, train[target].values, batch_size=2056, epochs=num_epochs, verbose=1,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, 256)
    # print (test[target].values[:10], pred_ans[:10])
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

    ####
    # data = pd.read_csv('./criteo_sample.txt')

    # sparse_features = ['C' + str(i) for i in range(1, 27)]
    # dense_features = ['I' + str(i) for i in range(1, 14)]

    # print (np.unique(data[sparse_features[1]]))

    # data[sparse_features] = data[sparse_features].fillna('-1', )
    # data[dense_features] = data[dense_features].fillna(0, )
    # target = ['label']

    # # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(data[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])

    # # 2.count #unique features for each sparse field,and record dense feature field name

    # fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
    #                           for feat in sparse_features] + [DenseFeat(feat, 1, )
    #                                                           for feat in dense_features]

    # dnn_feature_columns = fixlen_feature_columns
    # linear_feature_columns = fixlen_feature_columns

    # feature_names = get_feature_names(
    #     linear_feature_columns + dnn_feature_columns)

    # # 3.generate input data for model

    # train, test = train_test_split(data, test_size=0.2, random_state=2020)
    # train_model_input = {name: train[name] for name in feature_names}
    # test_model_input = {name: test[name] for name in feature_names}

    # # 4.Define Model,train,predict and evaluate

    # device = 'cpu'
    # use_cuda = True
    # if use_cuda and torch.cuda.is_available():
    #     print('cuda ready...')
    #     device = 'cuda:0'

    # model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                task='binary',
    #                l2_reg_embedding=1e-5, device=device)

    # model.compile("adagrad", "binary_crossentropy",
    #               metrics=["binary_crossentropy", "auc"], )

    # history = model.fit(train_model_input, train[target].values, batch_size=32, epochs=10, verbose=2,
    #                     validation_split=0.2)
    # pred_ans = model.predict(test_model_input, 256)
    # print("")
    # print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    # print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))