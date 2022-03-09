# -*- coding: utf-8 -*-
import pandas as pd
import torch
import h5py
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA, FastICA


from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

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

    print(input_data.head())

    sparse_features = ['weekday', 'hour', 'region', 'city', 'adexchange', 'slotwidth', 'slotheight', \
        'slotvisibility', 'slotformat', 'creative', 'advertiser', 'useragent', 'slotprice', \
        'IP', 'domain', 'slotid']

    # Label Encoding for sparse features
    for feat in sparse_features:
        lbe = LabelEncoder()
        input_data[feat] = lbe.fit_transform(input_data[feat])


    # Code structure is same for ICA and PCA.
    pca = PCA(n_components=4)
    # pca = FastICA(n_components=4)
    principalComponents = pca.fit_transform(input_data)
    cols = ['pc1', 'pc2', 'pc3', 'pc4']
    principalDf = pd.DataFrame(data = principalComponents
                , columns = cols)

    final_data = pd.concat([principalDf, tgt_data], axis=1, join='inner')
    input_columns= cols
    sparse_features = cols
    dense_features = []
    target= ['click']


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

    model = PNN(dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    # history = model.fit(train_model_input, train[target].values, batch_size=2056, epochs=10, verbose=1,
    #                     validation_split=0.2, labels=np.unique(final_data[target]))
    history = model.fit(train_model_input, train[target].values, batch_size=2056, epochs=10, verbose=1,
        validation_split=0.2)
                        # validation_split=0.2)
    pred_ans = model.predict(test_model_input, 256)
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