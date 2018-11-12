#!/usr/local/bin/python2
# -*- coding: utf-8 -*-
"""
@Time: 2018/11/11 15:20
@Author: zhaoxingfeng
@Function：调自己写的xgboost C++接口实现模型训练和预测等
@Version: V1.0
"""
from ctypes import *
import pandas as pd
import numpy as np
import ctypes


class Config(Structure):
    _fields_ = [("n_estimators", c_int),
                ("max_depth", c_int),
                ("learning_rate", c_float),
                ("min_data_in_leaf", c_int),
                ("min_child_weight", c_float),
                ("colsample_bytree", c_float),
                ("reg_gamma", c_float),
                ("reg_lambda", c_float)]

class XGBClassifier(object):
    def __init__(self, n_estimators=100, max_depth=10, learning_rate=0.1, min_samples_leaf=1,
                 colsample_bytree=1.0, min_child_weight=1.0, reg_gamma=0.0, reg_lambda=0.0):
        self.config = Config()
        self.config.max_depth = c_int(int(max_depth))
        self.config.n_estimators = c_int(int(n_estimators))
        self.config.learning_rate = c_float(learning_rate)
        self.config.min_data_in_leaf = c_int(int(min_samples_leaf))
        self.config.min_child_weight = c_float(min_child_weight)
        self.config.colsample_bytree = c_float(colsample_bytree)
        self.config.reg_gamma = c_float(reg_gamma)
        self.config.reg_lambda = c_float(reg_lambda)
        self.LIB = cdll.LoadLibrary(r"x64/Debug/xgboost-cpp.dll")
        self.xgboost_handle = ctypes.c_void_p()

    def fit(self, dataset, labels):
        dataset_feature = np.array(dataset, dtype=np.float32)
        dataset_labels = np.array(labels, dtype=np.float32)

        self.LIB.Train(byref(self.config), dataset_feature.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       dataset_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       c_int(dataset_feature.shape[0]), c_int(dataset_feature.shape[1]),
                       ctypes.byref(self.xgboost_handle))

    def predict_proba(self, dataset):
        dataset_feature = np.array(dataset, dtype=np.float32)

        preds = np.zeros(dataset_feature.shape[0], dtype=np.float32)
        self.LIB.Predict(dataset_feature.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         c_int(dataset_feature.shape[0]), c_int(dataset_feature.shape[1]),
                         ctypes.byref(self.xgboost_handle), preds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        return np.vstack((1.0 - preds, preds)).transpose()


if __name__ == '__main__':
    import time
    start = time.time()

    df = pd.read_csv(r"source/pima indians.csv")
    xgb = XGBClassifier(n_estimators=5,
                        max_depth=6,
                        learning_rate=0.4,
                        min_samples_leaf=10,
                        colsample_bytree=1.0,
                        min_child_weight=1,
                        reg_gamma=0.1,
                        reg_lambda=0.3)
    train_count = int(0.7 * len(df))
    xgb.fit(df.ix[:train_count, :-1], df.ix[:train_count, -1])

    from sklearn import metrics
    print(metrics.roc_auc_score(df.ix[:train_count, -1], xgb.predict_proba(df.ix[:train_count, :-1])[:, 1]))
    print(metrics.roc_auc_score(df.ix[train_count:, -1], xgb.predict_proba(df.ix[train_count:, :-1])[:, 1]))
    print(time.time() - start)
