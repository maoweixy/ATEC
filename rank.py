#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import pickle
import jieba
import pandas as pd
from lightgbm import LGBMClassifier
from target_encoding import TargetEncoder
from sklearn.model_selection import StratifiedKFold
import numpy as np


input_data_path = '/home/admin/workspace/job/input/test.jsonl'
output_predictions_path = '/home/admin/workspace/job/output/predictions.jsonl'

def tfidf_process(df):
    text = [' '.join(jieba.cut(str(x))) for x in df['memo_polish']]
    model = pickle.load(open("tfidf.pickle", "rb"))
    data = model.transform(text)
    df = pd.concat([
        df.iloc[:, :-1], # !!!!!!!
        pd.DataFrame(data.toarray().astype(np.float32))
    ], axis=1)
    return df

def target_encoding(X_train, arr, y_train, category_cols, model_id):
    enc = TargetEncoder()
    new_train = enc.transform_train(X_train[:, category_cols], y_train)
    new_test = enc.transform_test(arr[:, category_cols])
    new_test = np.c_[arr, new_test]
    return new_test

def rank():
    df = pd.read_json(input_data_path, encoding="utf-8", lines=True).set_index("id")
    df_train = pd.read_json("train.jsonl", encoding="utf-8", lines=True).set_index("id")
    df_train = df_train.iloc[
        ((df_train.label!=-1)&(df_train.x0!=-1111)).values, :].reset_index(drop=True)
    drop_list = [
        'x2', 'x55', 'x91', 'x96', 'x107', 'x184', 'x198',
        'x207', 'x209', 'x261', 'x319', 'x384', 'x436', 'x452', 'x456']
    df_train = df_train.drop(drop_list, axis=1)
    df = df.drop(drop_list, axis=1)
    df = tfidf_process(df)
    y_pred = 0
    category_cols = np.load("cate_cols.npy")
    seed_dict = {0:0,1:42,2:999}
    for seed_id in [0,1,2]:
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed_dict[seed_id])
        for model_id, (train_index, test_index) in enumerate(
            skf.split(df_train.iloc[:, :-2].values, df_train.label)
        ):
            print(model_id)
            arr = target_encoding(
                df_train.iloc[train_index, :-2].values, df.values, df_train.label.iloc[train_index].values, category_cols, model_id)
            loaded_model = pickle.load(open("lgb_model_fold_%d_seed_%d.dat"%(model_id, seed_dict[seed_id]), "rb"))
            y_pred += loaded_model.predict_proba(arr)[:, 1] / 12
    prediction = pd.DataFrame({"id": ["%d"%(i) for i in range(df.shape[0])], "label": y_pred})
    prediction.to_json(output_predictions_path, lines=True, orient='records')
    return True


if __name__ == '__main__':
    if rank():
        sys.exit(0)
    else:
        sys.exit(1)
