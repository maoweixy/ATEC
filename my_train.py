#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pickle
import jieba
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')

def pkl_save(filename, file):
    output = open(filename, 'wb')
    pickle.dump(file, output)
    output.close()

def get_df():
    df = pd.read_json(
        "/mnt/atec/train.jsonl", encoding="utf-8", lines=True).set_index("id")
    df = df.iloc[
        ((df.label>=0) & (df.x0!=-1111)).values,
        :#(df.nunique()!=1).values
    ].reset_index(drop=True)
    return df

def lgb_metric(y_true, y_pred):
    score = 0
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    precision, recall = precision[::-1], recall[::-1]
    for threshold, weight in zip([0.8, 0.85, 0.9], [0.3, 0.3, 0.4]):
        last_idx = np.arange(precision.shape[0])[precision >= threshold][-1]
        score += weight * recall[last_idx]
    return "my_score", score, True

def lgb_run(df):
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    score = 0
    for model_id, (train_index, test_index) in enumerate(
        skf.split(df.iloc[:, :-2].values, df.label)
    ):
        print("[LOG] 第%d个Fold开始训练 ..."%(model_id+1))
        X_train, X_test = df.iloc[train_index, :-1].values, df.iloc[test_index, :-1].values
        y_train, y_test = df.label[train_index], df.label[test_index]
        lgb_model = LGBMClassifier(
            max_depth=8, objective="binary", learning_rate=0.005, 
            min_child_weight=0.005, reg_alpha=0.001, reg_lambda=2,
            num_leaves=64, n_estimators=10000,# device="gpu",
            random_state=0
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=lgb_metric,
            early_stopping_rounds=100,
            verbose=False
        )
        score += lgb_model.best_score_["valid_0"]["my_score"]/4
        pkl_save("lgb_model_%d.dat"%model_id, lgb_model)
        print("[LOG] 第%d个Fold的最佳迭代轮数为%d，得分为"%(
            model_id+1, lgb_model.best_iteration_), round(lgb_model.best_score_["valid_0"]["my_score"], 6))
    print("[LOG] 交叉验证得分为", round(score, 6))
    # 0.64276（线下）
    # 0.6362282878411911（线上）

def tfidf_process(df):
    text = [' '.join(jieba.cut(str(x))) for x in df['memo_polish']]
    tfidf = TfidfVectorizer(max_features=500)
    model = tfidf.fit(text)
    pickle.dump(model, open("tfidf.pickle", "wb"))
    data = model.transform(text)
    df = pd.concat([
        df.iloc[:, :-2],
        pd.DataFrame(data.toarray().astype(np.float32)),
        df.iloc[:, [-1]]
    ], axis=1)
    return df
    
def train():
    df = get_df()
    df = tfidf_process(df)
    return
    lgb_run(df)

if __name__ == '__main__':
    if train():
        sys.exit(0)
    else:
        sys.exit(1)