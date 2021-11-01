#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pickle
import jieba
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from target_encoding import TargetEncoder
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import FastICA

import warnings
warnings.filterwarnings('ignore')

def pkl_save(filename, file):
    output = open(filename, 'wb')
    pickle.dump(file, output)
    output.close()

def get_df():
    df = pd.read_json("/mnt/atec/train.jsonl", encoding="utf-8", lines=True).set_index("id")
    df = df.iloc[
        ((df.label!=-1)&(df.x0!=-1111)).values, :].reset_index(drop=True)
    drop_list = [
        'x2', 'x55', 'x91', 'x96', 'x107', 'x184', 'x198',
        'x207', 'x209', 'x261', 'x319', 'x384', 'x436', 'x452', 'x456']
    df = df.drop(drop_list, axis=1)
    return df

def lgb_metric(y_true, y_pred):
    score = 0
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    precision, recall = precision[::-1], recall[::-1]
    for threshold, weight in zip([0.8, 0.85, 0.9], [0.3, 0.3, 0.4]):
        last_idx = np.arange(precision.shape[0])[precision >= threshold][-1]
        score += weight * recall[last_idx]
    return "my_score", score, True

def target_encoding(X_train, X_test, y_train, category_cols, model_id):
    enc = TargetEncoder()
    new_train = enc.transform_train(X_train[:, category_cols], y_train)
    new_test = enc.transform_test(X_test[:, category_cols])
    return np.c_[X_train, new_train], np.c_[X_test, new_test]

def lgb_run(df, category_cols):
    for seed in [0, 42, 999]:
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        score, importance = 0, 0
        for model_id, (train_index, test_index) in enumerate(
            skf.split(df.iloc[:, :-2].values, df.label)
        ):
            print("[LOG] 第%d个Fold开始训练 ..."%(model_id+1))
            X_train, X_test = (
                df.iloc[train_index, :-1].values, df.iloc[test_index, :-1].values)
            y_train, y_test = df.label[train_index], df.label[test_index]
            X_train, X_test = target_encoding(
                X_train, X_test, y_train.values, category_cols, model_id)
            lgb_model = LGBMClassifier(
                max_depth=11, objective="binary", learning_rate=0.005, 
                min_child_weight=0.005, reg_alpha=0, reg_lambda=0,
                num_leaves=128, n_estimators=10000, #device="gpu",
                random_state=0, colsample_bytree=0.8
            )
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric=lgb_metric,
                early_stopping_rounds=100,
                verbose=200
            )
            score += lgb_model.best_score_["valid_0"]["my_score"]/4
            pkl_save("lgb_model_fold_%d_seed_%d.dat"%(model_id, seed), lgb_model)
            importance += lgb_model.feature_importances_[:480]
            print("[LOG] 第%d个Fold的最佳迭代轮数为%d，得分为"%(
                model_id+1, lgb_model.best_iteration_), round(lgb_model.best_score_["valid_0"]["my_score"], 6))
        np.save("lgb_importance.npy", importance)
        print("[LOG] 交叉验证得分为", round(score, 6))

def tfidf_process(df):
    temp_df = pd.read_json("/mnt/atec/train.jsonl", encoding="utf-8", lines=True).set_index("id")
    text = [
        ' '.join(jieba.cut(str(x))) for x in temp_df['memo_polish'].fillna("")]
    tfidf = TfidfVectorizer(max_features=1200)
    model = tfidf.fit(text)
    pickle.dump(model, open("tfidf.pickle", "wb"))
    text2 = [
        ' '.join(jieba.cut(str(x))) for x in df['memo_polish'].fillna("")]
    data = model.transform(text2)
    df = pd.concat([
        df.iloc[:, :-2],
        pd.DataFrame(data.toarray().astype(np.float32)),
        df.iloc[:, [-1]]
    ], axis=1)
    return df

def get_category_cols(df):
    return np.arange(df.shape[1])[(df.dtypes=="int64")&(
        df.nunique().isin([2,3,4,5]))]

def groupby_transform(df):
    col_num = df.shape[1]
    target_cols = np.arange(col_num)[df.nunique()>5000]
    sub = df.iloc[:, target_cols]
    for col1 in tqdm(sub.columns):
        itv = pd.IntervalIndex(pd.qcut(sub[col1], 10, duplicates="drop"))
        for m in ["mean", "sum", "skew", "var"]:
            for col2 in sub.columns:
                if col1==col2:
                    continue
                s = df.groupby(itv)[col2].transform(m)
                df[col1+"_"+m+"_"+col2] = s
    return df

def operation(df):
    sub = df.copy()
    ipt = np.load("lgb_importance.npy")
    select = np.arange(ipt.shape[0])[ipt>np.quantile(ipt,0.95)]
    sub = sub.iloc[:, select].apply(lambda x: (x-x.mean())/x.std())
    for i in tqdm(range(sub.columns.shape[0]-1)):
        for j in range(i+1, sub.columns.shape[0]):
            col1 = sub.columns[i]
            col2 = sub.columns[j]
            df[col1+"_"+col2+"_add"] = sub[col1] + sub[col1]
            df[col1+"_"+col2+"_sub"] = sub[col1] - sub[col1]
            df[col1+"_"+col2+"_div"] = sub[col1] / sub[col1]
    return df

def get_nn(df):
    s = pd.Series(np.load("nn.npy"), index=df.index, name="nn")
    return pd.concat(
        [df.iloc[:, :-2], s, df.iloc[:, -2:]],
        axis=1
    )

def get_reduction(df):
    trans = FastICA(50, random_state=0)
    temp = df.iloc[:, :-2]
    temp = temp.apply(
        lambda x: x if x.std()==0 else (x-x.median())/x.std(), 0).fillna(0)
    res = trans.fit_transform(temp.values)
    res = pd.DataFrame(res, columns=["PCA_%d"%i for i in range(res.shape[1])])
    return pd.concat(
        [df.iloc[:, :-2], res, df.iloc[:, -2:]],
        axis=1
    )

def train():
    df = get_df()
    category_cols = get_category_cols(df.iloc[:, :-2])
    np.save("cate_cols.npy", category_cols)
    df = tfidf_process(df)
    lgb_run(df, category_cols)

if __name__ == '__main__':
    if train():
        sys.exit(0)
    else:
        sys.exit(1)
