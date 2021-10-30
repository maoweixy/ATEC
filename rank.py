#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import pickle
import jieba
import pandas as pd
from xgboost import XGBClassifier
import numpy as np


input_data_path = '/home/admin/workspace/job/input/test.jsonl'
output_predictions_path = '/home/admin/workspace/job/output/predictions.jsonl'

def tfidf_process(df):
    text = [' '.join(jieba.cut(str(x))) for x in df['memo_polish']]
    model = pickle.load(open("tfidf.pickle", "rb"))
    data = model.transform(text)
    df = pd.concat([
        df.iloc[:, :-1],
        pd.DataFrame(data.toarray().astype(np.float32))
    ], axis=1)
    return df

def rank():
    df = pd.read_json(input_data_path, encoding="utf-8", lines=True).set_index("id")
    df = tfidf_process(df)
    y_pred = 0
    for model_id in range(4):
        loaded_model = pickle.load(open("lgb_model_%d.dat"%model_id, "rb"))
        y_pred += loaded_model.predict_proba(df.values)[:, 1]/4
    prediction = pd.DataFrame({"id": ["%d"%(i) for i in range(df.shape[0])], "label": y_pred})
    prediction.to_json(output_predictions_path, lines=True, orient='records')
    return True


if __name__ == '__main__':
    if rank():
        sys.exit(0)
    else:
        sys.exit(1)