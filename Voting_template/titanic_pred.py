# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:55:38 2020

"""

import numpy as np 
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier

gender_df = pd.read_csv('data/gender_submission.csv', header=0)
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df['Age'].fillna(value=0, inplace=True)
test_df['Age'].fillna(value=0, inplace=True)

mapping = {'male': 0, 'female': 1}
train_df = train_df.replace({"Sex" : mapping})
test_df = test_df.replace({"Sex" : mapping})

mapping = {'S': 0, 'C': 1, 'Q' : 2}
train_df = train_df.replace({"Embarked" : mapping})
test_df = test_df.replace({"Embarked" : mapping})

train_df['Embarked'].fillna(value=0, inplace=True)
test_df['Embarked'].fillna(value=0, inplace=True)

gender_array = np.asarray(gender_df.iloc[:, 0:2])
train_array = np.asarray(train_df.iloc[:, 0:12])
test_array = np.asarray(test_df.iloc[:, 0:12])


train_array = np.delete(train_array, [3, 8, 10], axis=1)
test_array = np.delete(test_array, [2, 7, 9], axis=1) 

train_array = np.array(train_array.tolist())
test_array = np.array(test_array.tolist())


train_Y = train_array[:, 1]
train_X = np.delete(train_array, 1, 1)

test_X = test_array
test_Y = gender_array[:, 1:2].reshape(-1, )


params = {
    'num_rounds': 500,
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 2,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.01,
    'seed': 1000,
    'nthread': 4,
}

plst = params.items()

num_rounds = 500

# dtrain = xgb.DMatrix(train_X, train_Y)
# model = xgb.train(plst, dtrain, num_rounds)
# dtest = xgb.DMatrix(test_X)
# ans = model.predict(dtest)


clf = XGBClassifier(n_estimators=100,
        # 如同學習率
        learning_rate= 0.3, 
        # 構建樹的深度，越大越容易過擬合    
        max_depth=6, 
        # 隨機取樣訓練樣本 訓練例項的子取樣比
        subsample=1, 
        # 用於控制是否後剪枝的引數,越大越保守，一般0.1、0.2這樣子
        gamma=0, 
        # 控制模型複雜度的權重值的L2正則化項引數，引數越大，模型越不容易過擬合。
        reg_lambda=1,  
        
        #最大增量步長，我們允許每個樹的權重估計。
        max_delta_step=0,
        # 生成樹時進行的列取樣 
        colsample_bytree=1, 

        # 這個引數預設是 1，是每個葉子裡面 h 的和至少是多少，對正負樣本不均衡時的 0-1 分類而言
        # 假設 h 在 0.01 附近，min_child_weight 為 1 意味著葉子節點中最少需要包含 100 個樣本。
        #這個引數非常影響結果，控制葉子節點中二階導的和的最小值，該引數值越小，越容易 overfitting。
        min_child_weight=1, 

        #隨機種子
        seed=1000)
clf.fit(train_X, train_Y,eval_metric='auc')
y_pred=clf.predict(test_X)
y_true= test_Y

print(str(accuracy_score(test_Y, y_pred)*100) + "%")






