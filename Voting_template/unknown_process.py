
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from statsmodels.imputation import mice
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
import catboost as cat
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# In[2]:


training = pd.read_csv('train_data.csv',header=None)
label = training[17].values
training = training.drop(17,axis=1)
training = training.drop(0,axis=1)

testing = pd.read_csv('test_data.csv',header=None)
test_idx = testing[0].values
testing = testing.drop(0,axis=1)
training_len = len(training)




# In[3]:


def count_mode(data_arr,dim):
    vec = data_arr[:,dim]
    ct = {}
    for v in vec:
        if v in ct:
            ct[v] += 1
        else:
            ct[v] = 1
    ct.pop('unknown')
    max_count = 0
    ret_k = 0
    for k , v in ct.items():
        if v > max_count:
            max_count = v
            ret_k = k
    return ret_k


# In[4]:


all_data = pd.concat([training,testing])
cat_feat = [2,3,4,5,7,8,9,11,16]
cat_feat_np = [i-1 for i in cat_feat]


for col in cat_feat:
    mp = {}
    cnt = 0
    for a in all_data[col]:
        if a != 'unknown' and a not in mp:
            mp[a] = cnt
            cnt += 1
    mp['unknown'] = 'unknown'
    all_data[col] = all_data[col].map(mp)
    print(sum(all_data[col] == 'unknown'))
all_data_arr = all_data.values


# In[5]:


for cur_dim in cat_feat_np:
    print(cur_dim)
    all_index = [i for i in range(all_data_arr.shape[1])]
    for d in cat_feat_np:
        all_index.remove(d)
    tmp_data = all_data_arr[:,all_index]
    unk_mask = all_data_arr[:,cur_dim]=='unknown'

    tmp_label = all_data_arr[:,cur_dim][~unk_mask].astype(int)
    tmp_train = tmp_data[~unk_mask]
    tmp_test = tmp_data[unk_mask]
    pred = np.zeros(tmp_test.shape[0])
    mode_num = count_mode(all_data_arr,cur_dim)
    print('mode_num is {}'.format(mode_num))
    pred[:] = mode_num
    all_data_arr[:,cur_dim][unk_mask] = pred[:]


# In[6]:

'''
imp = mice.MICEData(all_data_arr)
mice = mice.MICE(data = imp)
all_data_arr = mice.fit(10, 10)
all_data_arr = all_data_arr.astype(float)
'''

# In[7]:


training_data = all_data_arr[:training_len]
testing_data = all_data_arr[training_len:]

#training_data, testing_data, label, label_t = train_test_split(training_data, label, test_size = 0.2, random_state = 0)


lgb_param_dist = {
   'objective': 'binary',
   'learning_rate': 0.01,
   'subsample': 0.05, 
   'scale_pos_weight': 7.33,
   'n_estimators': 1500,
   'num_leaves': 100,
   'max_depth': 13,
   'colsample_bytree': 0.7,
   'reg_alpha' : 2,
   'reg_lambda': 1,
   'min_split_gain': 0.1, 
   'n_jobs': -1
}

parameters = {
   #'hidden_layer_sizes': [ (100, 2), (100, 80, 60, 40, 20 ,10)]
}


xgb_param_dist = {
	'max_depth': 7, 
	'learning_rate': 0.05, 
	'n_estimators': 500, 
	'silent': True,
	'objective': 'binary:logistic',
	'njobs': -1,
	'nthread': -1,
	'min_child_weight': 5, 
	'subsample': 0.8,
	'colsample_bytree': 0.7,
	#'random_state': 0,
	'gamma': 0.5,
    'reg_alpha': 3,
    'reg_lambda': 3,
	'scale_pos_weight': 7.33 
}


cat_param_dist = {
	'iterations': 2000, 
	'learning_rate': 0.03, 
	'depth': 9, 
	'grow_policy': 'Lossguide',
	'min_data_in_leaf': 6,
	'l2_leaf_reg': 12,
	'task_type': 'GPU',
	'loss_function': 'Logloss',
	'thread_count': -1,
	'scale_pos_weight': 7.33
}

mlp_param_dist = {
    'solver': 'adam',
    'alpha': 4,
    'hidden_layer_sizes':(100, 2),
	'max_iter': 300,
    #'learning_rate': 'adaptive',
    #'learning_rate_init': 1,
    'random_state':1,
	#'n_jobs': -1
	'verbose': 1
}

#sel = ExtraTreesClassifier(n_estimators = 100)
#sel = sel.fit(training_data, label)
#model = SelectFromModel(sel, prefit=True)
#training_data = model.transform(training_data)
#testing_data = model.transform(testing_data)
clf = lgb.LGBMClassifier(**lgb_param_dist)
#clf = cat.CatBoostClassifier(**cat_param_dist)
#sm = SMOTE()
#training_data, label = sm.fit_resample(training_data, label)
#clf = MLPClassifier(**mlp_param_dist)
#clf = GridSearchCV(gsearch , param_grid = parameters, scoring = 'f1', cv = 3)
clf.fit(training_data , label)
prediction = clf.predict(testing_data)

#print(classification_report(label_t, prediction, target_names=['0', '1']))
'''
print("Best score: %0.3f" % clf.best_score_)
print("Best parameters set:")
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
# In[8]:
'''

with open('hw4_lgb.csv','w') as f:
    f.write('Id,Prediction\n')
    for idex , p in zip(test_idx,prediction):
        f.write('{},{}\n'.format(idex,int(p)))
