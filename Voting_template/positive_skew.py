
# coding: utf-8

# In[5]:


'''
Import necessary library
'''

import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
import time
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import random

# In[6]:


'''
Read input data
'''
t1 = time.time()
train_data = pd.read_csv('train.csv',header=None)
test_data = pd.read_csv('test.csv',header=None)
train_size = len(train_data)
test_size = len(test_data)
print(train_size,test_size)


# In[7]:


'''
Deal with input data
'''
test_index = test_data.loc[:,0]
test_data = test_data.drop(0,axis = 1)
def test_rename(x):
    return x+1
test_data = test_data.rename(mapper=test_rename,axis=1)
train_label = train_data.iloc[:,1]
train_data_d = train_data.drop(labels=[0,1],axis=1)


# In[8]:


concated_df = pd.concat((train_data_d,test_data))


# In[9]:


cat_feature = [2,5,6,7,8,9,10,11,13,14,15,24,25,27]
num_feature = [3,4,12,16,17,18,19,20,21,22,23,26,28]


# In[10]:


'''
Deal with missing value
Fill category feature with mode, numeric feature with mean
'''
for col in cat_feature:
    concated_df[col] = concated_df[col].fillna(concated_df[col].mode()[0])
for col in num_feature:
    concated_df[col] = concated_df[col].fillna(concated_df[col].mean())
concated_df = pd.get_dummies(concated_df,columns=cat_feature)


# In[11]:


train_data_d = concated_df.iloc[:train_size,:]
test_data = concated_df.iloc[train_size: , : ]
print(len(train_data_d),len(test_data) )

# In[12]:


'''
Train/test dataframe into numpy array
'''
X_sp_train = train_data_d.values
y_sp_train = train_label.values
X_test = test_data.values


# In[13]:


'''
split to training set/testing set
'''
#X_sp_train, X_test, y_sp_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#print(np.sum(y_sp_train == 0), np.sum(y_sp_train == 1))
'''
SMOTE to oversampling
'''
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_sp_train, y_sp_train)
#print(np.sum(y_train == 0), np.sum(y_train == 1))

'''
Standardlize numeric feature
'''
std = StandardScaler()
std.fit(X_train[:,:14])
X_train[:,:14] = std.transform(X_train[:,:14])
X_test[:,:14] = std.transform(X_test[:,:14])
std.fit(X_sp_train[:,:14])
X_sp_train[:,:14] = std.transform(X_sp_train[:,:14])

print(len(X_train),len(X_test))

# In[25]:


'''
Train model
'''

print('Build Model')

#parameters = {
	#'max_depth': [3, 5, 7, 9],
    
    #'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    #'n_estimators':[100,200,500,1000,1500],
    #'min_child_weight': [1, 3, 5],
    #'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    #'subsample': [0.6, 0.7, 0.8, 0.9],
    #'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    #'reg_alpha': [0.05, 0.1, 1, 2, 3],
    #'reg_lambda': [0.05, 0.1, 1, 2, 3]
    #'''
#}


xgb_param_dist = {
	'max_depth': 5, 
	'learning_rate': 0.01, 
	'n_estimators': 1500, 
	'silent': True,
	'objective': 'binary:logistic',
	'njobs': -1,
	'nthread': -1,
	'min_child_weight': 7,
	'subsample': 0.9,
	'colsample_bytree': 0.7,
	#'random_state': 0,
	'gamma': 0.7,
	'scale_pos_weight': 7.33,
    'reg_alpha': 3,
    'reg_lambda': 1
}


lgb_param_dist = {
   'objective': 'binary',
   'learning_rate': 0.05,
   'subsample': 0.9, 
   'scale_pos_weight': 7.33,
   'n_estimators': 3000,
   'num_leaves': 80,
   'max_depth': 7,
   'colsample_bytree': 0.7,
   'reg_alpha' : 3,
   'reg_lambda': 1,
   'min_split_gain': 0.6, 
   'n_jobs': -1
}

cat_param_dist = {
	'iterations': 1500, 
	'learning_rate': 0.03, 
	'depth': 8, 
	#'border': 0.4,
	'grow_policy': 'Lossguide',
	'min_data_in_leaf': 7,
	#'calc_feature_importance': True,
	'l2_leaf_reg': 13,
	'task_type': 'GPU',
	'loss_function': 'Logloss',
	'thread_count': -1,
	'scale_pos_weight': 7.33
}

test_param_dist = {
	'alpha': sp_randint(1, 10)
}

mlp_param_dist = {
    'solver': 'adam',
    'alpha': 4,
    'hidden_layer_sizes':(45, 25, 15),
	'max_iter': 200,
    #'learning_rate': 'adaptive',
    #'learning_rate_init': 1,
    'random_state':1,
	#'n_jobs': -1
	'verbose': 1
}


clf1 = xgb.XGBClassifier(**xgb_param_dist)
clf2 = cat.CatBoostClassifier(**cat_param_dist)
clf3 = lgb.LGBMClassifier(**lgb_param_dist)
clf4 = MLPClassifier(**mlp_param_dist)
print('Training')
clf1.fit(X_sp_train, y_sp_train)
clf2.fit(X_sp_train, y_sp_train)
clf3.fit(X_sp_train, y_sp_train)
clf4.fit(X_train, y_train)


#clf = RandomizedSearchCV(clf1, test_param_dist, cv = 3, scoring = 'f1')

#best_estimator = clf.best_estimator_
#print(best_estimator)
#print(clf.best_score_)

# In[26]:

'''
print("Best score: %0.3f" % clf.best_score_)
print("Best parameters set:")
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
'''
#clf = VotingClassifier(estimators=[('xgb', clf1), ('cat', clf2), ('lgb', clf3)], voting='hard')
#clf1.fit(X_train, y_train)
#clf2.fit(X_train, y_train)
#clf3.fit(X_train, y_train)
print('Testing')
pred1 = clf1.predict(X_test)
pred2 = clf2.predict(X_test)
pred3 = clf3.predict(X_test)
pred4 = clf4.predict(X_test)
#pred_train = clf.predict(X_train)
#F1 = f1_score(y_test, pred)  
#acc = accuracy_score(y_test, pred)

#print(clf.get_params())
#print("training_acc: %4.g" % accuracy_score(y_train, pred_train))
#print("training_f1: %4.g" % f1_score(y_train, pred_train))
#print("accuracy : %.4g" % acc)
#print("F1 score:", F1)

#print('iter:', clf.n_iter_)
#print('layer:', clf.n_layers_)
# In[27]:

#print(classification_report(y_test, pred, target_names=['0', '1']))

'''
Output prediction
'''
'''
count = 0
with open('dnn_testing.csv','w') as f:
    f.write('Id,Prediction\n')
    for idx , p in zip(test_index , pred):
        f.write('{},{}\n'.format(idx,p))
        count = count + 1
print(count)
'''

count = 0
with open('voting_testing.csv','w') as f:
    f.write('Id,Prediction\n')
    for idx , p1, p2, p3, p4 in zip(test_index , pred1, pred2, pred3, pred4):
        if p1+int(p2)+p3+p4 >= 3:
            f.write('{},{}\n'.format(idx, 1))
        elif p1+int(p2)+p3+p4 <= 1:
            f.write('{},{}\n'.format(idx, 0))
        else:
            i = random.random()
            if i >= 0.5:
                f.write('{},{}\n'.format(idx, 1))
            else:
                f.write('{},{}\n'.format(idx, 0))
        count = count + 1
print(count)

'''
with open('log.txt', 'a') as f:
	f.write('score: '+str(F1)+'\n')
	f.write('accuracy: '+str(acc)+'\n')
'''
t2 = time.time()
print("time:", t2-t1)
