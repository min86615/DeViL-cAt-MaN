import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, GaussianNoise, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split


#read in the dataset
df = pd.read_csv('data/diabetes_data.csv')

#take a look at the data
df.head()

#split data into inputs and targets
X = df.drop(columns = ['diabetes'])
y = df['diabetes']

#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)





def Sequential_model():
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=8, activation='relu'))
    model.add(GaussianNoise(1.0))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

print("Train shape {}".format(X_train.shape))
print("Train labels shape {}".format(y_train.shape))
print("Test shape {}".format(X_test.shape))
print("Test labels shape {}".format(y_test.shape))


clf1 = LogisticRegression()
clf2 = KerasClassifier(build_fn=Sequential_model, epochs=10, batch_size=4, verbose=1)
clf2._estimator_type = "classifier"
clf = [("DNN1", clf1), ('DNN2', clf2)]
# clf = [("DNN1", clf1), ('DNN2', clf2)]
eclf = VotingClassifier(clf, weights=[1.0,1.0], voting="soft")
eclf.fit(X_train,y_train)
y_hat = eclf.predict(X_test)
result = f1_score(y_true=y_test, y_pred=y_hat)
print(str(accuracy_score(y_test, y_hat)*100) + "%")
print("f1 score : ", result)


