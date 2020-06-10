# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:42:43 2020

@author: VMRL-Robot
"""

import numpy as np
import keras
from keras.datasets import mnist
from keras import backend as K 
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input, add, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras import regularizers
from keras.utils import to_categorical

file_name = "CNN"
 
def get_data():
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
    train_Y = to_categorical(train_Y)
    test_Y = to_categorical(test_Y)
    return train_X, test_X, train_Y, test_Y

def get_model():
    input_shape=(train_X.shape[1], train_X.shape[2], 1)
    X_input = Input(input_shape)
    
    X = Conv2D(filters=64, kernel_size=(2, 2), padding = 'same', activation='relu')(X_input)
    X = Conv2D(filters=32, kernel_size=(4, 4), padding = 'same', activation='relu')(X)

    X = MaxPooling2D(pool_size=2)(X)

    X = Flatten()(X)
    X = Dense(6, kernel_initializer='normal', activation='relu')(X)
#    X = Dense(1, kernel_initializer='normal')(X)
    X = Dense(10, kernel_initializer='normal', activation='softmax')(X)

    model = Model(inputs = X_input, outputs = X, name='Conv2D')   
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True) 
    

    # Compile model
#    model.compile(loss='mean_squared_error', optimizer=adam)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
#    model.summary()
    
    return model




batch_size =64
learning_rate = 1*10e-5
EPOCHS = 10

train_loss = []
val_loss = []
test_loss = []
R_square = []


K.clear_session()
train_X, test_X, train_Y, test_Y = get_data()


model = get_model()
callbacks = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min')
mcp_save = keras.callbacks.ModelCheckpoint(file_name + '/best_weights.h5', save_best_only=True, monitor='val_loss', mode='min')
#    callbacks_list = [callbacks, mcp_save]
callbacks_list = [mcp_save]
history = model.fit(train_X, train_Y, batch_size=batch_size, epochs=EPOCHS, verbose=1, validation_split=0.2, callbacks=callbacks_list)


K.clear_session()
model = keras.models.load_model(file_name + '/best_weights.h5')
score = model.evaluate(test_X, test_Y, verbose=0)
pred = model.predict(test_X)
print('Test loss:', score[0])
print('Test acc:', score[1])


plt.plot(history.history['acc'], label='training data')
plt.plot(history.history['val_acc'], label='validation data')

plt.title('Training curve')
plt.ylabel('ACC value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()






