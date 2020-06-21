# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 00:13:33 2020

@author: Username
"""

import os
import numpy as np
import random
import shutil
# from random import shuffle
from shutil import copytree


DATASET_PATH  = "data\\train"

# if not os.path.isdir(os.path.join(VAL_DATASET_PATH, "one")):
#     os.mkdir(os.path.join(VAL_DATASET_PATH, "one"))

# validation_ratio = 0.3
for data in range(42):
    label_list = os.listdir(DATASET_PATH)
    one_label=label_list[data]
    VAL_DATASET_PATH = "onevsall_%s"%(one_label)
    if not os.path.isdir(VAL_DATASET_PATH):
        os.mkdir(VAL_DATASET_PATH)
    if not os.path.isdir(os.path.join(VAL_DATASET_PATH, "all")):
        os.mkdir(os.path.join(VAL_DATASET_PATH, "all"))
    copytree(os.path.join(DATASET_PATH, one_label), os.path.join(VAL_DATASET_PATH, "one"))
    label_list.pop(data)
    image_list = []
    for label in label_list:
        # if not os.path.isdir("%s//%s"%(VAL_DATASET_PATH, label)):
        #     os.mkdir(os.path.join(VAL_DATASET_PATH, label))
        # image_list.append(os.listdir(os.path.join(DATASET_PATH, label)))
        image_list = os.listdir(os.path.join(DATASET_PATH, label))
        random.shuffle(image_list)
        length = int(50)
        val_data = image_list[:length]
        # print(len(val_data))
        for i in val_data:
            shutil.copy(os.path.join(DATASET_PATH, label, i), os.path.join(VAL_DATASET_PATH, "all", i))
            
    print(data+1)        
    # if data == 0:
    #     break
        
    