# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 08:15:58 2020

@author: Asus
"""

import pixellib
from pixellib.instance import instance_segmentation
import cv2
import numpy as np
import matplotlib.pyplot as plt

segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5") 
# segment_image.segmentImage("test.jpg", output_image_name = "testQQ.jpg")
segmask, output = segment_image.segmentImage("test.jpg")
img = cv2.imread("test.jpg")
# img = img[segmask["rois"][0][0]:segmask["rois"][0][2], segmask["rois"][0][1]:segmask["rois"][0][3]]

array = np.array(segmask["masks"][:, :, 0])*255
array = array.astype("uint8")
contours, hierarchy = cv2.findContours(array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img , contours, -1, (0, 255, 0), 1)
resized = cv2.resize(img, (800,600), interpolation = cv2.INTER_AREA)
# cv2.imshow("test", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


out = np.zeros_like(img) # Extract out the object and place into output image
out[array == 255] = img[array == 255]
(y, x) = np.where(array == 255)
(topy, topx) = (np.min(y), np.min(x))
(bottomy, bottomx) = (np.max(y), np.max(x))
out = out[topy:bottomy+1, topx:bottomx+1]

cv2.imshow('Output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('Test_crop.png', out)
