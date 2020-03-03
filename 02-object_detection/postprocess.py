# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import time
import sys
if 'lib\\models\\research\\' not in sys.path:
    sys.path.append('lib\\models\\research\\')
    
import object_detection.utils.visualization_utils
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2
import pickle

# +
with open('label_map.pbtxt', 'rb') as f:
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    text_format.Merge(f.read(), label_map)
    
label_map_dict = {i.id: {'name' : i.name} for i in label_map.item}

# -

model = tf.saved_model.load('data/frcnn_model/saved_model')
model = model.signatures['serving_default']

out_fn = 'data/out1.avi'

fn = 'data/eval/Videos/видео-20200124-100159-d6ad16e5.MOV'
vid = cv2.VideoCapture(fn)
if not vid.isOpened(): raise ValueError("Error opening file '" + fn + "'")
try:
    with open(out_fn + '.boxes1', 'wb') as out_boxes:
        while True:
            ret, frame = vid.read()
            if not ret: break
            frame = np.flip(np.transpose(frame, [1,0,2]), 1)
            im = tf.convert_to_tensor(frame)[tf.newaxis,...]
            res = model(im)
            for nm in ['detection_scores', 'detection_boxes', 'detection_classes']:
                pickle.dump(res[nm].numpy(), out_boxes)
                print('frame')
finally:
    vid.release()
    

fn = 'data/eval/Videos/видео-20200124-100159-d6ad16e5.MOV'
vid = cv2.VideoCapture(fn)
if not vid.isOpened(): raise ValueError("Error opening file '" + fn + "'")
try:
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if os.path.exists(out_fn): os.remove(out_fn)
    out = cv2.VideoWriter(
        out_fn, 
        fourcc=cv2.VideoWriter_fourcc(*'MP42'), 
        fps=float(vid.get(cv2.CAP_PROP_FPS)), 
#         frameSize = (width, height),
        frameSize = (height, width),
    )
    try:
        with open(out_fn + '.boxes', 'rb') as in_boxes:
            while True:
                ret, frame = vid.read()
                if not ret: break
                frame = np.flip(np.transpose(frame, [1,0,2]), 1)

                detection_scores, detection_boxes, detection_classes = [pickle.load(in_boxes) for _ in range(3)]
                flt = detection_scores > .8
                im = object_detection.utils.visualization_utils.visualize_boxes_and_labels_on_image_array(
                      frame,  
                      detection_boxes[flt],
                      detection_classes[flt].astype(np.int),
                      detection_scores[flt],
                      label_map_dict,
                      instance_masks=None,
                      use_normalized_coordinates=True,
                      line_thickness=8
                )
                out.write(im)
                print("frame")
    finally:
        out.release()
finally:
    vid.release()
    


