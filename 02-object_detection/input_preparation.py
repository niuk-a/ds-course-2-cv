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
import xml.etree.ElementTree
import tensorflow as tf
import os
import numpy as np
import hashlib
import time
import sys
if 'lib\\models\\research\\' not in sys.path:
    sys.path.append('lib\\models\\research\\')
    
import object_detection.utils.visualization_utils
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2

import joblib

# +
name2type={
    'int64' : tf.train.Int64List,
    'bytes' : tf.train.BytesList,
    'float' : tf.train.FloatList
}
def feature(*values, dtype):
    return tf.train.Feature(**{dtype + '_list' : name2type[dtype](value=values)})

def func(prefix, out_file, label_map_dict, fn):
    with tf.io.TFRecordWriter(out_file) as writer:
        t = time.time()
        with open(prefix + '/Annotations/' + fn[:-4] + '.xml') as f:
            doc = xml.etree.ElementTree.parse(f)
        images = sorted(doc.findall('image'), key=lambda im: int(im.attrib['id']))
        # Because our labeling can contains holes, this code calculate shifts to came into next labeled frame.
        # for example images with ids=[3, 6, 7, 10] will give us deltas=[4, 3, 1, 3]
        d = np.array([im.attrib['id'] for im in images], dtype=int)
        d[1:] -= d[:-1]
        d[0] += 1
        vid = cv2.VideoCapture(prefix + '/Videos/' + fn)
        if not vid.isOpened(): raise ValueError("Error opening file '" + fn + "'")
        try:
            for next_frame, im in zip(d, images):
                width = int(im.attrib['width'])
                height = int(im.attrib['height'])
                frame_name = im.attrib['name']
                xmin = []
                ymin = []
                xmax = []
                ymax = []
                classes = []
                classes_text = []
                for obj in im:
                    for l, n, d in zip(
                        [xmin, ymin, xmax, ymax],
                        ['xtl', 'ytl', 'xbr', 'ybr'], 
                        [width, height, width, height]
                    ):
                        l.append(float(obj.attrib[n]) / d)    
                    label = obj.attrib['label']
                    classes_text.append(label.encode('utf8'))
                    classes.append(label_map_dict[label])
                # -------> read next frame
                for i in range(next_frame):
                    ret, frame = vid.read()
                    if not ret: raise ValueError("can't read next frame")
                im_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
                im_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if im_width != width:
                    assert im_width == height and im_height == width, fn +'#' + frame_name
                    frame = np.flip(np.transpose(frame, [1,0,2]), 1)
                else: 
                    assert im_width == width and im_height == height, fn +'#' + frame_name
                ret, im = cv2.imencode('.jpeg', frame)
                if not ret: raise ValueError("can't convert frame to jpeg")
                bb = im.tobytes()
                key = hashlib.sha256(bb).hexdigest()

                filename = prefix + '/Videos/' + fn + '#' + frame_name
                record = tf.train.Example(features=tf.train.Features(feature={
                    'image/height': feature(height, dtype='int64'),
                    'image/width': feature(width, dtype='int64'),
                    'image/filename': feature(filename.encode('utf8'), dtype='bytes'),
                    'image/source_id': feature(filename.encode('utf8'), dtype='bytes'),
                    'image/key/sha256': feature(key.encode('utf8'), dtype='bytes'),
                    'image/encoded': feature(bb, dtype='bytes'),
                    'image/format': feature(b'jpeg', dtype='bytes'),
                    'image/object/bbox/xmin': feature(*xmin, dtype='float'),
                    'image/object/bbox/xmax': feature(*xmax, dtype='float'),
                    'image/object/bbox/ymin': feature(*ymin, dtype='float'),
                    'image/object/bbox/ymax': feature(*ymax, dtype='float'),
                    'image/object/class/text': feature(*classes_text, dtype='bytes'),
                    'image/object/class/label': feature(*classes, dtype='int64'),
                    'image/object/difficult': feature(*([0] * len(classes)), dtype='int64'),
                    'image/object/truncated': feature(*([0] * len(classes)), dtype='int64'),
                    'image/object/view' : feature(*([b'Frontal'] * len(classes)), dtype='int64'),
                }))
                writer.write(record.SerializeToString())
        finally:
            vid.release()
        print(fn, time.time() - t)
        
prefix = 'test'
assert set(n[:-4] for n in os.listdir('data/' + prefix + '/Annotations/')) == \
        set(n[:-4] for n in os.listdir('data/' + prefix + '/Videos/'))

with open('label_map.pbtxt', 'rb') as f:
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    text_format.Merge(f.read(), label_map)
    
label_map_dict = {i.name: i.id for i in label_map.item}

joblib.Parallel(n_jobs=6)( joblib.delayed(func)(
    'data/' + prefix, 'data/ds/' + prefix + '/' + fn[:-4] + '.record', label_map_dict, fn
) for fn in os.listdir('data/' + prefix + '/Videos/'))

# +
for r in tf.data.TFRecordDataset('data/ds/train/VID_20200120_155502.record').take(1): pass
record = tf.train.Example()
record.ParseFromString(r.numpy())

im = cv2.imdecode(np.frombuffer(
    record.features.feature['image/encoded'].bytes_list.value[0], 
    dtype=np.int8,
), flags=1)

for xmin, xmax, ymin, ymax, name in zip(
    record.features.feature['image/object/bbox/xmin'].float_list.value,
    record.features.feature['image/object/bbox/xmax'].float_list.value,
    record.features.feature['image/object/bbox/ymin'].float_list.value,
    record.features.feature['image/object/bbox/ymax'].float_list.value,
    record.features.feature['image/object/class/text'].bytes_list.value
):
    object_detection.utils.visualization_utils.draw_bounding_box_on_image_array(
        im, ymin, xmin, ymax, xmax,
        color='red', thickness=4,
        display_str_list=[name.decode('UTF-8')],
        use_normalized_coordinates=True
    )
    
plt.figure(figsize=(16,16))
plt.imshow(im)
# -

for fn in os.listdir('data/ds/test'):
    with tf.io.TFRecordWriter('data/ds2/test_' + fn) as writer:
        for r in tf.data.TFRecordDataset('data/ds/test/' + fn):
            record = tf.train.Example()
            record.ParseFromString(r.numpy())        
            sz = len(record.features.feature['image/object/class/label'].int64_list.value)
            record.features.feature['image/object/view'].bytes_list.value.extend([b'Frontal'] * sz)
            ymin = list(record.features.feature['image/object/bbox/ymin'].float_list.value)
            ymax = list(record.features.feature['image/object/bbox/ymax'].float_list.value)
            record.features.feature['image/object/bbox/ymin'].float_list.Clear()
            record.features.feature['image/object/bbox/ymax'].float_list.Clear()
            record.features.feature['image/object/bbox/ymin'].float_list.value.extend(ymax)
            record.features.feature['image/object/bbox/ymax'].float_list.value.extend(ymin)
            writer.write(record.SerializeToString())



# +
for r in tf.data.TFRecordDataset('data/ds2/test_видео-20200124-095814-bfb8e79d.record').take(300): pass
record = tf.train.Example()
record.ParseFromString(r.numpy())

im = cv2.imdecode(np.frombuffer(
    record.features.feature['image/encoded'].bytes_list.value[0], 
    dtype=np.int8,
), flags=1)

for xmin, xmax, ymin, ymax, name in zip(
    record.features.feature['image/object/bbox/xmin'].float_list.value,
    record.features.feature['image/object/bbox/xmax'].float_list.value,
    record.features.feature['image/object/bbox/ymin'].float_list.value,
    record.features.feature['image/object/bbox/ymax'].float_list.value,
    record.features.feature['image/object/class/text'].bytes_list.value
):
    object_detection.utils.visualization_utils.draw_bounding_box_on_image_array(
        im, ymin, xmin, ymax, xmax,
        color='red', thickness=4,
        display_str_list=[name.decode('UTF-8')],
        use_normalized_coordinates=True
    )
    
plt.figure(figsize=(8,8))
plt.imshow(im)
