from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
# import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_image
import face_preprocess

import matplotlib.pyplot as plt


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self, args):
    self.args = args
    ctx = mx.gpu(args.gpu)
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))

    self.threshold = args.threshold
    self.det_minsize = 50
    self.det_threshold = [0.6,0.7,0.8]
    #self.det_factor = 0.9
    self.image_size = image_size
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    if args.det==0:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
    else:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector

  def get_input_cb(self, face_img, image_size, alignmarks):
    ret = self.detector.detect_face(face_img, det_type=self.args.det)
    if ret is None:
      plt.imshow(face_img)
      pos = plt.ginput(5)
      points = np.array(pos).astype(np.int)
      bbox = None
    else:
      bbox, points = ret
      bbox = bbox[0, 0:4]
      points = points[0, :].reshape((2, 5)).T
    #print(bbox)
    #print(points)
    nimg = face_preprocess.preprocess(
        face_img, bbox, points, image_size=image_size, alignmarks=alignmarks)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2, 0, 1))
    return aligned

  # def get_input_cb_diffPoints(self, face_img):
  #   ret = self.detector.detect_face(face_img, det_type=self.args.det)
  #   if ret is None:
  #     plt.imshow(face_img)
  #     pos = plt.ginput(5)
  #     points = np.array(pos).astype(np.int)
  #     bbox = None
  #   else:
  #     bbox, points = ret
  #     bbox = bbox[0, 0:4]
  #     points = points[0, :].reshape((2, 5)).T
  #   two_points = np.zeros([2,2])
  #   two_points[0, 0] = (points[0, 0] + points[1, 0])/2
  #   two_points[0, 1] = (points[0, 1] + points[1, 1])/2
  #   two_points[1, 0] = (points[3, 0] + points[4, 0])/2
  #   two_points[1, 1] = (points[3, 1] + points[4, 1])/2
  #   print(two_points)
  #   #print(points)
  #   nimg = face_preprocess.preprocess_lightcnn(
  #       face_img, bbox, two_points, image_size='128,128')
  #   nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
  #   aligned = np.transpose(nimg, (2, 0, 1))
  #   return aligned