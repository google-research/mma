# coding=utf-8
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities."""

import os
import re

from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from sklearn.cluster import KMeans, MiniBatchKMeans


_GPUS = None
FLAGS = flags.FLAGS
flags.DEFINE_bool('log_device_placement', False, 'For debugging purpose.')


class EasyDict(dict):

  def __init__(self, *args, **kwargs):
    super(EasyDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


def get_config():
  config = tf.ConfigProto()
  if len(get_available_gpus()) > 1:
    config.allow_soft_placement = True
  if FLAGS.log_device_placement:
    config.log_device_placement = True
  config.gpu_options.allow_growth = True
  return config


def setup_tf():
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  tf.logging.set_verbosity(tf.logging.ERROR)


def smart_shape(x):
  s = x.shape
  st = tf.shape(x)
  return [s[i] if s[i].value is not None else st[i] for i in range(4)]


def ilog2(x):
  """Integer log2."""
  return int(np.ceil(np.log2(x)))


def find_latest_checkpoint(folder):
  """Replacement for tf.train.latest_checkpoint.

  It does not rely on the "checkpoint" file which sometimes contains
  absolute path and is generally hard to work with when sharing files
  between users / computers.

  Args:
    folder: string, path to the checkpoint directory.

  Returns:
    string, file name of the latest checkpoint.
  """
  r_step = re.compile(r'.*model\.ckpt-(?P<step>\d+)\.meta')
  matches = tf.gfile.Glob(os.path.join(folder, 'model.ckpt-*.meta'))
  matches = [(int(r_step.match(x).group('step')), x) for x in matches]
  ckpt_file = max(matches)[1][:-5]
  return ckpt_file


def get_latest_global_step(folder):
  """Loads the global step from the latest checkpoint in directory.

  Args:
    folder: string, path to the checkpoint directory.

  Returns:
    int, the global step of the latest checkpoint or 0 if none was found.
  """
  try:
    checkpoint_reader = tf.train.NewCheckpointReader(
        find_latest_checkpoint(folder))
    return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
  except:  # pylint: disable=bare-except
    return 0


def get_latest_global_step_in_subdir(folder):
  """Loads the global step from the latest checkpoint in sub-directories.

  Args:
    folder: string, parent of the checkpoint directories.

  Returns:
    int, the global step of the latest checkpoint or 0 if none was found.
  """
  sub_dirs = (
      x for x in tf.gfile.Glob(os.path.join(folder, '*'))
      if tf.gfile.Stat(x).IsDirectory())
  step = 0
  for x in sub_dirs:
    step = max(step, get_latest_global_step(x))
  return step


def getter_ema(ema, getter, name, *args, **kwargs):
  """Exponential moving average getter for variable scopes.

  Args:
    ema: ExponentialMovingAverage object, where to get variable moving averages.
    getter: default variable scope getter.
    name: variable name.
    *args: extra args passed to default getter.
    **kwargs: extra args passed to default getter.

  Returns:
    If found the moving average variable, otherwise the default variable.
  """
  var = getter(name, *args, **kwargs)
  ema_var = ema.average(var)
  return ema_var if ema_var else var


def model_vars(scope=None):
  return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def gpu(x):
  return '/gpu:%d' % (x % max(1, len(get_available_gpus())))


def get_available_gpus():
  global _GPUS
  if _GPUS is None:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    local_device_protos = device_lib.list_local_devices(session_config=config)
    _GPUS = tuple(
        [x.name for x in local_device_protos if x.device_type == 'GPU'])
  return _GPUS


# Adapted from:
#  https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. For each tower, a
      list of its gradients.

  Returns:
    List of pairs of (gradient, variable) where the gradient has been
        averaged across all towers.
  """
  if len(tower_grads) <= 1:
    return tower_grads[0]

  average_grads = []
  for grads_and_vars in zip(*tower_grads):
    grad = tf.reduce_mean([gv[0] for gv in grads_and_vars], 0)
    average_grads.append((grad, grads_and_vars[0][1]))
  return average_grads


def para_list(fn, *args):
  """Run on multiple GPUs in parallel and return list of results."""
  gpus = len(get_available_gpus())
  if gpus <= 1:
    return zip(*[fn(*args)])
  splitted = [tf.split(x, gpus) for x in args]
  outputs = []
  for gpu_id, x in enumerate(zip(*splitted)):
    with tf.name_scope('tower%d' % gpu_id):
      with tf.device(
          tf.train.replica_device_setter(
              worker_device='/gpu:%d' % gpu_id, ps_device='/cpu:0',
              ps_tasks=1)):
        outputs.append(fn(*x))
  return zip(*outputs)


def para_mean(fn, *args):
  """Run on multiple GPUs in parallel and return means."""
  gpus = len(get_available_gpus())
  if gpus <= 1:
    return fn(*args)
  splitted = [tf.split(x, gpus) for x in args]
  outputs = []
  for gpu_id, x in enumerate(zip(*splitted)):
    with tf.name_scope('tower%d' % gpu_id):
      with tf.device(
          tf.train.replica_device_setter(
              worker_device='/gpu:%d' % gpu_id, ps_device='/cpu:0',
              ps_tasks=1)):
        outputs.append(fn(*x))
  if isinstance(outputs[0], (tuple, list)):
    return [tf.reduce_mean(x, 0) for x in zip(*outputs)]
  return tf.reduce_mean(outputs, 0)


def para_cat(fn, *args):
  """Run on multiple GPUs in parallel and return concatenated outputs."""
  gpus = len(get_available_gpus())
  if gpus <= 1:
    return fn(*args)
  splitted = [tf.split(x, gpus) for x in args]
  outputs = []
  for gpu_id, x in enumerate(zip(*splitted)):
    with tf.name_scope('tower%d' % gpu_id):
      with tf.device(
          tf.train.replica_device_setter(
              worker_device='/gpu:%d' % gpu_id, ps_device='/cpu:0',
              ps_tasks=1)):
        outputs.append(fn(*x))
  if isinstance(outputs[0], (tuple, list)):
    return [tf.concat(x, axis=0) for x in zip(*outputs)]
  return tf.concat(outputs, axis=0)


def get_low_confidence_from_each_clusters(data, n_clusters, grow_size,
                                          confidences):
  """Cluster data into n_clusters clusters and pick low confidence samples from
  each cluster such that the total number of samples picked is grow_size."""
  data = data.reshape(data.shape[0], -1)  # reshape in case not a vector
  kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                           batch_size=n_clusters*4,
                           reassignment_ratio=0.1,
                           random_state=0, max_iter=1000,
                           ).fit(data)
  pred = kmeans.labels_
  sizes = np.histogram(pred, bins=range(n_clusters+1))[0]  # size of clusters
  grow_size_clusters = np.rint((grow_size * (sizes / np.sum(sizes)))).astype(int)
  # To make sure we get grow_size samples in total
  # If the current number of samples is smaller than grow_size, add randomly
  while np.sum(grow_size_clusters) < grow_size:
    idx = np.random.choice(np.where(sizes > grow_size_clusters)[0], 1)[0]
    grow_size_clusters[idx] += 1
  # If the current number of samples is larger than grow_size, remove from the
  # largest cluster
  while np.sum(grow_size_clusters) > grow_size:
    idx = np.argmax(grow_size_clusters)
    grow_size_clusters[idx] -= 1

  selected = []
  for c in range(n_clusters):
    idx_c = np.where(pred == c)[0]
    idx = idx_c[confidences[idx_c].argsort()[:grow_size_clusters[c]]]
    selected.append(idx)
  return np.concatenate(selected).reshape(-1)


def idx_to_fixlen(indices, length):
  """Pad zeros to indices to make it a certain length."""
  return np.concatenate([indices, np.zeros(max(0, length - indices.size),
                                           dtype=np.int32)-1])


def fixlen_to_idx(indices):
  """Get the non-negative values from indices."""
  nlabeled = np.where(indices >= 0)[0][-1] + 1
  return indices[:nlabeled]


def get_class_dist(labels, nclass):
  class_cnt = [0.0] * nclass
  for label in labels:
    class_cnt[label] += 1
  class_cnt = np.array(class_cnt)
  class_cnt /= sum(class_cnt)
  return class_cnt
