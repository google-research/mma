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

"""Input data for image models.
"""

import functools
import itertools
import os

import numpy as np
import tensorflow as tf
from absl import flags
from tqdm import tqdm

from libml import utils

_DATA_CACHE = None
DATA_DIR = os.environ['ML_DATA']
flags.DEFINE_string('dataset', 'cifar10.1@250', 'Data to train on.')
flags.DEFINE_integer('para_parse', 4, 'Parallel parsing.')
flags.DEFINE_integer('para_augment', 4, 'Parallel augmentation.')
flags.DEFINE_integer('shuffle', 8192, 'Size of dataset shuffling.')
flags.DEFINE_string('p_unlabeled', '', 'Probability distribution of unlabeled.')
flags.DEFINE_bool('whiten', False, 'Whether to normalize images.')
FLAGS = flags.FLAGS


def record_parse(serialized_example):
  features = tf.parse_single_example(
    serialized_example,
    features={'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64)})
  image = tf.image.decode_image(features['image'])
  image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
  label = features['label']
  return dict(image=image, label=label)


def record_parse_mnist(serialized_example):
  features = tf.parse_single_example(
    serialized_example,
    features={'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64)})
  image = tf.image.decode_image(features['image'])
  image = tf.pad(image, [[2] * 2, [2] * 2, [0] * 2])
  image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
  label = features['label']
  return dict(image=image, label=label)


def record_parse_orig(serialized_example):
  # to keep the original image without processing
  features = tf.parse_single_example(
    serialized_example,
    features={'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64)})
  image = tf.image.decode_image(features['image'])
  label = features['label']
  return dict(image=image, label=label)


def default_parse(dataset, parse_fn=record_parse):
  para = 4 * max(1, len(utils.get_available_gpus())) * FLAGS.para_parse
  return dataset.map(parse_fn, num_parallel_calls=para)


mnist_parse = functools.partial(default_parse, parse_fn=record_parse_mnist)


def memoize(dataset):
  data = []
  with tf.Graph().as_default(), tf.Session(config=utils.get_config()) as session:
    dataset = dataset.prefetch(16)
    it = dataset.make_one_shot_iterator().get_next()
    try:
      while 1:
        data.append(session.run(it))
    except tf.errors.OutOfRangeError:
      pass
  images = np.stack([x['image'] for x in data])
  labels = np.stack([x['label'] for x in data])

  def tf_get(index):
    def get(index):
      return images[index], labels[index]

    image, label = tf.py_func(get, [index], [tf.float32, tf.int64])
    return dict(image=image, label=label)

  dataset = tf.data.Dataset.range(len(data)).repeat()
  dataset = dataset.shuffle(len(data) if len(data) < FLAGS.shuffle else FLAGS.shuffle)
  return dataset.map(tf_get)


def augment_mirror(x):
  return tf.image.random_flip_left_right(x)


def augment_shift(x, w):
  y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode='REFLECT')
  return tf.random_crop(y, tf.shape(x))


def augment_resize(x, p):
  height, width = 28, 28
  pixeldiff = tf.random.uniform(1, minval=-p*s, maxval=p*s, dtype=tf.dtypes.int32)
  new_image = tf.image.resize_image_with_pad(x, height+pixeldiff, width+pixeldiff)
  return tf.image.resize_with_crop_or_pad(new_image, tf.shape(x))


# copied from https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/#zooming
def augment_zoom(x):
  # Generate 20 crop settings, ranging from a 1% to 20% crop.
  scales = list(np.arange(0.9, 1.0, 0.01))
  boxes = np.zeros((len(scales), 4))
  for i, scale in enumerate(scales):
    x1 = y1 = 0.5 - (0.5 * scale)
    x2 = y2 = 0.5 + (0.5 * scale)
    boxes[i] = [x1, y1, x2, y2]

  def random_crop(img): # Create different crops for an image, return a random crop
    crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=tf.shape(img))
    return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

  choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

  # Only apply cropping 50% of the time
  return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))


def augment_noise(x, std):
  return x + std * tf.random_normal(tf.shape(x), dtype=x.dtype)


def compute_mean_std(data):
  data = data.map(lambda x: x['image']).batch(1024).prefetch(1)
  data = data.make_one_shot_iterator().get_next()
  count = 0
  stats = []
  with tf.Session(config=utils.get_config()) as sess:
    def iterator():
      while True:
        try:
          yield sess.run(data)
        except tf.errors.OutOfRangeError:
          break

    for batch in tqdm(iterator(), unit='kimg', desc='Computing dataset mean and std'):
      ratio = batch.shape[0] / 1024.
      count += ratio
      stats.append((batch.mean((0, 1, 2)) * ratio, (batch ** 2).mean((0, 1, 2)) * ratio))
  mean = sum(x[0] for x in stats) / count
  sigma = sum(x[1] for x in stats) / count - mean ** 2
  std = np.sqrt(sigma)
  print('Mean %s  Std: %s' % (mean, std))
  return mean, std


class DataSet:
  pass


def dataset(filenames):
  filenames = sorted(sum([tf.gfile.Glob(x) for x in filenames], []))
  if not filenames:
    raise ValueError('Empty dataset, did you mount gcsfuse bucket?')
  return tf.data.TFRecordDataset(filenames)


class DynamicDataset:
  def __init__(self, name, graph, train_filenames, test_filenames, parse_fn=record_parse,
               augment=(lambda x: x, lambda x: x), height=32, width=32, colors=3, nclass=10,
               mean=0, std=1, p_labeled=None, p_unlabeled=None):
    self.name = name
    self.graph = graph
    self.session = tf.Session(config=utils.get_config(), graph=self.graph)
    self.images, self.labels = self.dataset_numpy(train_filenames, parse_fn)
    self.ntrain = self.images.shape[0]
    with self.graph.as_default():
      self.test = default_parse(dataset(test_filenames), parse_fn)
    self.height = height
    self.width = width
    self.colors = colors
    self.nclass = nclass
    self.augment = augment

    self.all_indices = None  # all indices used here. None means using all data
    self.labeled_indices, self.unlabeled_indices = None, None
    self.no_label_indices = None

    self.mean = mean
    self.std = std
    self.p_labeled = p_labeled
    self.p_unlabeled = p_unlabeled

  def tf_get(self, index):
    def get(index):
      return self.images[index], self.labels[index]

    image, label = tf.py_func(get, [index], [tf.float32, tf.int64])
    return dict(image=image, label=label)

  def dataset_numpy(self, filenames, parse_fn=record_parse):
    dataset_tf = default_parse(dataset(filenames), parse_fn)
    data = []
    with self.graph.as_default():
      dataset_tf = dataset_tf.prefetch(16)
      it = dataset_tf.make_one_shot_iterator().get_next()
      try:
        while 1:
          data.append(self.session.run(it))
      except tf.errors.OutOfRangeError:
        pass
    images = np.stack([x['image'] for x in data])
    labels = np.stack([x.get('label', -1) for x in data])
    return images, labels

  def generate_labeled_and_unlabeled(self, labeled_indices):
    if len(labeled_indices) != len(set(labeled_indices)):
      raise ValueError('labeled_indices has duplication.')
    self.labeled_indices = np.array(labeled_indices)
    if self.all_indices is None:  # using all data
      raise ValueError('all_indices should not be None.')
      self.unlabeled_indices = np.array(list(frozenset(range(len(self.images))) - frozenset(labeled_indices)))
    else:
      self.unlabeled_indices = np.array(list(frozenset(self.all_indices) - frozenset(labeled_indices)))


  @classmethod
  def creator_small_data(cls, name, seed, train_size, labeled_size, augment,
                         colors=3, nclass=10, height=32, width=32,
                         parse_fn=record_parse,
                         sampc=False):
    """Create dataset w/ training being a subset
    train_size: size of training data.
    Will always shuffle data with the given seed, and pick the first train_size
    samples as training, the first labeled_size as the labeled.
    """
    if not isinstance(augment, list):
      augment = [augment] * 2  # Labeled, Unlabeled

    # Create an instance of cls
    def create():
      p_labeled = p_unlabeled = None

      if FLAGS.p_unlabeled:
        sequence = FLAGS.p_unlabeled.split(',')
        p_unlabeled = np.array(list(map(float, sequence)), dtype=np.float32)
        p_unlabeled /= np.max(p_unlabeled)

      graph = tf.Graph()
      with graph.as_default():
        if name != 'fashion_mnist':
          train_file = os.path.join(DATA_DIR, '%s-train.tfrecord' % name)
          test_file = os.path.join(DATA_DIR, '%s-test.tfrecord' % name)
        else:
          train_file = os.path.join(DATA_DIR, '%s-train.tfrecord*' % name)
          test_file = os.path.join(DATA_DIR, '%s-test.tfrecord*' % name)

        dataname = '%s.%d@%d_train%d%s' % (name, seed, labeled_size, train_size, '_sampc' if sampc else '')
        instance = cls(dataname,
                       graph, [train_file], [test_file], p_labeled=p_labeled, p_unlabeled=p_unlabeled,
                       augment=augment, parse_fn=parse_fn,
                       colors=colors, height=height, width=width, nclass=nclass)

        np.random.seed(seed)
        permu = np.random.permutation(len(instance.images))
        instance.images, instance.labels = instance.images[permu], instance.labels[permu]
        instance.all_indices = list(range(train_size))
        if not sampc:
          labeled_indices = list(range(labeled_size))
        else:
          dist = utils.get_class_dist(instance.labels, instance.nclass)
          cnts = [int(d * labeled_size) for d in dist]
          while sum(cnts) < labeled_size:
            cnts[0] += 1
          while sum(cnts) > labeled_size:
            if cnts[0] > 1:
              cnts[0] -= 1
            else:
              raise ValueError
          labeled_indices = []
          for c in range(instance.nclass):
            labeled_indices += list(np.where(instance.labels == c)[0][:cnts[c]])
        instance.generate_labeled_and_unlabeled(labeled_indices)

        return instance
    return '%s.%d@%d_train%d%s' % (name, seed, labeled_size, train_size, '_sampc' if sampc else ''), create



augment_stl10 = lambda x: dict(image=augment_shift(augment_mirror(x['image']), 12), label=x['label'])
augment_cifar10 = lambda x: dict(image=augment_shift(augment_mirror(x['image']), 4), label=x['label'])
augment_mnist = lambda x: dict(image=augment_shift(x['image'], 4), label=x['label'])
augment_svhn_extra = lambda x: dict(image=augment_shift(x['image'], 4), label=x['label'])
