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

"""Training setup."""
import json
import os.path
import shutil

import math
import numpy as np
import tensorflow as tf
from absl import flags
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
from tqdm import trange

from libml import data, utils

import sys
import time


FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', './MMA_exp/',
                    'Folder where to save training data.')
flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
flags.DEFINE_integer('batch', 64, 'Batch size.')
flags.DEFINE_integer('train_kimg', 1 << 14, 'Training duration in kibi-samples.')
flags.DEFINE_integer('report_kimg', 64, 'Report summary period in kibi-samples.')
flags.DEFINE_integer('save_kimg', 64, 'Save checkpoint period in kibi-samples.')

flags.DEFINE_integer('grow_size', 250, 'Grow the number of labeled by grow_size.')
flags.DEFINE_string('grow_by', 'max-direct', 'Grow by this measure.')

flags.DEFINE_integer('keep_ckpt', 50, 'Number of checkpoints to keep.')
flags.DEFINE_string('eval_ckpt', '', 'Checkpoint to evaluate. If provided, do not do training, just do eval.')


class Model:
  def __init__(self, train_dir, dataset, **kwargs):
    self.train_dir = os.path.join(train_dir, self.experiment_name(**kwargs))
    self.params = utils.EasyDict(kwargs)
    self.dataset = dataset
    self.session = None
    self.tmp = utils.EasyDict(print_queue=[], cache=utils.EasyDict())
    self.step = tf.train.get_or_create_global_step()
    self.ops = self.model(**kwargs)
    self.ops.update_step = tf.assign_add(self.step, FLAGS.batch)
    self.add_summaries(**kwargs)

    print(' Config '.center(80, '-'))
    print('train_dir', self.train_dir)
    print('%-32s %s' % ('Model', self.__class__.__name__))
    print('%-32s %s' % ('Dataset', dataset.name))
    for k, v in sorted(kwargs.items()):
      print('%-32s %s' % (k, v))
    print(' Model '.center(80, '-'))
    to_print = [tuple(['%s' % x for x in (v.name, np.prod(v.shape), v.shape)]) for v in utils.model_vars(None)]
    to_print.append(('Total', str(sum(int(x[1]) for x in to_print)), ''))
    sizes = [max([len(x[i]) for x in to_print]) for i in range(3)]
    fmt = '%%-%ds  %%%ds  %%%ds' % tuple(sizes)
    for x in to_print[:-1]:
      print(fmt % x)
    print()
    print(fmt % to_print[-1])
    print('-' * 80)
    self._create_initial_files()
    self.work_unit = None
    self.measurement = {}

  @property
  def arg_dir(self):
    return os.path.join(self.train_dir, 'args')

  @property
  def checkpoint_dir(self):
    return os.path.join(self.train_dir, 'tf')

  def train_print(self, text):
    self.tmp.print_queue.append(text)

  def _create_initial_files(self):
    for dir in (self.checkpoint_dir, self.arg_dir):
      if not tf.gfile.IsDirectory(dir):
        tf.gfile.MakeDirs(dir)
    self.save_args()

  def _reset_files(self):
    shutil.rmtree(self.train_dir)
    self._create_initial_files()

  def save_args(self, **extra_params):
    with tf.gfile.Open(os.path.join(self.arg_dir, 'args.json'), 'w') as f:
      json.dump({**self.params, **extra_params}, f, sort_keys=True, indent=4)

  @classmethod
  def load(cls, train_dir):
    with open(os.path.join(train_dir, 'args/args.json'), 'r') as f:
      params = json.load(f)
    instance = cls(train_dir=train_dir, **params)
    instance.train_dir = train_dir
    return instance

  def experiment_name_helper(self, exclude, **kwargs):
    args = []
    for x, y in sorted(kwargs.items()):
      if x in exclude:
        continue
      if y is None or x == 'nclass':
        continue
      if x in ['arch']:
        args.append(str(y))
      elif isinstance(y, bool):
        args += [x] if y else []
      else:
        args.append(x + str(y))
    return '_'.join([self.__class__.__name__] + args)

  def experiment_name(self, **kwargs):
    return self.experiment_name_helper(exclude=[], **kwargs)

  def eval_mode(self, ckpt=None):
    self.session = tf.Session(config=utils.get_config())
    saver = tf.train.Saver()
    if ckpt is None:
      ckpt = utils.find_latest_checkpoint(self.checkpoint_dir)
    else:
      ckpt = os.path.abspath(ckpt)
    saver.restore(self.session, ckpt)
    self.tmp.step = self.session.run(self.step)
    print('Eval model %s at global_step %d' % (self.__class__.__name__, self.tmp.step))
    return self

  def model(self, **kwargs):
    raise NotImplementedError()

  def add_summaries(self, **kwargs):
    raise NotImplementedError()


class ClassifySemi(Model):
  """Semi-supervised classification."""

  def __init__(self, train_dir, dataset, nclass, **kwargs):
    self.nclass = nclass
    self.max_labeled_size = dataset.images.shape[0]
    Model.__init__(self, train_dir, dataset, nclass=nclass, **kwargs)

  def train_step(self, train_session, data_labeled, data_unlabeled):
    x, y = self.dataset.session.run([data_labeled, data_unlabeled])
    self.tmp.step = train_session.run([self.ops.train_op, self.ops.update_step],
                                      feed_dict={self.ops.x: x['image'],
                                                 self.ops.y: y['image'],
                                                 self.ops.label: x['label']})[1]

  def train_for_contGrow(self, train_nimg, past_nimg, report_nimg,
                         grow_nimg, grow_size, max_labeled_size):
    """Function for training the model.

    Args:
      train_nimg: will train for train_nimg/batch iterations
      past_nimg: has previously trained for train_nimg/batch iterations
      report_nimg: report results every report_nimg samples
      grow_nimg: grow every grow_nimg samples
      grow_size: number of samples to query each time
      max_labeled_size: maximum labelling budget
    """
    if max_labeled_size == -1:
      max_labeled_size = self.dataset.labeled_indices.size + self.dataset.unlabeled_indices.size

    if grow_nimg > 0:
      print('grow_kimg:', grow_nimg >> 10)
      print('grow_by: ', FLAGS.grow_by)
      print('grow_size:', grow_size)
    else:
      grow_nimg = train_nimg
      print('Will not grow.')
    print('----')

    if FLAGS.eval_ckpt:
      accurices = self.eval_checkpoint(FLAGS.eval_ckpt)
      return
    batch = FLAGS.batch
    scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                      pad_step_number=10))
    with tf.train.MonitoredTrainingSession(
        scaffold=scaffold,
        checkpoint_dir=self.checkpoint_dir,
        config=utils.get_config(),
        save_checkpoint_steps=FLAGS.save_kimg << 10,
        save_summaries_steps=report_nimg - batch) as train_session:

      self.session = train_session._tf_sess()
      self.tmp.step = self.session.run(self.step)

      need_update = True
      while self.tmp.step < train_nimg:
        if grow_nimg > 0 and (self.tmp.step - past_nimg) % grow_nimg == 0:
          # Grow
          with self.dataset.graph.as_default():
            labeled_indices = utils.fixlen_to_idx(self.session.run(self.ops.label_index))
            self.dataset.generate_labeled_and_unlabeled(list(labeled_indices))
            # Get unlabeled data
            unlabeled_data = tf.data.Dataset.from_tensor_slices(self.dataset.unlabeled_indices) \
                .map(self.dataset.tf_get) \
                .map(self.dataset.augment[1]) \
                .batch(batch) \
                .prefetch(16) \
                .make_one_shot_iterator() \
                .get_next()  # not shuffled, not repeated
          need_update |= self.grow_labeled(FLAGS.grow_by, grow_size,
                                           max_labeled_size, unlabeled_data)
        if need_update:
          # If we need to update the labeled and unlabeled set to be used for training
          need_update = False
          labeled_indices = utils.fixlen_to_idx(self.session.run(self.ops.label_index))
          self.dataset.generate_labeled_and_unlabeled(list(labeled_indices))
          with self.dataset.graph.as_default():
            train_labeled = tf.data.Dataset.from_tensor_slices(self.dataset.labeled_indices) \
                .repeat() \
                .shuffle(FLAGS.shuffle) \
                .map(self.dataset.tf_get) \
                .map(self.dataset.augment[0]) \
                .batch(batch).prefetch(16)
            train_labeled = train_labeled.make_one_shot_iterator().get_next()
            train_unlabeled = tf.data.Dataset.from_tensor_slices(self.dataset.unlabeled_indices) \
                .repeat() \
                .shuffle(FLAGS.shuffle) \
                .map(self.dataset.tf_get) \
                .map(self.dataset.augment[1]) \
                .batch(batch) \
                .prefetch(16)
            train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()
          print('# of labeled/unlabeled samples to be used:',
                self.dataset.labeled_indices.size,
                self.dataset.unlabeled_indices.size)
        # The actual training
        loop = trange(self.tmp.step % report_nimg,
                      report_nimg,
                      batch,
                      leave=False,
                      unit='img',
                      unit_scale=batch,
                      desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg),
                                            train_nimg // report_nimg))
        for _ in loop:
          self.train_step(train_session, train_labeled, train_unlabeled)
          while self.tmp.print_queue:
            loop.write(self.tmp.print_queue.pop(0))
      while self.tmp.print_queue:
        print(self.tmp.print_queue.pop(0))


  def grow_labeled(self, grow_by_, grow_size, max_labeled_size, unlabeled_data):
    """Grow the labeled set.
    Args:
      grow_by_: spcifies the AL method used to grow. It consists of 3 parts:
                1). uncertainty measurement: max, std, entropy, diff2, w/
                    optionl suffix .aug denoting using 2 augmentations of samples
                2). diversification method: direct, kmeanprop, id (info density)
                3). for any diversification method, embd means calcualting
                    distance with embedding of the sample rather than the original
                Examples include diff2.aug-direct, max-kmeanprop-embd, random
                kmeanprop:  cluster all unlabeled images, pick low confidence
                            samples from each clusters where #_picked_from_cluster_i
                            is propotional to size_of_cluster_i
                id:         cosine is usually the best and beta=1
      grow_size: number of samples to query
      max_labeled_size: maximum labelling budget
      unlabeled_data: currently unlabeled data
    Return:
      Whether has grown with new labels or not.
    """
    if grow_size == 0:
      return False

    def parse_grow_by(grow_by_):
      """Parse string grow_by_. For example, max.aug-kmeanprop-embd gives
      grow_by = ['max', 'kmeanprop', 'embd'], grow_by_aug = True.
      """
      grow_by = grow_by_.split('-')
      grow_by += [''] * (3 - len(grow_by))
      # get uncertainty measurement and aug
      grow_by[0] = grow_by[0].split('.')[0]
      assert grow_by[0] in ['random', 'max', 'std', 'entropy', 'diff2']
      grow_by_aug = grow_by[0].endswith('.aug')

      # check if the option is valid
      if grow_by[0] != 'random':
        assert grow_by[1] in ['direct', 'kmeanprop',  'id']
      if grow_by[1] != 'direct':
        assert grow_by[2] in ['', 'embd']
      # std cannot be used with id
      if grow_by[1] == 'id':
        assert grow_by[0] != 'std'
      return grow_by, grow_by_aug

    grow_by, grow_by_aug = parse_grow_by(grow_by_)

    n_labeled = self.dataset.labeled_indices.size
    n_unlabeled = self.dataset.unlabeled_indices.size

    if max_labeled_size > 0 and n_labeled >= max_labeled_size:
      print('Currently have {} labeled. Max # ({}) reached. '
            'Do not grow.'.format(n_labeled, max_labeled_size))
      return False

    if max_labeled_size > 0 and max_labeled_size - n_labeled < grow_size:
      grow_size = max_labeled_size - n_labeled
      print('Labeling budget is adjusted to {}'.format(grow_size))

    if n_unlabeled <= grow_size:
      new_labeled_indices = self.dataset.unlabeled_indices
      print('Not enough unlabeled samples left, will use all those left.')
    elif grow_by[0] == 'random':  # randomly select
      new_labeled_indices = np.random.choice(self.dataset.unlabeled_indices,
                                             grow_size,
                                             replace=False)
    else:
      def diff2(p):
        """Difference between the top 2 (highest - 2nd highest)."""
        psorted = np.sort(p)
        return psorted[-1] - psorted[-2]

      measure2func = {'max': lambda p: np.max(p),
                      'std': lambda p: np.std(p),
                      'entropy': lambda p: -entropy(p),  # negation, smaller the better
                      'diff2': lambda p: diff2(p),
                      }

      unlabeled_images = self.dataset.images[self.dataset.unlabeled_indices]

      # Get prediction and confidence
      if not grow_by_aug:
        # If not using augmentation, get prediction of the original samples
        predictions = np.concatenate(
            [self.session.run(self.ops.classify_op,
                              feed_dict={self.ops.x: unlabeled_images[x:x + FLAGS.batch]})
             for x in range(0, n_unlabeled, FLAGS.batch)],
            axis=0)
      else:
        # If using augmentation, get predictions of two augmentations of each sample
        predictions0, predictions1 = [], []
        # Get predictions batch by batch
        for i in range(int(math.ceil(n_unlabeled / FLAGS.batch))):
          unlabeled_images_aug = self.dataset.session.run(unlabeled_data)
          unlabeled_images_aug = unlabeled_images_aug['image']
          unlabeled_images0 = unlabeled_images_aug[:, 0]
          unlabeled_images1 = unlabeled_images_aug[:, 1]
          # Predict
          predictions0.append(self.session.run(self.ops.classify_op,
                                               feed_dict={self.ops.x: unlabeled_images0}))
          predictions1.append(self.session.run(self.ops.classify_op,
                                               feed_dict={self.ops.x: unlabeled_images1}))

        # Concatenate list of np.array into one np.array
        predictions0 = np.concatenate(predictions0, axis=0)
        predictions1 = np.concatenate(predictions1, axis=0)
        # Average the two predictions
        predictions = (predictions0 + predictions1) / 2.0
      # Measure the confidence of each sample
      confidences = np.array([measure2func[grow_by[0]](p) for p in predictions])

      # If "direct", take directly the least confident
      if grow_by[1] == 'direct':
        less_confident_idx = np.argpartition(confidences, grow_size)[:grow_size]
        new_labeled_indices = self.dataset.unlabeled_indices[less_confident_idx]
      # For kmeanprop and id, cluster/measure similarity of the whole unlabeled set
      elif grow_by[1] in ['kmeanprop', 'id']:
        if grow_by[2] == 'embd':  # get embedding of data
          unlabeled_images = np.concatenate([self.session.run(self.ops.embedding_op,
                                                              feed_dict={self.ops.x: unlabeled_images[x:x + FLAGS.batch]})
                                             for x in range(0, unlabeled_images.shape[0], FLAGS.batch)],
                                            axis=0)
          unlabeled_images = unlabeled_images.reshape(unlabeled_images.shape[0], -1)
        if grow_by[1] == 'kmeanprop':
          # Perform k-means with 20 clusters and pick from each cluster those w/
          # lowest confidence.
          selected_idx = utils.get_low_confidence_from_each_clusters(unlabeled_images,
                                                                     20,
                                                                     grow_size,
                                                                     confidences)
          new_labeled_indices = self.dataset.unlabeled_indices[selected_idx]
        elif grow_by[1] == 'id':
          # Compute pairwise distance
          avg_dists = pairwise_distances(unlabeled_images, metric='cosine').mean(axis=1)
          # Compute uncertainty measurement and get final measurement
          if grow_by[0] in ['max', 'diff2']:
            info_measure = (1 - confidences) * avg_dists
          elif grow_by[0] == 'entropy':
            info_measure = -confidences * avg_dists
          else:
            raise ValueError
          selected_idx = np.argpartition(info_measure, -grow_size)[-grow_size:]
          new_labeled_indices = self.dataset.unlabeled_indices[selected_idx]

    # update labeled_indices
    combined_labeled_indices = list(self.dataset.labeled_indices) + list(new_labeled_indices)
    self.dataset.generate_labeled_and_unlabeled(combined_labeled_indices)
    self.session.run(self.ops.update_label_index,
                     feed_dict={self.ops.label_index_input: \
                                    utils.idx_to_fixlen(self.dataset.labeled_indices,
                                                        self.dataset.ntrain)})
    print('Now have #labeled/unlabeled: {} {}'.format(grow_by_,
                                                      self.dataset.labeled_indices.size,
                                                      self.dataset.unlabeled_indices.size))
    return True  # labeled data added


  def eval_checkpoint(self, ckpt=None):
    self.eval_mode(ckpt)
    accuracies = self.eval_stats()
    print('kimg %-5d  accuracy labeled/unlabeled/test  %.2f %.2f %.2f' % tuple([self.tmp.step >> 10] + accuracies))
    with tf.gfile.Open(ckpt + '_res.json', 'w') as f:
      output = {'ckpt': self.tmp.step,
                'labeled': accuracies[0],
                'unlabeled': accuracies[1],
                'test': accuracies[3]}
      json.dump(f, output)
    return accuracies

  def eval_stats(self, batch=None, feed_extra=None, classify_op=None):
    def collect_samples(data):
      data_it = data.batch(1).prefetch(16).make_one_shot_iterator().get_next()

      images, labels = [], []
      while 1:
        try:
          v = self.dataset.session.run(data_it)
        except tf.errors.OutOfRangeError:
          break
        images.append(v['image'])
        labels.append(v['label'])

      images = np.concatenate(images, axis=0)
      labels = np.concatenate(labels, axis=0)
      return images, labels

    if 'test' not in self.tmp.cache:
      with self.dataset.graph.as_default():
        self.tmp.cache.test = collect_samples(self.dataset.test)
        self.tmp.cache.train_labeled = collect_samples(
            tf.data.Dataset.from_tensor_slices(self.dataset.labeled_indices).map(self.dataset.tf_get)
            )
        if self.dataset.unlabeled_indices.size > 0:
          all_unlabeled_indices = self.dataset.unlabeled_indices
          if self.dataset.no_label_indices is not None:
            all_unlabeled_indices = np.concatenate((all_unlabeled_indices, self.dataset.no_label_indices))
          self.tmp.cache.train_unlabeled = collect_samples(
              tf.data.Dataset.from_tensor_slices(all_unlabeled_indices).map(self.dataset.tf_get)
              )

    batch = batch or FLAGS.batch
    classify_op = self.ops.classify_op if classify_op is None else classify_op
    accuracies = []
    for subset in ['train_labeled', 'train_unlabeled', 'test']:
      if subset not in self.tmp.cache:
        accuracies.append(-1)
        continue
      images, labels = self.tmp.cache[subset]
      predicted = np.concatenate([
        self.session.run(classify_op, feed_dict={
          self.ops.x: images[x:x + batch], **(feed_extra or {})})
        for x in range(0, images.shape[0], batch)
      ], axis=0)
      accuracies.append((predicted.argmax(1) == labels).mean() * 100)
    self.train_print('kimg %-5d  accuracy labeled/unlabeled/test  %.2f %.2f %.2f' % tuple([self.tmp.step >> 10] + accuracies))
    return np.array(accuracies, 'f')

  def add_summaries(self, feed_extra=None, **kwargs):
    del kwargs

    def gen_stats():
      return self.eval_stats(feed_extra=feed_extra)

    accuracies = tf.py_func(gen_stats, [], tf.float32)
    tf.summary.scalar('accuracy/train_labeled', accuracies[0])
    tf.summary.scalar('accuracy/train_unlabled', accuracies[1])
    tf.summary.scalar('accuracy', accuracies[2])


class ClassifyFully(ClassifySemi):
  """Fully-supervised classification."""
  def train_step(self, train_session, data_labeled, data_unlabeled):
    del data_unlabeled
    x = self.dataset.session.run(data_labeled)
    self.tmp.step = train_session.run([self.ops.train_op, self.ops.update_step],
                                      feed_dict={self.ops.x: x['image'],
                                                 self.ops.label: x['label']})[1]

  def tune(self, train_nimg):
    batch = FLAGS.batch
    with self.graph.as_default():
      train_labeled = tf.data.Dataset.from_tensor_slices(self.dataset.labeled_indices).map(self.dataset.tf_get).batch(batch).prefetch(16)
      train_labeled = train_labeled.make_one_shot_iterator().get_next()

      for _ in trange(0, train_nimg, batch, leave=False, unit='img', unit_scale=batch, desc='Tuning'):
        x = self.dataset.session.run([train_labeled])
        self.session.run([self.ops.tune_op], feed_dict={self.ops.x: x['image'],
                                                        self.ops.label: x['label']})
