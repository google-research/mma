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

"""MixMatch training.
- Ensure class consistency by producing a group of `nu` augmentations of the same image and guessing the label for the
  group.
- Sharpen the target distribution.
- Use the sharpened distribution directly as a smooth label in MixUp.
"""

import functools
import os, sys

from absl import app
from absl import flags
import tensorflow as tf

from libml import layers, utils
from libml.data_pair import DATASETS
from libml.models import MultiModel

FLAGS = flags.FLAGS


class MixMode:
  # x = labelled example
  # y = unlabelled example
  # xx.yxy = mix x with x, mix y with both x and y.
  MODES = 'xx.yy xxy.yxy xx.yxy xx.yx'.split()

  def __init__(self, mode):
    assert mode in self.MODES
    self.mode = mode

  @staticmethod
  def augment_pair(x0, l0, x1, l1, beta, **kwargs):
    del kwargs
    if beta > 0:
      mix = tf.distributions.Beta(beta, beta).sample([tf.shape(x0)[0], 1, 1, 1])
      mix = tf.maximum(mix, 1 - mix)
    else:
      mix = tf.ones([tf.shape(x0)[0], 1, 1, 1])
    index = tf.random_shuffle(tf.range(tf.shape(x0)[0]))
    xs = tf.gather(x1, index)
    ls = tf.gather(l1, index)
    xmix = x0 * mix + xs * (1 - mix)
    lmix = l0 * mix[:, :, 0, 0] + ls * (1 - mix[:, :, 0, 0])
    return xmix, lmix

  @staticmethod
  def augment(x, l, beta, **kwargs):
    return MixMode.augment_pair(x, l, x, l, beta, **kwargs)

  def __call__(self, xl, ll, betal):
    assert len(xl) == len(ll) >= 2
    assert len(betal) == 2
    if self.mode == 'xx.yy':
      mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
      mx1, ml1 = self.augment(tf.concat(xl[1:], 0), tf.concat(ll[1:], 0), betal[1])
      return [mx0] + tf.split(mx1, len(xl) - 1), [ml0] + tf.split(ml1, len(ll) - 1)
    elif self.mode == 'xxy.yxy':
      mx, ml = self.augment(tf.concat(xl, 0), tf.concat(ll, 0), sum(betal) / len(betal))
      return tf.split(mx, len(xl)), tf.split(ml, len(ll))
    elif self.mode == 'xx.yxy':
      mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
      mx1, ml1 = self.augment(tf.concat(xl, 0), tf.concat(ll, 0), betal[1])
      mx1, ml1 = [tf.split(m, len(xl))[1:] for m in (mx1, ml1)]
      return [mx0] + mx1, [ml0] + ml1
    elif self.mode == 'xx.yx':
      mx0, ml0 = self.augment(xl[0], ll[0], betal[0])
      mx1, ml1 = zip(*[self.augment_pair(xl[i], ll[i], xl[0], ll[0], betal[1])
               for i in range(1, len(xl))])
      return (mx0,) + mx1, (ml0,) + ml1
    raise NotImplementedError(self.mode)


class MixMatch(MultiModel):
  def distribution_summary(self, p_data, p_model, p_target=None):
    def kl(p, q):
      p /= tf.reduce_sum(p)
      q /= tf.reduce_sum(q)
      return -tf.reduce_sum(p * tf.log(q / p))

    tf.summary.scalar('metrics/kld', kl(p_data, p_model))
    if p_target is not None:
      tf.summary.scalar('metrics/kld_target', kl(p_data, p_target))

    for i in range(self.nclass):
      tf.summary.scalar('matching/class%d_ratio' % i, p_model[i] / p_data[i])
    for i in range(self.nclass):
      tf.summary.scalar('matching/val%d' % i, p_model[i])

  def augment(self, x, l, beta, **kwargs):
    assert 0, 'Do not call.'

  def guess_label(self, y, classifier, p_data, p_model, T, **kwargs):
    del kwargs
    logits_y = [classifier(yi, training=True) for yi in y]
    logits_y = tf.concat(logits_y, 0)
    # Compute predicted probability distribution py.
    p_model_y = tf.reshape(tf.nn.softmax(logits_y), [len(y), -1, self.nclass])
    p_model_y = tf.reduce_mean(p_model_y, axis=0)
    # Compute the target distribution.
    p_target = tf.pow(p_model_y, 1. / T)
    p_target /= tf.reduce_sum(p_target, axis=1, keep_dims=True)
    return utils.EasyDict(p_target=p_target, p_model=p_model_y)

  def model(self, nu, w_match, warmup_kimg, batch, lr, wd, ema, dbuf, beta, mixmode, logit_norm, **kwargs):

    def classifier(x, logit_norm=logit_norm, **kw):
      v = self.classifier(x, **kw, **kwargs)[0]
      if not logit_norm:
        return v
      return v * tf.rsqrt(tf.reduce_mean(tf.square(v)) + 1e-8)

    def embedding(x, **kw):
      return self.classifier(x, **kw, **kwargs)[1]

    label_index = tf.Variable(utils.idx_to_fixlen(self.dataset.labeled_indices, self.dataset.ntrain),
                  trainable=False, name='label_index', dtype=tf.int32)
    label_index_input = tf.placeholder(tf.int32, self.dataset.ntrain, 'label_index_input')
    update_label_index = tf.assign(label_index, label_index_input)

    hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
    x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
    y_in = tf.placeholder(tf.float32, [None, nu] + hwc, 'y')
    l_in = tf.placeholder(tf.int32, [None], 'labels')
    wd *= lr
    w_match *= tf.clip_by_value(tf.cast(self.step, tf.float32) / (warmup_kimg << 10), 0, 1)
    augment = MixMode(mixmode)

    # Moving average of the current estimated label distribution
    p_model = layers.PMovingAverage('p_model', self.nclass, dbuf)
    p_target = layers.PMovingAverage('p_target', self.nclass, dbuf)  # Rectified distribution (only for plotting)

    # Known (or inferred) true unlabeled distribution
    p_data = layers.PData(self.dataset)

    y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
    guess = self.guess_label(tf.split(y, nu), classifier, p_data(), p_model(), **kwargs)
    ly = tf.stop_gradient(guess.p_target)
    lx = tf.one_hot(l_in, self.nclass)
    xy, labels_xy = augment([x_in] + tf.split(y, nu), [lx] + [ly] * nu, [beta, beta])
    x, y = xy[0], xy[1:]
    labels_x, labels_y = labels_xy[0], tf.concat(labels_xy[1:], 0)
    del xy, labels_xy

    batches = layers.interleave([x] + y, batch)
    skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    logits = [classifier(batches[0], training=True)]
    post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
    for batchi in batches[1:]:
      logits.append(classifier(batchi, training=True))
    logits = layers.interleave(logits, batch)
    logits_x = logits[0]
    logits_y = tf.concat(logits[1:], 0)

    loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x)
    loss_xe = tf.reduce_mean(loss_xe)
    loss_l2u = tf.square(labels_y - tf.nn.softmax(logits_y))
    loss_l2u = tf.reduce_mean(loss_l2u)
    tf.summary.scalar('losses/xe', loss_xe)
    tf.summary.scalar('losses/l2u', loss_l2u)
    self.distribution_summary(p_data(), p_model(), p_target())

    ema = tf.train.ExponentialMovingAverage(decay=ema)
    ema_op = ema.apply(utils.model_vars())
    ema_getter = functools.partial(utils.getter_ema, ema)
    post_ops.extend([ema_op,
             p_model.update(guess.p_model),
             p_target.update(guess.p_target)])
    if p_data.has_update:
      post_ops.append(p_data.update(lx))
    post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

    train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe + w_match * loss_l2u, colocate_gradients_with_ops=True)
    with tf.control_dependencies([train_op]):
      train_op = tf.group(*post_ops)

    # Tuning op: only retrain batch norm.
    skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    classifier(batches[0], training=True)
    train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if v not in skip_ops])

    return utils.EasyDict(
      x=x_in, y=y_in, label=l_in, train_op=train_op, tune_op=train_bn,
      classify_raw=tf.nn.softmax(classifier(x_in, logit_norm=False, training=False)),  # No EMA, for debugging.
      classify_op=tf.nn.softmax(classifier(x_in, logit_norm=False, getter=ema_getter, training=False)),
      embedding_op=embedding(x_in, training=False),
      label_index=label_index, update_label_index=update_label_index, label_index_input=label_index_input,
      )


