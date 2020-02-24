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

"""Custom neural network layers.

Low-level primitives such as custom convolution with custom initialization.
"""
import numpy as np
import tensorflow as tf

from libml.data import DataSet


def smart_shape(x):
  s, t = x.shape, tf.shape(x)
  return [t[i] if s[i].value is None else s[i] for i in range(len(s))]


def entropy_from_logits(logits):
  """Computes entropy from classifier logits.

  Args:
    logits: a tensor of shape (batch_size, class_count) representing the
    logits of a classifier.

  Returns:
    A tensor of shape (batch_size,) of floats giving the entropies
    batchwise.
  """
  distribution = tf.contrib.distributions.Categorical(logits=logits)
  return distribution.entropy()


def entropy_penalty(logits, entropy_penalty_multiplier, mask):
  """Computes an entropy penalty using the classifier logits.

  Args:
    logits: a tensor of shape (batch_size, class_count) representing the
      logits of a classifier.
    entropy_penalty_multiplier: A float by which the entropy is multiplied.
    mask: A tensor that optionally masks out some of the costs.

  Returns:
    The mean entropy penalty
  """
  entropy = entropy_from_logits(logits)
  losses = entropy * entropy_penalty_multiplier
  losses *= tf.cast(mask, tf.float32)
  return tf.reduce_mean(losses)


def kl_divergence_from_logits(logits_a, logits_b):
  """Gets KL divergence from logits parameterizing categorical distributions.

  Args:
    logits_a: A tensor of logits parameterizing the first distribution.
    logits_b: A tensor of logits parameterizing the second distribution.

  Returns:
    The (batch_size,) shaped tensor of KL divergences.
  """
  distribution1 = tf.contrib.distributions.Categorical(logits=logits_a)
  distribution2 = tf.contrib.distributions.Categorical(logits=logits_b)
  return tf.contrib.distributions.kl_divergence(distribution1, distribution2)


def mse_from_logits(output_logits, target_logits):
  """Computes MSE between predictions associated with logits.

  Args:
    output_logits: A tensor of logits from the primary model.
    target_logits: A tensor of logits from the secondary model.

  Returns:
    The mean MSE
  """
  diffs = tf.nn.softmax(output_logits) - tf.nn.softmax(target_logits)
  squared_diffs = tf.square(diffs)
  return tf.reduce_mean(squared_diffs, -1)


def interleave_offsets(batch, nu):
  groups = [batch // (nu + 1)] * (nu + 1)
  for x in range(batch - sum(groups)):
    groups[-x - 1] += 1
  offsets = [0]
  for g in groups:
    offsets.append(offsets[-1] + g)
  assert offsets[-1] == batch
  return offsets


def interleave(xy, batch):
  nu = len(xy) - 1
  offsets = interleave_offsets(batch, nu)
  xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
  for i in range(1, nu + 1):
    xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
  return [tf.concat(v, axis=0) for v in xy]


def logit_norm(v):
  return v * tf.rsqrt(tf.reduce_mean(tf.square(v)) + 1e-6)


def renorm(v):
  return v / tf.reduce_sum(v, axis=-1, keepdims=True)


def closed_form_uniform_argmax(logt, unsort_index, nclass):
  # Direct implementation from stackoverflow:
  # https://math.stackexchange.com/questions/158561/characterising-argmax-of-uniform-distributions
  p = [0]
  logt = logt.astype('d')
  qni = np.zeros(logt.shape[0], 'd')
  for i in range(1, nclass + 1):
    qi, qni = qni, np.exp((nclass - i) * logt[:, i - 1] - logt[:, i:].sum(1))
    p.append(p[-1] + (qni - qi) / (nclass - i + 1))
  p = np.array(p[1:], 'f').T
  return p[[[i] for i in range(logt.shape[0])], unsort_index]


def shakeshake(a, b, training):
  if not training:
    return 0.5 * (a + b)
  mu = tf.random_uniform([tf.shape(a)[0]] + [1] * (len(a.shape) - 1), 0, 1)
  mixf = a + mu * (b - a)
  mixb = a + mu[::1] * (b - a)
  return tf.stop_gradient(mixf - mixb) + mixb


class PMovingAverage:
  def __init__(self, name, nclass, buf_size):
    self.ma = tf.Variable(tf.ones([buf_size, nclass]) / nclass, trainable=False, name=name)

  def __call__(self):
    v = tf.reduce_mean(self.ma, axis=0)
    return v / tf.reduce_sum(v)

  def update(self, entry):
    entry = tf.reduce_mean(entry, axis=0)
    return tf.assign(self.ma, tf.concat([self.ma[1:], [entry]], axis=0))


class PData:
  def __init__(self, dataset):
    self.has_update = False
    if dataset.p_unlabeled is not None:
      self.p_data = tf.constant(dataset.p_unlabeled, name='p_data')
    elif dataset.p_labeled is not None:
      self.p_data = tf.constant(dataset.p_labeled, name='p_data')
    else:
      self.p_data = tf.Variable(renorm(tf.ones([dataset.nclass])), trainable=False, name='p_data')
      self.has_update = True

  def __call__(self):
    return self.p_data / tf.reduce_sum(self.p_data)

  def update(self, entry, decay=0.999):
    entry = tf.reduce_mean(entry, axis=0)
    return tf.assign(self.p_data, self.p_data * decay + entry * (1 - decay))
