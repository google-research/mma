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

import itertools
from absl import flags
import numpy as np
import tensorflow as tf
from libml import data
from libml.data import DynamicDataset, augment_cifar10, augment_svhn_extra

FLAGS = flags.FLAGS

flags.DEFINE_integer('nu', 2, 'Number of augmentations for class-consistency.')


def stack_augment(augment):
  def func(x):
    xl = [augment(x) for _ in range(FLAGS.nu)]

    return dict(image=tf.stack([x['image'] for x in xl]),
                label=tf.stack([x['label'] for x in xl]))

  return func


DATASETS = {}

# ------ cifar10 ------ #
DATASETS.update([DynamicDataset.creator_small_data('cifar10', seed, train, label,
                                                   [augment_cifar10, stack_augment(augment_cifar10)])
                 for seed, train, label in
                 itertools.product(range(10), range(5000, 50001, 5000), [250])])

# ------ cifar100 ------ #
DATASETS.update([DynamicDataset.creator_small_data('cifar100', seed, train, label,
                                                   [augment_cifar10, stack_augment(augment_cifar10)], nclass=100)
                 for seed, train, label in
                 itertools.product(range(10), range(5000, 50001, 5000), [2500])])

# ------ svhn ------ #
DATASETS.update([DynamicDataset.creator_small_data('svhn', seed, train, label,
                                                   [augment_svhn_extra, stack_augment(augment_svhn_extra)],
                                                   sampc=True)
                 for seed, train, label in
                 itertools.product([1,2,3,4,5], [5000, 10000, 20000, 50000, 73257], [250, 25000])])

# ------ svhn_extra ------ #
DATASETS.update([DynamicDataset.creator_small_data('svhn_extra', seed, train, label,
                                                   [augment_svhn_extra, stack_augment(augment_svhn_extra)],
                                                   sampc=True)
                 for seed, train, label in
                 itertools.product([1,2,3,4,5], [5000, 10000, 20000, 50000, 100000, 200000, 400000, 604388], [250])])

