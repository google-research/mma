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

import collections
import os, sys

from absl import app
from absl import flags

import numpy as np

from libml import utils
from libml.data_pair import DATASETS
from mixmatch import MixMatch
from mixmatch import MixMode

FLAGS = flags.FLAGS

# For each dataset, GROW_KIMG_DIC maps a grow_size to grow_kimg where grow_size
# is the number of samples to query each time and grow_kimg means new samples
# are queried after (grow_kimg << 10) / batch iterations.
GROW_KIMG_DIC_cifar10 = {25: 1024,
                         50: 2048,
                         125: 5120,
                         250: 10240}
GROW_KIMG_DIC_cifar100 = {500: 2048,
                          100: 1024}
GROW_KIMG_DIC_svhn = {50: 512}
GROW_KIMG_DIC_svhn_extra = {50: 512}

# For each dataset,
# START is the size of the randomly selected samples to start with
# END is the total labelling budget
# INITIAL_KIMG means we will train the model for (INITIAL_KIMG << 10) / batch
# iterations with the randomly selected initial set
START_cifar10, START_cifar100, START_svhn, START_svhn_extra = 250, 2500, 250, 250
END_cifar10, END_cifar100, END_svhn, END_svhn_extra = 4000, 10000, 4000, 4000
INITIAL_KIMG_cifar10, INITIAL_KIMG_cifar100 = 16384, 16384
INITIAL_KIMG_svhn, INITIAL_KIMG_svhn_extra = 8192, 8192

DICT = {'cifar10': (GROW_KIMG_DIC_cifar10, START_cifar10, END_cifar10, INITIAL_KIMG_cifar10),
        'cifar100': (GROW_KIMG_DIC_cifar100, START_cifar100, END_cifar100, INITIAL_KIMG_cifar100),
        'svhn': (GROW_KIMG_DIC_svhn, START_svhn, END_svhn, INITIAL_KIMG_svhn),
        'svhn_extra': (GROW_KIMG_DIC_svhn_extra, START_svhn_extra, END_svhn_extra, INITIAL_KIMG_svhn_extra),
       }

Grow = collections.namedtuple('Grow', 'grow_size grow_kimg grow_to')


class MixMatch_LinearGrow(MixMatch):

  def train_lineargrow(self, report_nimg):
    GROW_KIMG_DIC, START_n, END_n, INITIAL_KIMG = DICT[FLAGS.dataset.split('.')[0]]

    grow_size = FLAGS.grow_size
    grow_kimg = GROW_KIMG_DIC[grow_size]

    GROW = (Grow(0, INITIAL_KIMG, START_n),
            Grow(grow_size, grow_kimg, END_n))
    train_kimgs = [GROW[0].grow_kimg, (END_n - START_n) // grow_size * grow_kimg]
    train_kimgs = list(np.cumsum(train_kimgs))
    past_kimgs = [0] + train_kimgs[:-1]
    for g, train_kimg, past_kimg in zip(GROW, train_kimgs, past_kimgs):
      print('******* Phase {} *******'.format(g))
      self.train_for_contGrow(train_kimg << 10, past_kimg << 10, report_nimg,
                              g.grow_kimg << 10, g.grow_size, g.grow_to)
      print('******* Phase {} finished. *******'.format(g))


def main(argv):
  del argv  # Unused.
  assert FLAGS.dataset.split('.')[0] in ['cifar10', 'cifar100', 'svhn', 'svhn_extra']
  dataset = DATASETS[FLAGS.dataset]()
  log_width = utils.ilog2(dataset.width)
  model = MixMatch_LinearGrow(
      os.path.join(FLAGS.train_dir,
                   dataset.name.split('@')[0] + '_train' + \
                   dataset.name.split('train')[-1] + '_Grow'),
      dataset,
      lr=FLAGS.lr,
      wd=FLAGS.wd,
      arch=FLAGS.arch,
      batch=FLAGS.batch,
      nclass=dataset.nclass,
      ema=FLAGS.ema,
      beta=FLAGS.beta,
      logit_norm=FLAGS.logit_norm,
      T=FLAGS.T,
      mixmode=FLAGS.mixmode,
      nu=FLAGS.nu,
      dbuf=FLAGS.dbuf,
      w_match=FLAGS.w_match,
      warmup_kimg=FLAGS.warmup_kimg,
      scales=FLAGS.scales or (log_width - 2),
      filters=FLAGS.filters,
      repeat=FLAGS.repeat,
      growby=FLAGS.grow_by,
      growsize=FLAGS.grow_size)
  model.train_lineargrow(FLAGS.report_kimg << 10)


if __name__ == '__main__':
  utils.setup_tf()
  flags.DEFINE_float('wd', 0.02, 'Weight decay.')
  flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
  flags.DEFINE_float('beta', 0.75, 'Mixup beta distribution.')
  flags.DEFINE_bool('logit_norm', False, 'Whether to use logit normalization.')

  flags.DEFINE_float('T', 0.5, 'Softmax sharpening temperature.')
  flags.DEFINE_enum('mixmode', 'xxy.yxy', MixMode.MODES, 'Mixup mode')
  flags.DEFINE_integer(
      'dbuf', 128, 'Distribution buffer size to estimate p_model.')
  flags.DEFINE_float('w_match', 100, 'Weight for distribution matching loss.')
  flags.DEFINE_integer(
      'warmup_kimg', 128, 'Warmup in kimg for the matching loss.')
  flags.DEFINE_integer(
      'scales', 0, 'Number of 2x2 downscalings in the classifier.')
  flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
  flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')

  FLAGS.set_default('dataset', 'cifar10')
  FLAGS.set_default('batch', 64)
  FLAGS.set_default('lr', 0.002)
  FLAGS.set_default('train_kimg', 1 << 16)
  app.run(main)
