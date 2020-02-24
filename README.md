# MMA - Combining MixMatch and Active Learning for Better Accuracy with Fewer Labels
Code for the paper: "[Combining MixMatch and Active Learning for Better Accuracy with Fewer Labels](https://arxiv.org/abs/1912.00594)" by Shuang Song, David Berthelot, and Afshin Rostamizadeh.

This is not an officially supported Google product.


## Setup

### Install dependencies

```bash
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
virtualenv -p python3 --system-site-packages env3
. env3/bin/activate
pip install -r requirements.txt
```

### Install datasets

```bash
export ML_DATA="path to where you want the datasets saved"
# Download datasets
CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py
```

## Running
We have hard-coded the parameters (batch for AL and number of iterations between each querying) used in the paper in [mixmatch_lineargrow.py](mixmatch_lineargrow.py). The parameters are documented and can be changed there.

To do the experiment on CIFAR-10 with diff as the uncertainty measurement on two augmentations of samples and no diversification method, i.e.,training mixmatch with 32 filters on CIFAR-10 shuffled with `seed=1`, starting from 250 randomly selected samples, querying 50 each time until 4000 labelled samples with diff.aug-direct:
```bash
CUDA_VISIBLE_DEVICES=0 python mixmatch_lineargrow.py --filters=32 --w_match=75 --beta=0.75 --dataset=cifar10.1@250_train50000 --grow_size=50 --grow_by=diff2.aug-direct
```

### Monitoring training progress
You can point tensorboard to the training folder (by default it is `--train_dir=./MMA_exp`) to monitor the training
process:
```bash
tensorboard.sh --port 6007 --logdir MMA_exp
```



## Citing this work
```
@misc{song2019combining,
      title={Combining MixMatch and Active Learning for Better Accuracy with Fewer Labels},
      author={Shuang Song and David Berthelot and Afshin Rostamizadeh},
      year={2019},
      eprint={1912.00594},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
