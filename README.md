# [WIP]textcnn-conv-deconv-pytorch
Text convolution-deconvolution auto-encoder and classification model in PyTorch.  
PyTorch implementation of [Deconvolutional Paragraph Representation Learning](https://arxiv.org/abs/1708.04729v3) described in NIPS 2017.  
**This repository is still developing.**

## Requirement
- Python 3
- PyTorch >= 0.3
- numpy

## Usage
### Train
#### Paragraph reconstruction
Download data. [Hotel reviews](https://drive.google.com/file/d/0B52eYWrYWqIpQzhBNkVxaV9mMjQ/view)  
Then, run following command.

```shell
$ python main_reconstruction.py -data_path=path/to/hotel_reviews.p 
```

Specify download data path by `-data_path`.

About other parameters.

```
usage: main_reconstruction.py [-h] [-lr LR] [-epochs EPOCHS]
                              [-batch_size BATCH_SIZE]
                              [-lr_decay_interval LR_DECAY_INTERVAL]
                              [-log_interval LOG_INTERVAL]
                              [-test_interval TEST_INTERVAL]
                              [-save_interval SAVE_INTERVAL]
                              [-save_dir SAVE_DIR] [-data_path DATA_PATH]
                              [-shuffle SHUFFLE] [-sentence_len SENTENCE_LEN]
                              [-embed_dim EMBED_DIM]
                              [-kernel_sizes KERNEL_SIZES] [-tau TAU]
                              [-use_cuda] [-enc_snapshot ENC_SNAPSHOT]
                              [-dec_snapshot DEC_SNAPSHOT]

text convolution-deconvolution auto-encoder model

optional arguments:
  -h, --help            show this help message and exit
  -lr LR                initial learning rate
  -epochs EPOCHS        number of epochs for train
  -batch_size BATCH_SIZE
                        batch size for training
  -lr_decay_interval LR_DECAY_INTERVAL
                        how many epochs to wait before decrease learning rate
  -log_interval LOG_INTERVAL
                        how many steps to wait before logging training status
  -test_interval TEST_INTERVAL
                        how many epochs to wait before testing
  -save_interval SAVE_INTERVAL
                        how many epochs to wait before saving
  -save_dir SAVE_DIR    where to save the snapshot
  -data_path DATA_PATH  data path
  -shuffle SHUFFLE      shuffle data every epoch
  -sentence_len SENTENCE_LEN
                        how many tokens in a sentence
  -embed_dim EMBED_DIM  number of embedding dimension
  -kernel_sizes KERNEL_SIZES
                        kernel size to use for convolution
  -tau TAU              temperature parameter
  -use_cuda             whether using cuda
  -enc_snapshot ENC_SNAPSHOT
                        filename of encoder snapshot
  -dec_snapshot DEC_SNAPSHOT
                        filename of decoder snapshot
```

#### Semi-supervised sequence classification
Run follow command.  

```shell
$ python main.py -data_path=path/to/trainingdata -label_path=path/to/labeldata
```

Specify training data and label data by `-data_path` and `-label_data` arguments.  
Both data must have same lines and training data must be separated by blank.  

About other parameters.  

```
usage: main_classification.py [-h] [-lr LR] [-epochs EPOCHS]
                              [-batch_size BATCH_SIZE]
                              [-lr_decay_interval LR_DECAY_INTERVAL]
                              [-log_interval LOG_INTERVAL]
                              [-test_interval TEST_INTERVAL]
                              [-save_interval SAVE_INTERVAL]
                              [-save_dir SAVE_DIR] [-data_path DATA_PATH]
                              [-label_path LABEL_PATH] [-separated SEPARATED]
                              [-shuffle SHUFFLE] [-sentence_len SENTENCE_LEN]
                              [-mlp_out MLP_OUT] [-dropout DROPOUT]
                              [-embed_dim EMBED_DIM]
                              [-kernel_sizes KERNEL_SIZES] [-tau TAU]
                              [-use_cuda] [-enc_snapshot ENC_SNAPSHOT]
                              [-dec_snapshot DEC_SNAPSHOT]
                              [-mlp_snapshot MLP_SNAPSHOT]

text convolution-deconvolution auto-encoder model

optional arguments:
  -h, --help            show this help message and exit
  -lr LR                initial learning rate
  -epochs EPOCHS        number of epochs for train
  -batch_size BATCH_SIZE
                        batch size for training
  -lr_decay_interval LR_DECAY_INTERVAL
                        how many epochs to wait before decrease learning rate
  -log_interval LOG_INTERVAL
                        how many steps to wait before logging training status
  -test_interval TEST_INTERVAL
                        how many steps to wait before testing
  -save_interval SAVE_INTERVAL
                        how many epochs to wait before saving
  -save_dir SAVE_DIR    where to save the snapshot
  -data_path DATA_PATH  data path
  -label_path LABEL_PATH
                        label path
  -separated SEPARATED  how separated text data is
  -shuffle SHUFFLE      shuffle the data every epoch
  -sentence_len SENTENCE_LEN
                        how many tokens in a sentence
  -mlp_out MLP_OUT      number of classes
  -dropout DROPOUT      the probability for dropout
  -embed_dim EMBED_DIM  number of embedding dimension
  -kernel_sizes KERNEL_SIZES
                        kernel size to use for convolution
  -tau TAU              temperature parameter
  -use_cuda             whether using cuda
  -enc_snapshot ENC_SNAPSHOT
                        filename of encoder snapshot
  -dec_snapshot DEC_SNAPSHOT
                        filename of decoder snapshot
  -mlp_snapshot MLP_SNAPSHOT
                        filename of mlp classifier snapshot
```

## Reference
[Deconvolutional Paragraph Representation Learning](https://arxiv.org/abs/1708.04729v3)  
Yizhe Zhang, Dinghan Shen, Guoyin Wang, Zhe Gan, Ricardo Henao, Lawrence Carin  
arXiv:1708.04729 [cs.CL]
