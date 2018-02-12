import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model
from datasets import TextClassificationDataset, ToTensor
from train import train_classification

import argparse


def main():
    parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-epochs', type=int, default=60, help='number of epochs for train')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('-lr_decay_interval', type=int, default=20,
                        help='how many epochs to wait before decrease learning rate')
    parser.add_argument('-log_interval', type=int, default=16,
                        help='how many steps to wait before logging training status')
    parser.add_argument('-test_interval', type=int, default=100,
                        help='how many steps to wait before testing')
    parser.add_argument('-save_interval', type=int, default=5,
                        help='how many epochs to wait before saving')
    parser.add_argument('-save_dir', type=str, default='snapshot', help='where to save the snapshot')
    # data
    parser.add_argument('-data_path', type=str, help='data path')
    parser.add_argument('-label_path', type=str, help='label path')
    parser.add_argument('-separated', type=str, default='sentencepiece', help='how separated text data is')
    parser.add_argument('-shuffle', default=False, help='shuffle the data every epoch')
    parser.add_argument('-sentence_len', type=int, default=60, help='how many tokens in a sentence')
    # model
    parser.add_argument('-mlp_out', type=int, default=7, help='number of classes')
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout')
    parser.add_argument('-embed_dim', type=int, default=300, help='number of embedding dimension')
    parser.add_argument('-kernel_sizes', type=int, default=2,
                        help='kernel size to use for convolution')
    parser.add_argument('-tau', type=float, default=0.01, help='temperature parameter')
    parser.add_argument('-use_cuda', action='store_true', default=True, help='whether using cuda')
    # option
    parser.add_argument('-enc_snapshot', type=str, default=None, help='filename of encoder snapshot ')
    parser.add_argument('-dec_snapshot', type=str, default=None, help='filename of decoder snapshot ')
    parser.add_argument('-mlp_snapshot', type=str, default=None, help='filename of mlp classifier snapshot ')
    args = parser.parse_args()

    dataset = TextClassificationDataset(args.data_path,
                                        args.label_path,
                                        args.separated,
                                        sentence_len=args.sentence_len,
                                        transoform=ToTensor())
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    print("Vocab number")
    print(dataset.vocab_length())

    k = args.embed_dim
    v = dataset.vocab_length()
    if args.enc_snapshot is None or args.dec_snapshot is None or args.mlp_snapshot is None:
        print("Start from initial")
        embedding = nn.Embedding(v, k, max_norm=1.0, norm_type=2.0)

        encoder = model.ConvolutionEncoder(embedding)
        decoder = model.DeconvolutionDecoder(embedding, args.tau)
        mlp = model.MLPClassifier(args.mlp_out, args.dropout)
    else:
        print("Restart from snapshot")
        encoder = torch.load(args.enc_snapshot)
        decoder = torch.load(args.dec_snapshot)
        mlp = torch.load(args.mlp_snapshot)

    train_classification(data_loader, data_loader, encoder, decoder, mlp, args)

if __name__ == '__main__':
    main()