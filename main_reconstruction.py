import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model
from datasets import TextClassificationDataset, ToTensor, load_hotel_review_data
from train import train_reconstruction

import argparse


def main():
    parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('-lr_decay_interval', type=int, default=4,
                        help='how many epochs to wait before decrease learning rate')
    parser.add_argument('-log_interval', type=int, default=256,
                        help='how many steps to wait before logging training status')
    parser.add_argument('-test_interval', type=int, default=10,
                        help='how many epochs to wait before testing')
    parser.add_argument('-save_interval', type=int, default=2,
                        help='how many epochs to wait before saving')
    parser.add_argument('-save_dir', type=str, default='rec_snapshot', help='where to save the snapshot')
    # data
    parser.add_argument('-data_path', type=str, help='data path')
    parser.add_argument('-shuffle', default=False, help='shuffle data every epoch')
    parser.add_argument('-sentence_len', type=int, default=210, help='how many tokens in a sentence')
    # model
    parser.add_argument('-embed_dim', type=int, default=300, help='number of embedding dimension')
    parser.add_argument('-kernel_sizes', type=int, default=2,
                        help='kernel size to use for convolution')
    parser.add_argument('-tau', type=float, default=0.01, help='temperature parameter')
    parser.add_argument('-use_cuda', action='store_true', default=True, help='whether using cuda')
    # option
    parser.add_argument('-enc_snapshot', type=str, default=None, help='filename of encoder snapshot ')
    parser.add_argument('-dec_snapshot', type=str, default=None, help='filename of decoder snapshot ')
    args = parser.parse_args()

    train_data, test_data = load_hotel_review_data(args.data_path, args.sentence_len)
    train_loader, test_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle),\
                                  DataLoader(test_data, batch_size=args.batch_size, shuffle=args.shuffle)

    k = args.embed_dim
    v = train_data.vocab_lennght()
    if args.enc_snapshot is None or args.dec_snapshot is None or args.mlp_snapshot is None:
        print("Start from initial")
        embedding = nn.Embedding(v, k, max_norm=1.0, norm_type=2.0)

        encoder = model.ConvolutionEncoder(embedding)
        decoder = model.DeconvolutionDecoder(embedding, args.tau)
    else:
        print("Restart from snapshot")
        encoder = torch.load(args.enc_snapshot)
        decoder = torch.load(args.dec_snapshot)

    train_reconstruction(train_loader, test_loader, encoder, decoder, args)

if __name__ == '__main__':
    main()