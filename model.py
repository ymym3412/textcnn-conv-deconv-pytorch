import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class ConvolutionEncoder(nn.Module):
    def __init__(self, embedding, sentence_len, filter_size, filter_shape, latent_size):
        super(ConvolutionEncoder, self).__init__()
        self.embed = embedding
        self.convs1 = nn.Conv2d(1, filter_size, (filter_shape, self.embed.weight.size()[1]), stride=2)
        self.bn1 = nn.BatchNorm2d(filter_size)
        self.convs2 = nn.Conv2d(filter_size, filter_size * 2, (filter_shape, 1), stride=2)
        self.bn2 = nn.BatchNorm2d(filter_size * 2)
        self.convs3 = nn.Conv2d(filter_size * 2, latent_size, (sentence_len, 1), stride=2)

        # weight initialize for conv layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def __call__(self, x):
        x = self.embed(x)

        # x.size() is (L, emb_dim) if batch_size is 1.
        # So interpolate x's dimension if batch_size is 1.
        if len(x.size()) < 3:
            x = x.view(1, *x.size())
        # reshape for convolution layer
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])

        h1 = F.relu(self.bn1(self.convs1(x)))
        h2 = F.relu(self.bn2(self.convs2(h1)))
        h3 = F.relu(self.convs3(h2))

        return h3


class DeconvolutionDecoder(nn.Module):
    def __init__(self, embedding, tau, sentence_len, filter_size, filter_shape, latent_size):
        super(DeconvolutionDecoder, self).__init__()
        self.tau = tau
        self.embed = embedding
        self.deconvs1 = nn.ConvTranspose2d(latent_size, filter_size * 2, (sentence_len, 1), stride=2)
        self.bn1 = nn.BatchNorm2d(filter_size * 2)
        self.deconvs2 = nn.ConvTranspose2d(filter_size * 2, filter_size, (filter_shape, 1), stride=2)
        self.bn2 = nn.BatchNorm2d(filter_size)
        self.deconvs3 = nn.ConvTranspose2d(filter_size, 1, (filter_shape, self.embed.weight.size()[1]), stride=2)

        # weight initialize for conv_transpose layer
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def __call__(self, h3):
        h2 = F.relu(self.bn1(self.deconvs1(h3)))
        h1 = F.relu(self.bn2(self.deconvs2(h2)))
        x_hat = F.relu(self.deconvs3(h1))
        x_hat = x_hat.squeeze()

        # x.size() is (L, emb_dim) if batch_size is 1.
        # So interpolate x's dimension if batch_size is 1.
        if len(x_hat.size()) < 3:
            x_hat = x_hat.view(1, *x_hat.size())
        # normalize
        norm_x_hat = torch.norm(x_hat, 2, dim=2, keepdim=True)
        rec_x_hat = x_hat / norm_x_hat

        # compute probability
        norm_w = Variable(self.embed.weight.data).t()
        prob_logits = torch.bmm(rec_x_hat, norm_w.unsqueeze(0)
                         .expand(rec_x_hat.size(0), *norm_w.size())) / self.tau
        log_prob = F.log_softmax(prob_logits, dim=2)
        return log_prob


class MLPClassifier(nn.Module):
    def __init__(self, output_dim, dropout):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(500, 300)
        self.out = nn.Linear(300, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(self.fc1(x))
        out = self.out(h)
        return F.log_softmax(out, dim=1)
