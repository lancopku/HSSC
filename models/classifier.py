'''
 @Date  : 2018/1/23
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn 
 @homepage: shumingma.com
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
import models


class classification(nn.Module):

    def __init__(self, config, use_attention=True, encoder=None, decoder=None):
        super(classification, self).__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = models.rnn_encoder(config)

        self.use_cuda = config.use_cuda
        self.config = config

        self.classifier = nn.Linear(config.hidden_size, config.num_label)
        self.label_criterion = nn.CrossEntropyLoss(size_average=True)

    def compute_loss(self, scores, targets):
        loss = self.label_criterion(scores, targets)
        return loss

    def classify(self, state):
        scores = self.classifier(state.max(0)[0])
        return scores

    def forward(self, src, src_len, dec, targets, label):
        src = src.t()

        contexts, state = self.encoder(src, src_len.data.tolist())
        label_scores = self.classify(contexts)

        loss = self.compute_loss(label_scores, label)
        return loss, None

    def sample(self, src, src_len, label):

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, reverse_indices = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        label = torch.index_select(label, dim=0, index=indices)
        src = src.t()

        contexts, state = self.encoder(src, lengths.data.tolist())
        predicts = self.classify(contexts).max(1)[1]

        correct_five = torch.sum(torch.eq(predicts.data, label.data).float())
        correct_two = torch.sum(torch.eq(torch.ge(predicts.data, 3), torch.ge(label.data, 3)).float())

        return None, None, correct_five, correct_two

    def beam_sample(self, src, src_len, label, beam_size=1):
        return self.sample(src, src_len, label)
