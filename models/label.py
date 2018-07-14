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


class label(nn.Module):

    def __init__(self, config, use_attention=True, encoder=None, decoder=None):
        super(label, self).__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = models.rnn_encoder(config)
        tgt_embedding = self.encoder.embedding if config.shared_vocab else None
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = models.label_rnn_decoder(config, embedding=tgt_embedding, use_attention=use_attention)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.use_cuda = config.use_cuda
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=utils.PAD, size_average=False)

        self._classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size*2),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.2),
                                         nn.Linear(config.hidden_size*2, config.num_label))

        self.label_criterion = nn.CrossEntropyLoss(size_average=True)

    def compute_loss(self, scores, targets):
        scores = scores.view(-1, scores.size(2))
        loss = self.criterion(scores, targets.contiguous().view(-1))
        num_total = targets.ne(utils.PAD).data.sum()
        return loss / num_total

    def compute_label_loss(self, scores, targets):
        loss = self.label_criterion(scores, targets)
        return loss

    def classify(self, state):
        scores = self._classifier(state.max(0)[0])
        return scores

    def forward(self, src, src_len, dec, targets, label):
        src = src.t()
        dec = dec.t()
        targets = targets.t()

        contexts, state = self.encoder(src, src_len.data.tolist())

        self.decoder.semantic_attention.init_context(context=contexts)
        self.decoder.sentiment_attention.init_context(context=contexts)

        outputs, hiddens = [], []
        for input in dec.split(1):
            semantic_output, sentiment_output, state, attn_weights = self.decoder(input.squeeze(0), state)
            outputs.append(semantic_output)
            hiddens.append(sentiment_output)

        outputs = torch.stack(outputs)
        hiddens = torch.stack(hiddens)

        #print(hiddens.size())
        label_scores = self.classify(torch.cat([contexts, hiddens], dim=0))
        #print(label_scores.size())

        loss = self.compute_loss(outputs, targets) + self.compute_label_loss(label_scores, label)
        return loss, outputs

    def sample(self, src, src_len, label):

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, reverse_indices = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        label = torch.index_select(label, dim=0, index=indices)
        bos = Variable(torch.ones(src.size(0)).long().fill_(utils.BOS), volatile=True)
        src = src.t()

        if self.use_cuda:
            bos = bos.cuda()

        contexts, state = self.encoder(src, lengths.data.tolist())

        self.decoder.semantic_attention.init_context(context=contexts)
        self.decoder.sentiment_attention.init_context(context=contexts)

        inputs, outputs, attn_matrix, hiddens = [bos], [], [], []
        for i in range(self.config.max_time_step):
            semantic_output, sentiment_output, state, attn_weights = self.decoder(inputs[i], state)
            predicted = semantic_output.max(1)[1]
            #predicted = torch.multinomial(torch.nn.functional.softmax(output/0.6), num_samples=1).squeeze(1)
            inputs += [predicted]
            outputs += [predicted]
            attn_matrix += [attn_weights]
            hiddens += [sentiment_output]

        outputs = torch.stack(outputs)
        hiddens = torch.stack(hiddens)
        predicts = self.classify(torch.cat([contexts, hiddens], dim=0)).max(1)[1]
        sample_ids = torch.index_select(outputs, dim=1, index=reverse_indices).t().data

        attn_matrix = torch.stack(attn_matrix)
        alignments = attn_matrix.max(2)[1]
        alignments = torch.index_select(alignments, dim=1, index=reverse_indices).t().data

        correct_five = torch.sum(torch.eq(predicts.data, label.data).float())
        correct_two = torch.sum(torch.eq(torch.ge(predicts.data, 3), torch.ge(label.data, 3)).float())

        return sample_ids, alignments, correct_five, correct_two


    def beam_sample(self, src, src_len, label, beam_size):

        # (1) Run the encoder on the src.

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        label = torch.index_select(label, dim=0, index=indices)
        src = src.t()
        batch_size = src.size(1)
        contexts, encState = self.encoder(src, lengths.data.tolist())
        vector, contexts = torch.chunk(contexts, chunks=2, dim=-1)
        predicts = self.classify(vector).max(1)[1]

        correct_five = torch.sum(torch.eq(predicts.data, label.data).float())
        correct_two = torch.sum(torch.eq(torch.ge(predicts.data, 3), torch.ge(label.data, 3)).float())

        #  (1b) Initialize for the decoder.
        def var(a):
            return Variable(a, volatile=True)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        contexts = rvar(contexts.data)

        if self.config.cell == 'lstm':
            decState = (rvar(encState[0].data), rvar(encState[1].data))
        else:
            decState = rvar(encState.data)
        #decState.repeat_beam_size_times(beam_size)
        beam = [models.Beam(beam_size, n_best=1,
                          cuda=self.use_cuda, length_norm=self.config.length_norm)
                for __ in range(batch_size)]
        if self.decoder.attention is not None:
            self.decoder.attention.init_context(contexts)

        # (2) run the decoder to generate sentences, using beam search.

        for i in range(self.config.max_time_step):

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(-1))

            # Run one step.
            output, decState, attn = self.decoder(inp, decState)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
                # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j])
                b.beam_update(decState, j)

        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []

        for j in ind.data:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])

        return allHyps, allAttn, correct_five, correct_two