from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import TripletLoss
from .utils.meters import AverageMeter
import pdb


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        return inputs, pids

    def _forward(self, inputs, targets):
        outputs = self.model(inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec


class HHLTrainer(object):
    def __init__(self, model, criterion_c, criterion_t, beta):
        super(HHLTrainer, self).__init__()
        self.model = model
        self.criterion_c = criterion_c
        self.criterion_t = criterion_t
        self.beta = beta
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, source_train_loader, source_triplet_train_loader, target_train_loader, optimizer, print_freq=1):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions_c = AverageMeter()
        precisions_t = AverageMeter()

        target_train_iter = iter(target_train_loader)
        source_triplet_train_iter = iter(source_triplet_train_loader)

        end = time.time()
        for i, inputs_source in enumerate(source_train_loader):
            data_time.update(time.time() - end)
            # load source_triplet and target
            try:
                inputs_source_triplet = next(source_triplet_train_iter)
                inputs_target = next(target_train_iter)
            except:
                target_train_iter = iter(target_train_loader)
                source_triplet_train_iter = iter(source_triplet_train_loader)
                inputs_source_triplet = next(source_triplet_train_iter)
                inputs_target = next(target_train_iter)

            inputs_source, pids_source = self._parse_data(inputs_source)
            inputs_source_triplet, pids_source_triplet = self._parse_data(inputs_source_triplet)
            inputs_target, pids_target = self._parse_target_data(inputs_target)

            loss, prec_c, prec_t = self._forward(inputs_source, pids_source, inputs_source_triplet, pids_source_triplet,
                                        inputs_target, pids_target, epoch)

            losses.update(loss.item(), pids_source.size(0))
            precisions_c.update(prec_c, pids_source.size(0))
            precisions_t.update(prec_t, pids_source_triplet.size(0) + pids_target.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec_c {:.2%} ({:.2%})\t'
                      'Prec_t {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(source_train_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions_c.val, precisions_c.avg,
                              precisions_t.val, precisions_t.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        return inputs, pids

    def _parse_target_data(self, inputs):
        imgs, _, pids, _ = inputs
        imgs = torch.cat(imgs)
        pids = torch.cat(pids)
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        return inputs, pids

    def _forward(self, inputs_source, pids_source, inputs_source_triplet, pids_source_triplet,
                                        inputs_target, pids_target, epoch):
        outputs_source, _ = self.model(inputs_source)
        _, outputs_source_triplet = self.model(inputs_source_triplet)
        _, outputs_target = self.model(inputs_target)

        # cross-entropy loss for source
        loss_c = self.criterion_c(outputs_source, pids_source)
        prec_c, = accuracy(outputs_source.data, pids_source.data)
        prec_c = prec_c[0]
        # HHL loss
        loss_t, prec_t = self.criterion_t(torch.cat((outputs_source_triplet, outputs_target)), torch.cat((pids_source_triplet, pids_target)))
        # overall loss
        loss = loss_c + self.beta * loss_t
        return loss, prec_c, prec_t
