from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid.utils.data.sampler import RandomIdentitySampler

from reid.datasets.domain_adaptation import DA
from reid import models
from reid.trainers import Trainer, HHLTrainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor, CameraPreprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.loss import TripletLoss


def get_data(data_dir, source, target, height, width, batch_size, triplet_batch_size, num_instances, target_batch_size, re=0, workers=8):

    dataset = DA(data_dir, source, target)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=re),
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    source_train_loader = DataLoader(
        Preprocessor(dataset.source_train, root=osp.join(dataset.source_images_dir, dataset.source_train_path),
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)

    source_triplet_train_loader = DataLoader(
        Preprocessor(dataset.source_train, root=osp.join(dataset.source_images_dir, dataset.source_train_path),
                     transform=train_transformer),
        batch_size=triplet_batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(dataset.source_train, num_instances),
        pin_memory=True, drop_last=True)

    target_train_loader = DataLoader(
        CameraPreprocessor(dataset.target_train, root=dataset.target_images_dir, target_path=dataset.target_train_path,
                           target_camstyle_path=dataset.target_train_camstyle_path, transform=train_transformer, num_cam=dataset.target_num_cam),
        batch_size=target_batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.target_images_dir, dataset.query_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.target_images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, source_train_loader, source_triplet_train_loader, target_train_loader, query_loader, gallery_loader


def main(args):
    cudnn.benchmark = True
    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    dataset, num_classes, source_train_loader, source_triplet_train_loader, \
        target_train_loader, query_loader, gallery_loader = \
        get_data(args.data_dir, args.source, args.target, args.height,
                 args.width, args.batch_size, args.triplet_batch_size, args.num_instances, args.target_batch_size, args.re, args.workers)

    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes, triplet_features=args.triplet_features)

    # Load from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {} "
              .format(start_epoch))
    # model = nn.DataParallel(model).cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)
        return

    # Criterion
    # cross-entropy loss
    criterion_c = nn.CrossEntropyLoss().to(device)
    # triplet loss
    criterion_t = TripletLoss(margin=args.margin).to(device)

    # Optimizer
    if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters())) \
                         | set(map(id, model.module.triplet.parameters())) \
                         | set(map(id, model.module.feat.parameters())) \
                         | set(map(id, model.module.feat_bn.parameters()))

        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Trainer
    trainer = HHLTrainer(model, criterion_c, criterion_t, args.beta)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 40
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, source_train_loader, source_triplet_train_loader, target_train_loader, optimizer)

        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
        }, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d} \n'.
              format(epoch))

    # Final test
    print('Test with best model:')
    evaluator = Evaluator(model)
    evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, args.output_feature, args.rerank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="baseline")
    # source
    parser.add_argument('-s', '--source', type=str, default='duke',
                        choices=['market', 'duke', 'cuhk03_detected'])
    # target
    parser.add_argument('-t', '--target', type=str, default='market',
                        choices=['market', 'duke'])
    # images
    parser.add_argument('-b', '--batch-size', type=int, default=128, help="batch size for source")
    parser.add_argument('--triplet-batch-size', type=int, default=64, help="triplet batch size for source")
    parser.add_argument('--target-batch-size', type=int, default=16, help="batch size for target")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    parser.add_argument('--num-instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(triplet_batch_size // num_instances) identities, and "
                             "each identity has num_instances instances for source, "
                             "default: 8")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--triplet-features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    #random erasing
    parser.add_argument('--re', type=float, default=0)
    #  perform re-ranking
    parser.add_argument('--rerank', action='store_true', help="perform re-ranking")
    # triplet loss weight
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--margin', type=float, default=0.3,
                        help="margin of the triplet loss, default: 0.3")

    main(parser.parse_args())
