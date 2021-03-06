# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main script to launch AugMix training on CIFAR-10/100.

Supports WideResNet, AllConv, ResNeXt models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.

Example usage:
  `python cifar.py`
"""
from __future__ import print_function

import argparse
import os
import shutil
import time

import augmentations
from models.cifar.allconv import AllConvNet
import numpy as np
from models.third_party.ResNeXt_DenseNet.models.densenet import densenet
from models.third_party.ResNeXt_DenseNet.models.resnext import resnext29
from models.third_party.WideResNet_pytorch.wideresnet import WideResNet
from models.third_party.WideResNet_pytorch.resnet_cifar import resnet20


from models.cifar.attentions.attender2 import Attender

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch import nn
from thop import profile
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from utils.radam import RAdam, RAdam_4step, AdamW
import torch.optim as optim

def margin_loss(x, labels, size_average=True):
    batch_size = x.size(0)
    
    left = F.relu(0.9 - x).view(batch_size, -1)
    right = F.relu(x - 0.1).view(batch_size, -1)

    loss = labels * left + 0.5 * (1.0 - labels) * right
    loss = loss.sum(dim=1).mean()

    return loss

parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument(
    '--model',
    '-m',
    type=str,
    default='attender',
    choices=['resnet20', 'attender', 'dualT', 'perceiver', 'vit', 'wrn', 'allconv', 'densenet', 'resnext'],
    help='Choose architecture.')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.1,
    help='Initial learning rate.')

parser.add_argument('--optimizer', default='sgd', type=str, choices=['adam', 'adamw', 'radam', 'radam4s', 'sgd'])
parser.add_argument('--beta1', default=0.9, type=float,
                    help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='beta2 for adam')
parser.add_argument('--warmup', default=0, type=float,
                    help='warmup steps for adam')

# 4 step options
parser.add_argument('--update_all', action='store_true')
parser.add_argument('--additional_four', action='store_true')

parser.add_argument(
    '--batch-size', '-b', type=int, default=96, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=128)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0005,
    help='Weight decay (L2 penalty).')
# WRN Architecture options
parser.add_argument(
    '--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
parser.add_argument(
    '--droprate', default=0.0, type=float, help='Dropout probability')
# AugMix options
parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument(
    '--aug-severity',
    default=3,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--no-augmix',
    '-na',
    action='store_true',
    help='Turn off augmix augmentation.')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all operations (+brightness,contrast,color,sharpness).')
# Checkpointing options
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default='./snapshots',
    help='Folder to save checkpoints.')
parser.add_argument(
    '--resume',
    '-r',
    type=str,
    default='',
    help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=50,
    help='Training loss print frequency (batches).')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=12,
    help='Number of pre-fetching threads.')

args = parser.parse_args()

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


# def get_lr(step, total_steps, lr_max, lr_min):
#   """Compute learning rate according to cosine annealing schedule."""
#   return lr_min + (lr_max - lr_min) * 0.5 * (1 +
#                                              np.cos(step / total_steps * np.pi))


def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if args.all_ops:
    aug_list = augmentations.augmentations_all

# dirichlet refers to a generalised beta dist
  ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(args.mixture_width):
    image_aug = image.copy()
    depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, args.aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=False, no_augmix=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd
    self.no_augmix = no_augmix

  def __getitem__(self, i):
    x, y = self.dataset[i]
    
    if self.no_augmix:
      return self.preprocess(x), y
    elif self.no_jsd:
      return aug(x, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)


def train(net, train_loader, optimizer, scheduler, criterion=None):
  """Train for one epoch."""
  net.train()
  loss_ema = 0.
  for i, (images, targets) in enumerate(train_loader):
    optimizer.zero_grad()
    
    mages = images.cuda()
    targets = targets.cuda()
    logits = net(images)
    loss = F.cross_entropy(logits, targets)

    # if args.no_jsd or args.no_augmix:
    #   images = images.cuda()
    #   # targets = torch.eye(10).index_select(dim=0, index=targets)
    #   targets = targets.cuda()
    #   logits = net(images)
    #   if criterion is not None:
    #     loss = margin_loss(logits, targets)
    #   else:
    #     loss = F.cross_entropy(logits, targets)
    # else:
    #   images_all = torch.cat(images, 0).cuda()
    #   targets = targets.cuda()
    #   logits_all = net(images_all)
    #   logits_clean, logits_aug1, logits_aug2 = torch.split(
    #       logits_all, images[0].size(0))

    #   # Cross-entropy is only computed on clean images
    #   # loss from classification only carried out on original image
    #   loss = F.cross_entropy(logits_clean, targets)

    #   # classification on the clean, and the augmented to get probs.
    #   # but loss not used here
    #   p_clean, p_aug1, p_aug2 = F.softmax(
    #       logits_clean, dim=1), F.softmax(
    #           logits_aug1, dim=1), F.softmax(
    #               logits_aug2, dim=1)

    #   # Clamp mixture distribution to avoid exploding KL divergence
    #   # forcing into a shared embedding
    #   # aug2 refers to figure 4 being carried out twice, double augmentation
    #   # kl is not symmetrical, JS can be symmetrical, making it JS symmetric by adding the different combinations of dists together
    #   p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    #   loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
    #                 F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
    #                 F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    loss.backward()
    optimizer.step()
    loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    if i % args.print_freq == 0:
      print('Train Loss {:.3f}'.format(loss_ema))

  return loss_ema


def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data) * images.shape[0]
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)


def test_c(net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  for corruption in CORRUPTIONS:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))

  return np.mean(corruption_accs)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
  
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def main():
  
  # fix the seed for reproducibility
  seed = args.seed + get_rank()
  torch.manual_seed(seed)
  np.random.seed(seed)
  # random.seed(seed)

  cudnn.benchmark = True
  
  
  normalize = [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

  augmentations = []
  args.disable_aug = False
  if not args.disable_aug:
      from utils.autoaug import CIFAR10Policy
      print("Using Auto augmentation....")
      augmentations += [
          CIFAR10Policy()
      ]
  augmentations += [
      transforms.RandomHorizontalFlip(),
      # transforms.RandomCrop(32, padding=4),
      transforms.ToTensor(),
      *normalize,
  ]

  augmentations = transforms.Compose(augmentations)
  train_data = datasets.CIFAR10(
        './data/cifar', train=True, transform=augmentations, download=True)

  test_data = datasets.CIFAR10(
      root='./data/cifar', train=False, download=True, transform=transforms.Compose([
          transforms.ToTensor(),
          *normalize,
  ]))
  

  # Load datasets
  # train_transform = transforms.Compose(
  #     [transforms.RandomHorizontalFlip(),
  #      transforms.RandomCrop(32, padding=4)])
  # preprocess = transforms.Compose(
  #     [transforms.ToTensor(),
  #      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  #      #transforms.Normalize([0.5] * 3, [0.5] * 3)])
  # test_transform = preprocess

  # if args.dataset == 'cifar10':
  #   train_data = datasets.CIFAR10(
  #       './data/cifar', train=True, transform=train_transform, download=True)
  #   test_data = datasets.CIFAR10(
  #       './data/cifar', train=False, transform=test_transform, download=True)
  #   base_c_path = './data/cifar/CIFAR-10-C/'
  #   num_classes = 10
  # else:
  #   train_data = datasets.CIFAR100(
  #       './data/cifar', train=True, transform=train_transform, download=True)
  #   test_data = datasets.CIFAR100(
  #       './data/cifar', train=False, transform=test_transform, download=True)
  #   base_c_path = './data/cifar/CIFAR-100-C/'
  #   num_classes = 100

  # train_data = AugMixDataset(train_data, preprocess, args.no_jsd, args.no_augmix)
  
  num_classes = 10
  base_c_path = './data/cifar/CIFAR-10-C/'
  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True)

  test_loader = torch.utils.data.DataLoader(
      test_data,
      batch_size=args.eval_batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)

  # Create model
  if args.model == 'densenet':
    net = densenet(num_classes=num_classes)
  elif args.model == 'wrn':
    net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
  elif args.model == 'allconv':
    net = AllConvNet(num_classes)
  elif args.model == 'resnext':
    net = resnext29(num_classes=num_classes)
  elif args.model == 'resnet20':
    net = resnet20()
  elif args.model == 'attender':
    net = Attender(num_classes=num_classes)
    # input = torch.randn(1, 3, 32, 32)
    # macs, params = profile(net, inputs=(input, ))
    # print("Macs: ", macs)
    # print("params: ", params)

  # optimizer = torch.optim.SGD(
  #     net.parameters(),
  #     args.learning_rate,
  #     momentum=args.momentum,
  #     weight_decay=args.decay,
  #     nesterov=True)
  
  optimizer = torch.optim.Adam(
      net.parameters(),
      args.learning_rate)
  
  
  if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
  elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate) #, betas=(args.beta1, args.beta2), weight_decay=args.decay)
  elif args.optimizer.lower() == 'radam':
      optimizer = RAdam(net.parameters(), lr=args.learning_rate) #, betas=(args.beta1, args.beta2), weight_decay=args.decay)
  elif args.optimizer.lower() == 'radam4s':
      optimizer = RAdam_4step(net.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.decay, update_all=args.update_all, additional_four=args.additional_four)
  elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate) #, betas=(args.beta1, args.beta2), weight_decay=args.decay, warmup=args.warmup)
  
  n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
  print('number of params:', n_parameters)

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  start_epoch = 0

  if args.resume:
    if os.path.isfile(args.resume):
      checkpoint = torch.load(args.resume)
      start_epoch = checkpoint['epoch'] + 1
      best_acc = checkpoint['best_acc']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print('Model restored from epoch:', start_epoch)

  if args.evaluate:
    # Evaluate clean accuracy first because test_c mutates underlying data
    test_loss, test_acc = test(net, test_loader)
    print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
        test_loss, 100 - 100. * test_acc))

    test_c_acc = test_c(net, test_data, base_c_path)
    print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))
    return

  # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
  
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, 
                                                         threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
  
  # scheduler = torch.optim.lr_scheduler.LambdaLR(
  #     optimizer,
  #     lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
  #         step,
  #         args.epochs * len(train_loader),
  #         1,  # lr_lambda computes multiplicative factor
  #         1e-6 / args.learning_rate))

  if not os.path.exists(args.save):
    os.makedirs(args.save)
  if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

  log_path = os.path.join(args.save,
                          args.dataset + '_' + args.model + '_training_log.csv')
  with open(log_path, 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

  best_acc = 0
  print('Beginning training from epoch:', start_epoch + 1)
  for epoch in range(start_epoch, args.epochs):
    begin_time = time.time()

    criterion = None #"margin_loss"
    train_loss_ema = train(net, train_loader, optimizer, scheduler, criterion=criterion)
    test_loss, test_acc = test(net, test_loader)
    scheduler.step(test_loss)

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    checkpoint = {
        'epoch': epoch,
        'dataset': args.dataset,
        'model': args.model,
        'state_dict': net.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }

    save_path = os.path.join(args.save, 'checkpoint.pth.tar')
    torch.save(checkpoint, save_path)
    if is_best:
      shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

    with open(log_path, 'a') as f:
      f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
          (epoch + 1),
          time.time() - begin_time,
          train_loss_ema,
          test_loss,
          100 - 100. * test_acc,
      ))

    print(
        'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
        ' Test Error {4:.2f}'
        .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
                test_loss, 100 - 100. * test_acc))

  test_c_acc = test_c(net, test_data, base_c_path)
  print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

  with open(log_path, 'a') as f:
    f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
            (args.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))


if __name__ == '__main__':
  main()
