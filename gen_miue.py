import torch
import os
import argparse
import time
from utils import AverageMeter, cosine_annealing, logger, accuracy, save_checkpoint, Augment, setup_seed, nt_xent, l2sim
from torch.nn import functional as F
from torchvision.models import resnet18, resnet50
from models.resnet import resnet18, resnet50
from models import vgg19, DenseNet121, WideResNet, vgg11
from dataloaders import PrepareDataLoaders
from torch import nn

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='GEN-MI-UE')
    parser.add_argument('--experiment', type=str, required=True, help='name of experiment')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vgg19',
                                                                             'mobilenet', 'densenet121', 'wrn34-10'],
                        help='the model arch used in experiment')

    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet',
                                                                 'miniimagenet', 'imagenet100'],
                        help='the dataset used in experiment')

    parser.add_argument('--data', type=str, default='./data/CIFAR10', help='the directory of dataset')
    parser.add_argument('--num-classes', default=10, type=int, help='the number of classes in the dataset')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--poison-batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cosine-warmup', default=0, type=int)
    parser.add_argument('--data-aug', type=bool, default=True,
                        help='if using data augmentation')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='the temperature when conducting mi-ue loss')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='the step size of perturbation')
    parser.add_argument('--poison-size', type=int, default=32,
                        help='the image size of poisons')
    parser.add_argument('--post-aug', action='store_true',
                        help='if generate post-augmentation poisons')
    parser.add_argument('--perturb-iters', default=10, type=int,
                        help='the number of PGD steps for updating poisons')
    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='the optimizer used in training')
    parser.add_argument('--epochs', default=100, type=int,
                        help='the number of total epochs to run')
    parser.add_argument('--lr', default=0.1, type=float, help='optimizer learning rate')
    parser.add_argument('--resume', action='store_true', help='if resume training')
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    parser.add_argument('--gpu-id', type=str, default='0', help='the gpu id')
    parser.add_argument('--poison-freq', default=1, type=int,
                        help='the frequency of epoch to generate poisons')
    parser.add_argument('--epsilon', default=8, type=float)
    parser.add_argument('--model-iters', default=10, type=int,
                        help='the number of PGD steps for updating models')
    parser.add_argument('--model-alpha', type=float, default=1.6,
                        help='the step size of updating models')
    parser.add_argument('--poison-random', action='store_true',
                        help='if using random when generating poisons')
    parser.add_argument('--pgd-random', action='store_true',
                        help='if using random when conducting pgd attacks')
    parser.add_argument('--attack', type=str, default='none',
                        help='type of attack when updating models')
    parser.add_argument('--poison-loss-type', default='nt_xent_l2', choices=['nt_xent', 'l2', 'nt_xent_l2',],
                        help='the type of loss function when generating poisons')
    parser.add_argument('--train-loss-type', default='cross_entropy', choices=['cross_entropy', 'mixed'],
                        help='the type of loss function when training models')
    parser.add_argument('--pgd-loss-type', default='cross_entropy', choices=['cross_entropy', 'mixed'],
                        help='the type of loss function when conducting pgd attacks')
    parser.add_argument('--poison-update-epsilon', type=float, default=8,
                        help='the budget epsilon when updating poisons, do not affect final budget')
    parser.add_argument('--zeta', type=float, default=0.0,
                        help='the strength of mi_ue loss')
    parser.add_argument('--eot', action='store_true',
                        help='if using EoT')
    parser.add_argument('--eot-nums', type=int, default=5,
                        help='nums when conducting EoT')

    arguments = parser.parse_args()
    arguments.epsilon = arguments.epsilon / 255
    arguments.alpha = arguments.alpha / 255
    arguments.poison_update_epsilon = arguments.poison_update_epsilon / 255

    return arguments



def perturb(train_loader, model, poison, aug, perturb_iters, type='nt_xent', alpha=0.2 / 255, t=0.1, random=False, eps=8/255):
    model.requires_grad_(False)
    for i, (data, target, index) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        poison[index] = error_minimizing(model, data, target, index, poison, aug, iters=perturb_iters, type=type, alpha=alpha, t=t, random=random, eps=eps, eot=args.eot, eot_nums=args.eot_nums)
    model.requires_grad_(True)
    return poison


def error_minimizing(model, data, target, index, poison, aug,  eps=8. / 255., alpha=0.2 / 255., iters=20,
                     type='nt_xent', t=0.1, random=False, eot=False, eot_nums=5):

    if random:
        poison = poison + torch.randn_like(poison).uniform_(-eps, eps).cuda()
        poison = torch.clamp(poison, min=-eps, max=eps)

    delta = poison[index]
    delta = torch.nn.Parameter(delta)


    for _ in range(iters):
        inputs = data + delta

        if eot:
            for j in range(eot_nums):
                img = aug(inputs)
                features, logits = model.eval()(img)
                model.zero_grad()
                if type == 'nt_xent':
                    loss = nt_xent(features, target, t=t)
                elif type == 'l2':
                    loss = l2sim(features, target)
                elif type == 'nt_xent_l2':
                    loss = nt_xent(features, target, t=t) + args.zeta * l2sim(features, target)
                else:
                    raise {'type error'}
                gd = torch.autograd.grad(loss, [img])[0]

                upd_loss = (img * gd).sum()
                upd_loss.backward()

        else:
            features, logits = model.eval()(inputs)
            model.zero_grad()
            if type == 'nt_xent':
                loss = nt_xent(features, target, t=t)
            elif type == 'l2':
                loss = l2sim(features, target)
            elif type == 'nt_xent_l2':
                loss = nt_xent(features, target, t=t) + args.zeta * l2sim(features, target)
            else:
                raise {'type error'}


            loss.backward()

        delta.data = delta.data - alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(data + delta.data, min=0, max=1) - data

    if args.dataset == 'imagenet100':
        delta = delta.to(torch.float16)
        print(loss)
    return delta.detach()


def pgd(input, target, model, step=10, epsilon=0.031, alpha=0.007, random=False, loss_type='cross_entropy', zeta=1.0):
    model.eval()
    adv_input = input.clone()
    if random:
        adv_input = adv_input + 0.5 * torch.FloatTensor(*adv_input.shape).uniform_(-epsilon, epsilon).cuda()
    for _ in range(step):
        adv_input.requires_grad = True
        with torch.enable_grad():
            feature, output = model(adv_input)
            if loss_type == 'cross_entropy':
                loss = F.cross_entropy(output, target)
            elif loss_type == 'mixed':
                loss = F.cross_entropy(output, target) + zeta * nt_xent(feature, target, t=0.1)
            model.zero_grad()
            loss.backward()
        with torch.no_grad():
            adv_input = adv_input + alpha * adv_input.grad.sign()
            delta = adv_input - input
            delta = torch.clamp(delta, -epsilon, epsilon)
            adv_input = (input + delta).detach()
        adv_input = torch.clamp(adv_input, 0., 1.)
    model.train()
    return adv_input


def train_epoch(train_loader, poison_train_loader, model, poison, optimizer, scheduler, epoch, log, aug,
                poison_loss_type='nt_xent', attack='none', train_loss_type='cross_entropy', zeta=1.0, eps=8/255, alpha=1/255):
    losses = AverageMeter()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()
    pert_time_meter = AverageMeter()

    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    start = time.time()


    for i, (data, target, index) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()

        #inputs = aug(data + poison[index])
        inputs = data + poison[index]

        if attack == 'pgd':
            inputs = pgd(inputs, target, model, alpha=args.model_alpha, epsilon=args.epsilon, step=args.model_iters,
                       random=args.pgd_random, loss_type=args.pgd_loss_type, zeta=args.zeta)

        data_time = time.time() - start
        data_time_meter.update(data_time)

        features, logits = model.train()(inputs)
        #print(features.size())
        #raise Va
        if train_loss_type == 'cross_entropy':
            loss = F.cross_entropy(logits, target)
        elif train_loss_type == 'mixed':
            loss = F.cross_entropy(logits, target) + zeta * nt_xent(features, target, t=0.1)
        else:
            raise {'train loss type not support yet'}

        #print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), len(data))
        train_time = time.time() - start
        train_time_meter.update(train_time)
        start = time.time()

        torch.cuda.empty_cache()


    if epoch % args.poison_freq == 0:
        model.eval()
        poison = perturb(poison_train_loader, model, poison, aug, args.perturb_iters,
                         type=poison_loss_type, alpha=alpha, t=args.temperature, random=args.poison_random, eps=eps)
        model.train()


    pert_time_meter.update(time.time() - start)
    log.info(
        f'Epoch[{epoch}/{args.epochs}]\t'
        f'current lr = {current_lr:.3f}\t'
        f'avg loss = {losses.avg:.4f}\t'
        f'train time = {train_time_meter.sum:.2f}\t'
        f'data time = {data_time_meter.sum:.2f}\t'
        f'pert time = {pert_time_meter.sum:.2f}\t'
        f'epoch time = {train_time_meter.sum+pert_time_meter.sum:.2f}'
    )
    scheduler.step()
    return poison


def evaluation(loader, model):
    top1 = AverageMeter()
    for i, (data, target, _) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            _, outputs = model.eval()(data)
        prec1 = accuracy(outputs.data, target)[0]
        top1.update(prec1.item(), len(data))
    return top1.avg


def main(args):
    if args.seed is not None:
        setup_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    save_dir = os.path.join('gen', args.dataset, args.backbone, args.experiment)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log = logger(path=save_dir)
    log.info(str(args))


    dataloaders = PrepareDataLoaders(args.dataset, root=args.data, output_size=args.poison_size, for_gen=True,
                                     supervised=True, post_aug=args.post_aug, data_aug=args.data_aug)
    train_loader = dataloaders.get_train_loader(args.batch_size, args.num_workers)
    test_loader = dataloaders.get_test_loader(args.batch_size, args.num_workers)

    poison_train_loader = dataloaders.get_train_loader(args.poison_batch_size, args.num_workers)



    if args.backbone == 'resnet18':
        model = resnet18(num_classes=args.num_classes, get_feature=True)
        if args.dataset in ['cifar10', 'cifar100']:
            model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            model.maxpool = nn.Identity()
    elif args.backbone == 'resnet50':
        if args.dataset in ['cifar10', 'cifar100']:
            model = resnet50(num_classes=args.num_classes, get_feature=True)
            model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
    elif args.backbone == 'vgg19':
        model = vgg19(num_classes=args.num_classes)
    elif args.backbone == 'vgg11':
        model = vgg11(num_classes=args.num_classes)
    elif args.backbone == 'densenet121':
        model = DenseNet121(num_classes=args.num_classes)
    elif args.backbone == 'wrn34-10':
        model = WideResNet(num_classes=args.num_classes)
    else:
        raise AssertionError('model is not defined')

    model = model.cuda()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        raise AssertionError('optimizer is not defined')

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(step,
                                                args.epochs,
                                                1,
                                                1e-6 / args.lr,
                                                warmup_steps=0)
    )
    if args.dataset == 'imagenet100':
        poison = torch.zeros(len(train_loader.dataset), 3, args.poison_size, args.poison_size, dtype=torch.float16).cuda()
    else:
        poison = torch.zeros(len(train_loader.dataset), 3, args.poison_size, args.poison_size).cuda()

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(os.path.join(save_dir, 'model.pt'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim'])
        for i in range(start_epoch - 1):
            scheduler.step()
        log.info(f"RESUME FROM EPOCH {start_epoch-1}")
        poison = torch.load(os.path.join(save_dir, 'poison.pt'), map_location='cuda')

    for epoch in range(start_epoch, args.epochs + 1):
        eps = (args.epochs - epoch + 1) / args.epochs * (args.poison_update_epsilon - 8/255) + 8/255
        final_alpha = 4 / (255 * args.perturb_iters)
        alpha = (args.epochs - epoch + 1) / args.epochs * (args.alpha - final_alpha) + final_alpha
        log.info(f"current eps:{eps}, current alpha:{alpha}")
        if args.post_aug:
            aug = Augment().aug_id
        else:
            if args.data_aug:
                aug = Augment(args.poison_size).aug_standard
            else:
                aug = Augment().aug_id
        poison = train_epoch(train_loader, poison_train_loader, model, poison, optimizer, scheduler, epoch, log, aug,
                        poison_loss_type=args.poison_loss_type, attack=args.attack,
                        train_loss_type=args.train_loss_type, zeta=args.zeta, eps=eps, alpha=alpha)

        if epoch % 5 == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, filename=os.path.join(save_dir, 'model.pt'))

            if epoch == args.epochs:
                poison = torch.clamp(poison, -8/255, 8/255)
            torch.save(poison, os.path.join(save_dir, 'poison.pt'))
            val_acc = evaluation(train_loader, model)
            test_acc = evaluation(test_loader, model)
            log.info(
                f'EVALUATION\t'
                f'val no-aug accuracy = {val_acc:.4f}\t'
                f'test accuracy = {test_acc:.4f}'
            )



if __name__ == '__main__':
    args = get_args()

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.poison_size = 32
    if args.dataset == 'cifar100':
        args.num_classes = 100
        args.poison_size = 32
    if args.dataset == 'tinyimagenet':
        args.num_classes = 200
        args.poison_size = 64
    if args.dataset == 'miniimagenet':
        args.num_classes = 100
        args.poison_size = 84
    if args.dataset == 'imagenet100':
        args.num_classes = 100
        args.poison_size = 224

    main(args)
