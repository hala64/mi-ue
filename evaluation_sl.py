import torch
import torch.backends.cudnn as cudnn
import os
import argparse
import time
from utils import AverageMeter, cosine_annealing, logger, save_checkpoint, evaluation, setup_seed, classwise_evaluation
from torch.nn import functional as F
from dataloaders import PrepareDataLoaders
from torchvision.models import resnet18, resnet50
from models import vgg19, DenseNet121, simple_classifier, WideResNet, SL_ViT
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from cutout import Cutout
import kornia.augmentation as K

def get_args():
    parser = argparse.ArgumentParser(description='EVAL-SL')
    parser.add_argument('--experiment', type=str, required=True, help='name of experiment')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vgg19',
                                                                             'vit-b', 'densenet121', 'linear',
                                                                             '2nn', '3nn', 'lenet5', 'vit', 'wrn34-10'],
                        help='the model arch used in experiment')

    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet',
                                                                 'miniimagenet', 'imagenet100'],
                        help='the dataset used in experiment')
    parser.add_argument('--data', type=str, default='data/CIFAR10', help='the directory of dataset')
    parser.add_argument('--num-classes', default=10, type=int, help='the number of classes in the dataset')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--poison-path', type=str, default=None, help='the path of pretrained poison')
    parser.add_argument('--poison-ratio', type=float, default=1.0, help='the poisoning ratio')
    parser.add_argument('--poison-size', type=int, default=32,
                        help='the image size of poisons')

    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='the optimizer used in training')
    parser.add_argument('--epochs', default=200, type=int,
                        help='the number of total epochs to run')
    parser.add_argument('--lr', default=0.5, type=float, help='optimizer learning rate')
    parser.add_argument('--seed', default=None, type=int, help='random seed')

    parser.add_argument('--resume', action='store_true', help='if resume training')
    parser.add_argument('--cutout', action='store_true', help='use cutout')
    parser.add_argument('--cutmix', action='store_true', help='use cutmix')
    parser.add_argument('--mixup', action='store_true', help='use mixup')
    parser.add_argument('--gaussian-smooth', action='store_true', help='if use gaussian smooth')
    parser.add_argument('--random-noise', action='store_true', help='if use random noise')
    parser.add_argument('--gpu-id', type=str, default='0', help='the gpu id')

    parser.add_argument('--poisoned-class', default=-1, type=int,
                        help='which class could be poisoned, if all classes are poisoned, it is set to -1')

    parser.add_argument('--get-lr-process', action='store_true', help='if get learning process')

    return parser.parse_args()

def loss_mix(y, logits):
    criterion = F.cross_entropy
    loss_a = criterion(logits, y[:, 0].long(), reduction="none")
    loss_b = criterion(logits, y[:, 1].long(), reduction="none")
    return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()

def train_epoch(train_loader, model, optimizer, scheduler, epoch, log):
    losses = AverageMeter()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()
    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    start = time.time()
    if args.cutmix:
        cutmix = K.RandomCutMixV2(data_keys=["input", "class"])
    elif args.mixup:
        mixup = K.RandomMixUpV2(data_keys=["input", "class"])

    for i, (data, target, _) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        data_time = time.time() - start
        data_time_meter.update(data_time)

        if args.cutmix or args.mixup:
            if args.cutmix:
                data, target = cutmix(data, target)
                target = target.squeeze(0)
            elif args.mixup:
                data, target = mixup(data, target)
            #print(target)
            features = model.train()(data)
            loss = loss_mix(target, features)
        else:
            features = model.train()(data)
            loss = F.cross_entropy(features, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), data.shape[0])

        train_time = time.time() - start
        train_time_meter.update(train_time)
        start = time.time()
    log.info(
        f'TRAINING\t'
        f'Epoch[{epoch}/{args.epochs}]\t'
        f'avg loss = {losses.avg:.4f}\t'
        f'epoch time = {train_time_meter.sum:.2f}\t'
        f'data time = {data_time_meter.sum:.2f}\t'
        f'current lr = {current_lr:.4f}'
    )
    scheduler.step()


def main():
    if args.seed is not None:
        setup_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    save_dir = os.path.join('eval', args.dataset, args.backbone, 'SL', args.experiment)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log = logger(path=save_dir)
    log.info(str(args))

    try:
        poison = torch.load(args.poison_path, map_location='cpu')
        print('poison:', torch.max(poison))
    except:
        if args.poison_path == 'random':
            torch.manual_seed(1)
            poison = torch.randn([50000, 3, 32, 32]).uniform_(-8/255, 8/255)
            print('use random noise')
            print('poison:', poison)
        else:
            poison = None
            print('no poison founded!')


    dataloaders = PrepareDataLoaders(args.dataset, root=args.data, output_size=args.poison_size, for_gen=False,
                                    supervised=True, delta=poison, ratio=args.poison_ratio,
                                    cutout=args.cutout, random_noise=args.random_noise,
                                    gaussian_smooth=args.gaussian_smooth, poisoned_class=args.poisoned_class)
    train_loader = dataloaders.get_train_loader(args.batch_size, args.num_workers)
    test_loader = dataloaders.get_test_loader(args.batch_size, args.num_workers)

    im100 = True if args.dataset == 'imagenet100' else False
    if args.backbone == 'resnet18':
        model = resnet18(num_classes=args.num_classes).cuda()
        if args.dataset in ['cifar10', 'cifar100']:
            model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False).cuda()
            model.maxpool = nn.Identity().cuda()
    elif args.backbone == 'resnet50':
        model = resnet50(num_classes=args.num_classes).cuda()
        if args.dataset in ['cifar10', 'cifar100']:
            model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False).cuda()
            model.maxpool = nn.Identity().cuda()
    elif args.backbone == 'vgg19':
        model = vgg19(num_classes=args.num_classes, im100=im100).cuda()
    elif args.backbone == 'densenet121':
        model = DenseNet121(num_classes=args.num_classes, im100=im100).cuda()
    elif args.backbone == 'wrn34-10':
        model = WideResNet(num_classes=args.num_classes, im100=im100).cuda()
    elif args.backbone == 'vit':
        patch_size = 16 if im100 else 4
        model = SL_ViT(image_size=args.poison_size, num_classes=args.num_classes, patch_size=patch_size).cuda()
    elif args.backbone == 'linear':
        model = simple_classifier.Linear(n_classes=args.num_classes).cuda()
    elif args.backbone == '2nn':
        model = simple_classifier.two_NN(n_classes=args.num_classes).cuda()
    elif args.backbone == '3nn':
        model = simple_classifier.three_NN(n_classes=args.num_classes).cuda()
    elif args.backbone == 'lenet5':
        model = simple_classifier.LeNet5(num_classes=args.num_classes).cuda()
    else:
        raise AssertionError('model is not defined')

    if args.dataset == 'imagenet100':
        model = nn.DataParallel(model)

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

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(os.path.join(save_dir, 'model.pt'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim'])
        for i in range(start_epoch - 1):
            scheduler.step()
        log.info(f"RESUME FROM EPOCH {start_epoch-1}")

    if args.cutmix:
        cutmix = K.RandomCutMixV2(data_keys=["input", "class"])

    if args.get_lr_process:
        val_acc_record = []
        test_acc_record = []

    for epoch in range(start_epoch, args.epochs + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch, log)

        if args.get_lr_process:

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, filename=os.path.join(save_dir, 'model.pt'))
            val_acc = evaluation(train_loader, model)
            test_acc = evaluation(test_loader, model)
            val_acc_record.append(val_acc)
            test_acc_record.append(test_acc)
            print(val_acc_record, test_acc_record)
            log.info(
                f'val accuracy = {val_acc:.4f}\t'
                f'test accuracy = {test_acc:.4f}'
            )

        else:
            if epoch % 25 == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, filename=os.path.join(save_dir, 'model.pt'))

                if args.poisoned_class == -1:
                    val_acc = evaluation(train_loader, model)
                    test_acc = evaluation(test_loader, model)
                    log.info(
                        f'val accuracy = {val_acc:.4f}\t'
                        f'test accuracy = {test_acc:.4f}'
                    )
                else:
                    val_acc = classwise_evaluation(train_loader, model, args.num_classes)
                    test_acc = classwise_evaluation(test_loader, model, args.num_classes)
                    overall_val_acc = val_acc.mean()
                    overall_test_acc = test_acc.mean()
                    log.info(
                        f'val accuracy = {val_acc}\t'
                        f'test accuracy = {test_acc}\t'
                        f'overall val accuracy = {overall_val_acc}\t'
                        f'overall test accuracy = {overall_test_acc}'
                    )

    if args.get_lr_process:
        log.info(
            f'val accuracy record = {val_acc_record}\t'
            f'test accuracy record = {test_acc_record}\t'
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

    cudnn.benchmark = True
    main()
