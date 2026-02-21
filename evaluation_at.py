import torch.backends.cudnn as cudnn
import time
from dataloaders import PrepareDataLoaders
from torchvision.models import resnet18, resnet50
from models import vgg19, DenseNet121, WideResNet, SL_ViT
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from utils import *
import argparse


def get_args():
    parser = argparse.ArgumentParser('EVAL-AT')
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vgg19',
                                                                             'densenet121', 'vit', 'wrn34-10'])
    parser.add_argument('--num-classes', default=10, type=int)
    parser.add_argument('--poison-path', type=str, default=None)
    parser.add_argument('--poison-ratio', type=float, default=1.0)
    parser.add_argument('--poison-size', type=int, default=32)

    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet',
                                                                 'miniimagenet','imagenet100'])
    parser.add_argument('--data', type=str, default='data/CIFAR10', help='directory of data set')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', default=128, type=float)
    parser.add_argument('--epsilon', default=8, type=float)
    parser.add_argument('--alpha', default=2, type=float)
    parser.add_argument('--steps', default=10, type=int)
    parser.add_argument('--random-start', default=True)
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--milestones', default=(40, 80))
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--gpu-id', type=str, default='0')
    parser.add_argument('--resume', action='store_true', help='if resume training')
    parser.add_argument('--get-lr-process', action='store_true', help='if get learning process')

    arguments = parser.parse_args()
    arguments.epsilon = arguments.epsilon/255
    arguments.alpha = arguments.alpha/255
    return arguments


def pgd_epoch(loader, model, optimizer, scheduler, epoch, log):
    losses = AverageMeter()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()
    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    start = time.time()
    for data, target, _ in loader:
        data, target = data.cuda(), target.cuda()
        data_time = time.time() - start
        data_time_meter.update(data_time)
        delta = torch.zeros_like(data, requires_grad=True)
        if args.random_start:
            delta.data.uniform_(-args.epsilon, args.epsilon)
        model.eval()
        for _ in range(args.steps):
            loss = F.cross_entropy(model(delta + data), target)
            grad = torch.autograd.grad(loss, [delta])[0].data
            delta.data = delta.data + args.alpha * torch.sign(grad)
            delta.data = torch.clamp(delta.data, -args.epsilon, args.epsilon)
            delta.data = torch.clamp(data.data + delta.data, 0, 1) - data.data
        data = delta + data
        model.train()
        loss = F.cross_entropy(model.train()(data), target)
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



def pgd_inf_test(model, data, target):
    max_loss = torch.zeros_like(target)
    max_delta = torch.zeros_like(data)
    for _ in range(args.restarts):
        delta = torch.zeros_like(data, requires_grad=True)
        if args.random_start:
            delta.data.uniform_(-args.epsilon, args.epsilon)
        for _ in range(args.steps):
            with torch.enable_grad():
                output = model.eval()(delta+data)
                index = output.argmax(1).eq(target)
                if index.sum().item() == 0:
                    break
                loss = F.cross_entropy(output, target)
            grad = torch.autograd.grad(loss, [delta])[0].data

            d = delta[index]
            g = grad[index]
            d = d + args.alpha*torch.sign(g)
            d = torch.clamp(d, -args.epsilon, args.epsilon)
            d = torch.clamp(d+data[index], 0, 1) - data[index]
            delta.data[index] = d

        with torch.no_grad():
            loss = F.cross_entropy(model.eval()(data+delta), target, reduction='none')
            index = loss >= max_loss
            max_delta[index] = delta.data[index]
            max_loss = torch.max(max_loss, loss)
    adv_data = max_delta.data + data.data
    return adv_data


def main():
    if args.seed is not None:
        setup_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    save_dir = os.path.join('eval', args.dataset, args.backbone, 'AT', f'{args.epsilon}', args.experiment)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log = logger(path=save_dir)
    log.info(str(args))

    try:
        poison = torch.load(args.poison_path, map_location='cpu')
        print('poison:', torch.max(poison))
    except:
        poison = None
        print('no poison founded!')


    dataloaders = PrepareDataLoaders(args.dataset, root=args.data, output_size=args.poison_size, for_gen=False,
                                     supervised=True, delta=poison, ratio=args.poison_ratio)
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
    else:
        raise AssertionError('model is not defined')

    if args.dataset == 'imagenet100':
        model = nn.DataParallel(model)

    optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(os.path.join(save_dir, 'model.pt'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim'])
        for i in range(start_epoch - 1):
            scheduler.step()
        log.info(f"RESUME FROM EPOCH {start_epoch-1}")

    if args.get_lr_process:
        val_acc_record = []
        test_acc_record = []


    for epoch in range(start_epoch, args.epochs + 1):
        pgd_epoch(train_loader, model, optimizer, scheduler, epoch, log)

        val_acc = evaluation(train_loader, model)
        test_acc = evaluation(test_loader, model)
        log.info(
            f'val accuracy = {val_acc:.4f}\t'
            f'test accuracy = {test_acc:.4f}'
        )
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
        }, filename=os.path.join(save_dir, 'model.pt'))

        if args.get_lr_process:
            val_acc_record.append(val_acc)
            test_acc_record.append(test_acc)
            print(val_acc_record, test_acc_record)

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
