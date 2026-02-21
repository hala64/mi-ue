import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import os
from kornia import augmentation as KA
from torch.nn import functional as F
import random


class Augment(object):
    def __init__(self, size=32):
        self.rhf = KA.RandomHorizontalFlip(p=0.5)
        self.rc = KA.RandomCrop((size, size), int(size/8))
    
    def aug_standard(self, data):
        img = self.rhf(data)
        img = self.rc(img)
        return img

    def aug_id(self, data):
        return data



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class logger(object):
    def __init__(self, path, name='log.txt'):
        self.path = path
        self.name = name

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, self.name), 'a') as f:
            f.write(msg + "\n")


def pair_cosine_similarity(x, y=None, eps=1e-8):
    if (y == None):
        n = x.norm(p=2, dim=1, keepdim=True)
        return (x @ x.t()) / (n * n.t()).clamp(min=eps)
    else:
        n1 = x.norm(p=2, dim=1, keepdim=True)
        n2 = y.norm(p=2, dim=1, keepdim=True)
        return (x @ y.t()) / (n1 * n2.t()).clamp(min=eps)



def save_checkpoint(state, filename):
    torch.save(state, filename)

def cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)


def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * (step + 1) / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step -
                                                             warmup_steps) / (total_steps - warmup_steps) * np.pi))
        
    return lr


def cosine_tempering(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * (step + 1) / warmup_steps
    else:
        lr = lr_max - (lr_max - lr_min) * 0.5 * (1 + np.cos((step -
                                                             warmup_steps) / (total_steps - warmup_steps) * np.pi))
        
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def classwise_accuracy(output, target, num_classes, correct_per_class, total_per_class):
    """Updates the correct and total predictions for each class."""
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()

    for i in range(pred.size(1)):
        actual_class = target[i]
        predicted_class = pred[0, i]
        total_per_class[actual_class] += 1
        if actual_class == predicted_class:
            correct_per_class[actual_class] += 1


def classwise_evaluation(loader, model, num_classes=10):
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)

    for i, (data, target, _) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            outputs = model.eval()(data)

        classwise_accuracy(outputs.data, target, num_classes, correct_per_class, total_per_class)

    # Avoid division by zero
    class_acc = 100.0 * correct_per_class / total_per_class
    class_acc[total_per_class == 0] = 0  # Set accuracy to 0 where there are no samples

    return class_acc


def evaluation(loader, model):
    top1 = AverageMeter()
    for i, (data, target, _) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            outputs = model.eval()(data)
        prec1 = accuracy(outputs.data, target)[0]
        top1.update(prec1.item(), len(data))
    return top1.avg


def setup_seed(seed):
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Numpy
    np.random.seed(seed)
    # Python
    random.seed(seed)


def nt_xent(x, y, t=0.1):
    sim_matrix = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)

    sim_matrix = torch.exp(sim_matrix / t)

    mask = y.unsqueeze(1) == y.unsqueeze(0)

    sim_matrix = sim_matrix + 1e-8

    pos_sum = torch.where(mask, sim_matrix, torch.zeros_like(sim_matrix)).sum(dim=1)

    all_sum = sim_matrix.sum(dim=1)

    loss = -torch.log(pos_sum / all_sum).mean()

    return loss


def l2sim(x, y):
    feat_dim = len(x[0])
    y = y.cpu()
    results = []
    norms = torch.linalg.norm(x[:, None, :] - x, dim=2) # 128*128
    for i in range(len(x)):
        index = np.where(y == y[i])[0].tolist()
        temp1 = norms[i,index]
        raw_result = temp1.sum() / feat_dim
        results.append(torch.log(1 + raw_result))
    return sum(results) / len(results)
