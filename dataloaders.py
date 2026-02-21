import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import CIFAR10Index, CIFAR100Index, TinyImageNetIndex, MiniImageNetIndex, ImageNet100Index
from cutout import Cutout
from PIL import Image
import numpy as np


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=8.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):

        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, c))
        img = N + img
        img[img > 255] = 255
        img[img < 0] = 0
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


class PrepareDataLoaders:
    def __init__(self, dataset: str, root: str, output_size: int, for_gen: bool, supervised: bool,
                 post_aug: bool = False, delta: torch.FloatTensor = None, ratio=1.0,
                 cutout=False,  gaussian_smooth=False, random_noise=False, poisoned_class=-1, data_aug=True):
        self.dataset = dataset
        self.root = root
        self.for_gen = for_gen
        self.supervised = supervised
        self.output_size = output_size
        self.post_aug = post_aug
        self.data_aug = data_aug
        self.delta = delta
        self.ratio = ratio
        self.cutout = cutout
        self.gaussian_smooth = gaussian_smooth
        self.random_noise = random_noise
        self.poisoned_class = poisoned_class

    def get_train_loader(self, batch_size: int, num_workers: int):
        dataset = self._get_train_set()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        return dataloader

    def get_test_loader(self, batch_size: int, num_workers: int):
        dataset = self._get_test_set()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return dataloader

    def get_val_loader(self, batch_size: int, num_workers: int):
        dataset = self._get_val_set()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        return dataloader

    def _get_train_transform_for_generation(self):
        if self.supervised:
            if not self.post_aug:
                transform = transforms.ToTensor()
            elif self.dataset == 'imagenet100':
                if self.data_aug:
                    transform = transforms.Compose(
                        [transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.RandomHorizontalFlip(p=0.5),
                         transforms.ToTensor()])
                else:
                    transform = transforms.Compose(
                        [transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor()])
            else:
                if self.data_aug:
                    s = 0.0
                    transform = transforms.Compose([transforms.RandomResizedCrop(self.output_size, scale=(1 - 0.9 * s, 1.0)),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomApply(
                                                        [transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)],
                                                        p=0.8 * s),
                                                    transforms.RandomGrayscale(p=0.2 * s),
                                                    transforms.ToTensor()])
                else:
                    transform = transforms.ToTensor()
        else:
            transform = transforms.ToTensor()
        return transform

    def _get_train_transform_for_evaluation(self):
        if self.supervised:
            if self.dataset in ['cifar10', 'cifar100']:
                if self.gaussian_smooth:
                    trans = [transforms.GaussianBlur(kernel_size=3)]
                elif self.random_noise:
                    trans = [AddGaussianNoise(0, 8, 1)]
                else:
                    trans = []
                trans += [transforms.RandomCrop(32, 4),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor()]
                if self.cutout and self.dataset != 'tinyimagenet':
                    from cutout import Cutout
                    trans.append(Cutout())

                transform = transforms.Compose(trans)
            elif self.dataset == 'miniimagenet':
                transform = transforms.Compose([transforms.RandomCrop(84, 10),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor()])
            elif self.dataset == 'tinyimagenet':
                from torchtoolbox.transform import Cutout
                transform = transforms.Compose([transforms.RandomCrop(64, 8),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,hue=0.2),
                                                Cutout(),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor()])
            elif self.dataset == 'imagenet100':
                transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor(),
                                                ])
            else:
                raise AssertionError('dataset is not defined')
        else:
            if self.gaussian_smooth:
                trans = [transforms.GaussianBlur(kernel_size=3)]
            elif self.random_noise:
                trans = [AddGaussianNoise(0, 8, 1)]
            else:
                trans = []
            trans += [transforms.RandomResizedCrop(self.output_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                                            p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor()]
            if self.cutout:
                from cutout import Cutout
                trans.append(Cutout())
            transform = transforms.Compose(trans)
        return transform

    def _get_test_transform(self):
        if self.dataset in ['cifar10', 'cifar100', 'tinyimagenet', 'miniimagenet']:
            transform = transforms.ToTensor()
        elif self.dataset == 'imagenet100':
            transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            ])
        else:
            raise AssertionError('dataset is not defined')
        return transform

    def _make_dataset(self, train: bool, transform, delta):
        if self.dataset == 'cifar10':
            data_set = CIFAR10Index(root=self.root, train=train, transform=transform, delta=delta,
                                        ratio=self.ratio, download=True,
                                        poisoned_class=self.poisoned_class)
        elif self.dataset == 'cifar100':
            data_set = CIFAR100Index(root=self.root, train=train, transform=transform,
                                         delta=delta, ratio=self.ratio, download=True,
                                         poisoned_class=self.poisoned_class)
        elif self.dataset == 'tinyimagenet':
            data_set = TinyImageNetIndex(root=self.root, train=train, transform=transform, delta=delta)
        elif self.dataset == 'miniimagenet':
            data_set = MiniImageNetIndex(root=self.root, train=train, transform=transform, delta=delta)
        elif self.dataset == 'imagenet100':
            data_set = ImageNet100Index(train=train, transform=transform, delta=delta, poisoned_class=self.poisoned_class)

        else:
            raise AssertionError('dataset is not defined')
        return data_set

    def _get_train_set(self):
        if self.for_gen:
            transform = self._get_train_transform_for_generation()
        else:
            transform = self._get_train_transform_for_evaluation()


        dataset = self._make_dataset(train=True, transform=transform, delta=self.delta)
        return dataset

    def _get_test_set(self):
        transform = self._get_test_transform()
        dataset = self._make_dataset(train=False, transform=transform, delta=None)
        return dataset

    def _get_val_set(self):
        transform = self._get_test_transform()
        dataset = self._make_dataset(train=True, transform=transform, delta=self.delta)
        return dataset








class APDataLoaders:
    def __init__(self, dataset: str, root: str, output_size: int, ref_mode='standard', rrc=0.0, cj=0.0, rg=0.0):
        self.dataset = dataset
        self.root = root
        self.ref_mode = ref_mode
        self.output_size = output_size
        self.rrc = rrc
        self.cj = cj
        self.rg = rg

    def get_train_loader(self, batch_size: int, num_workers: int):
        dataset = self._get_train_set()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        return dataloader

    def get_test_loader(self, batch_size: int, num_workers: int):
        dataset = self._get_test_set()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return dataloader

    def get_val_loader(self, batch_size: int, num_workers: int):
        dataset = self._get_val_set()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        return dataloader
        
    def _get_train_transform_for_generation(self):
        if self.ref_mode == 'standard':
            transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor()])
        else: 
            transform = transforms.Compose([transforms.RandomResizedCrop(self.output_size, scale=(1 - 0.9 * self.rrc, 1.0)),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomApply(
                                                   [transforms.ColorJitter(0.4 * self.cj, 0.4 * self.cj, 0.4 * self.cj, 0.1 * self.cj)],
                                                   p=0.8 * self.cj),
                                                transforms.RandomGrayscale(p=0.2 * self.rg),
                                                transforms.ToTensor()])
        return transform

    def _get_test_transform(self):
        if self.dataset in ['cifar10', 'cifar100', 'tinyimagenet', 'miniimagenet']:
            transform = transforms.ToTensor()
        elif self.dataset == 'imagenet100':
            transform = transforms.Compose([transforms.ToTensor(),
                                            #lambda x: x.to(torch.float16)
                                            ])
        else:
            raise AssertionError('dataset is not defined')
        return transform

    def _make_dataset(self, train: bool, transform):
        if self.dataset == 'cifar10':
            data_set = CIFAR10Index(root=self.root, train=train, transform=transform, download=True)
        elif self.dataset == 'cifar100':
            data_set = CIFAR100Index(root=self.root, train=train, transform=transform, download=True)
        elif self.dataset == 'tinyimagenet':
            data_set = TinyImageNetIndex(root=self.root, train=train, transform=transform)
        elif self.dataset == 'miniimagenet':
            data_set = MiniImageNetIndex(root=self.root, train=train, transform=transform)
        elif self.dataset == 'imagenet100':
            data_set = ImageNet100Index(train=train, transform=transform)
        else:
            raise AssertionError('dataset is not defined')
        return data_set

    def _get_train_set(self):
        transform = self._get_train_transform_for_generation()

        dataset = self._make_dataset(train=True, transform=transform)
        return dataset

    def _get_test_set(self):
        transform = self._get_test_transform()
        dataset = self._make_dataset(train=False, transform=transform)
        return dataset

    def _get_val_set(self):
        transform = self._get_test_transform()
        dataset = self._make_dataset(train=True, transform=transform)
        return dataset
