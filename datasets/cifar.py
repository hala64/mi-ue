import torch
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
from PIL import Image

class CIFAR10Index(CIFAR10):
    def __init__(self, delta: torch.FloatTensor = None, ratio=1.0, poisoned_class=-1, **kwargs):
        super(CIFAR10Index, self).__init__(**kwargs)
        self.delta = delta

        assert ratio <= 1.0 and ratio > 0
        if self.delta is not None:
            if len(delta) == 10:
                self.delta = self.delta[torch.tensor(self.targets)]
            if delta.shape != self.data.shape:
                self.delta = self.delta.permute(0, 2, 3, 1)
                assert self.delta.shape == self.data.shape

            set_size = int(len(self.data) * ratio)
            if set_size < len(self.data):
                self.delta[set_size:] = 0.0

            if poisoned_class != -1:
                assert poisoned_class in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                for i in range(len(self.data)):
                    if self.targets[i] != poisoned_class:
                        delta[i, :, :] = 0.0


            self.delta = self.delta.mul(255).cpu().numpy()
            self.data = np.clip(self.data.astype(np.float32) + self.delta, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target, idx


class CIFAR100Index(CIFAR100):
    def __init__(self, delta: torch.FloatTensor = None, ratio=1.0, poisoned_class=-1, **kwargs):
        super(CIFAR100Index, self).__init__(**kwargs)
        self.delta = delta

        if self.delta is not None:
            if len(delta) == 100:
                self.delta = self.delta[torch.tensor(self.targets)]
            if delta.shape != self.data.shape:
                self.delta = self.delta.permute(0, 2, 3, 1)
                assert self.delta.shape == self.data.shape
            set_size = int(len(self.data) * ratio)
            if set_size < len(self.data):
                self.delta[set_size:] = 0.0

            if poisoned_class != -1:
                assert poisoned_class in list(range(100))
                for i in range(len(self.data)):
                    if self.targets[i] != poisoned_class:
                        delta[i, :, :] = 0.0

            self.delta = self.delta.mul(255).cpu().numpy()
            self.data = np.clip(self.data.astype(np.float32) + self.delta, 0, 255).astype(np.uint8)


    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target, idx

