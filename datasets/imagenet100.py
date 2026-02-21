import torch
import os
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image

class ImageNet100Index(ImageFolder):
    def __init__(self, root='./data/imagenet100/', train=True, transform=None, delta: torch.FloatTensor = None, poisoned_class=-1):
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        super().__init__(root, transform=transform)
        self.delta = delta
        self.poisoned_class = poisoned_class


    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.delta is not None:
            if len(self.delta) == 100:
                delta = self.delta[target]
            else:
                delta = self.delta[index]

            if self.poisoned_class != -1:
                assert self.poisoned_class in list(range(100))
                delta = torch.zeros_like(delta)

            delta = delta.mul(255).numpy().transpose(1, 2, 0)
            sample = np.asarray(sample)
            sample = np.clip(sample.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            sample = Image.fromarray(sample, mode='RGB')

        sample = self.transform(sample)
        return sample, target, index