import torchvision
#from utils import *
#import skimage
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchvision import transforms
from datasets import CIFAR10Index, ImageNet100Index, CIFAR100Index

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if not os.path.exists('./image'):
    os.makedirs('./image')

def show_image_data_poison(dataset, poison, num=16):
    index = np.random.choice(range(len(poison)), num)
    clean_images = [dataset[i][0] for i in index]
    poisons = [255 / 16 * (poison[i] + 8 / 255).detach().cpu() for i in index]
    poisoned_images = clean_images + [poison[i].cpu() for i in index]
    images = clean_images + poisons + poisoned_images
    grid_images = torchvision.utils.make_grid(images, nrow=8)
    plt.figure(figsize=(32, 32))
    plt.imshow(grid_images.permute(1, 2, 0))
    plt.savefig("./image/miue_cifar10.jpg")
    plt.show()

poison1 = torch.load('./gen/cifar10/resnet18/miue/poison.pt')
print(poison1)
#print(poison1.size())

dataset = CIFAR10Index(root='./data/CIFAR10', train=True, transform=transforms.ToTensor(),
                            delta=None,download=True,poisoned_class=-1)

show_image_data_poison(dataset, poison1)
