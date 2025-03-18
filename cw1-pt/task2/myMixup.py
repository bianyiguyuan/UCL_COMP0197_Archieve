import torch
import numpy as np
import random
from torchvision.utils import save_image

class MyMixUp:
    def __init__(self, alpha = 0.2, seed = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.alpha = alpha

    def mixup_indices(self, x):
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        lam = np.random.beta(self.alpha, self.alpha)
        return index, lam
    
    def visualize_mixup(self, dataset):
        images, labels = next(iter(dataset))
        mixed_images = self.mixup(images, labels)[0]
        save_image(mixed_images[:16], "mixup.png", nrow=4, normalize=True)
        