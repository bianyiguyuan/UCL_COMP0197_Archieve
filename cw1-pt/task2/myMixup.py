from torchvision.utils import save_image
from dataLoader import load_data
import torch
import torch.nn as nn
import numpy as np
import os

class MyMixUp(nn.Module):
    def __init__(self, alpha=0.4):
        super(MyMixUp, self).__init__()
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def mixup_data(self, x, y):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


    def forward(self, preds, targets_a, targets_b, lam):
        loss_a = self.criterion(preds, targets_a)  # shape: (batch_size,)
        loss_b = self.criterion(preds, targets_b)  # shape: (batch_size,)
        loss = (lam * loss_a + (1 - lam) * loss_b).mean()
        return loss

    def visualize_mixup(self, dataset):
        """Visualize Mixup samples"""
        images, labels = next(iter(dataset))
        mixed_images, _, _, _ = self.mixup_data(images, labels)
        script_dir = os.path.dirname(os.path.abspath(__file__))  
        save_path = os.path.join(script_dir, "mixup.png")
        save_image(mixed_images[:16].detach(), save_path, nrow=4, normalize=True)

if __name__ == '__main__':
    train_loader, test_loader, classes = load_data()
    mixup = MyMixUp(alpha=0.7)
    mixup.visualize_mixup(train_loader)
