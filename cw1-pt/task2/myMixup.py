from torchvision.utils import save_image
import torch
import torch.nn as nn

class MyMixUp(nn.Module):
    def __init__(self, alpha=0.4):
        super(MyMixUp, self).__init__()
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def mixup_data(self, x, y):
        """Applies Mixup to the batch"""
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def forward(self, preds, targets_a, targets_b, lam):
        """Compute Mixup loss"""
        return lam * self.criterion(preds, targets_a) + (1 - lam) * self.criterion(preds, targets_b)

    def visualize_mixup(self, dataset):
        """Visualize Mixup samples"""
        images, labels = next(iter(dataset))
        mixed_images, _, _, _ = self.mixup_data(images, labels)
        save_image(mixed_images[:16].detach(), "mixup.png", nrow=4, normalize=True)