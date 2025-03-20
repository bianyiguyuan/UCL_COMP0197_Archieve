from torchvision.utils import save_image
from dataLoader import load_data
import torch
import torch.nn as nn
import numpy as np
import os

class MyMixUp(nn.Module):
    """
    Implementation of the MixUp data augmentation technique for image classification.

    Methods:
    --------
    mixup_data(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
        Mixes up input images and labels using a lambda factor drawn from a Beta distribution.

    mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor
        Computes the MixUp loss by interpolating between two different label losses.

    forward(preds: torch.Tensor, targets_a: torch.Tensor, targets_b: torch.Tensor, lam: float) -> torch.Tensor
        Applies MixUp loss calculation during training.

    visualize_mixup(dataset: DataLoader) -> None
        Saves a visualization of 16 MixUp-augmented images to 'mixup.png'.
    """

    def __init__(self, alpha=0.4):
        """
        Initialize the MixUp module.

        Parameters:
        -----------
        alpha : float, optional
            The alpha parameter for the Beta distribution controlling mixup strength. Default is 0.4.
        """

        super(MyMixUp, self).__init__()
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def mixup_data(self, x, y):
        """
        Generate MixUp-augmented data.

        Parameters:
        -----------
        x : torch.Tensor
            Input images of shape (batch_size, channels, height, width).
        y : torch.Tensor
            Target labels of shape (batch_size,).

        Returns:
        --------
        mixed_x : torch.Tensor
            Mixed images after applying MixUp.
        y_a : torch.Tensor
            First set of labels before MixUp.
        y_b : torch.Tensor
            Second set of shuffled labels.
        lam : float
            Mixing factor from the Beta distribution.
        """
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
        """
        Compute the MixUp loss.

        Parameters:
        -----------
        criterion : nn.Module
            The loss function (e.g., CrossEntropyLoss).
        pred : torch.Tensor
            Model predictions of shape (batch_size, num_classes).
        y_a : torch.Tensor
            First set of labels before MixUp.
        y_b : torch.Tensor
            Second set of shuffled labels.
        lam : float
            Mixing factor.

        Returns:
        --------
        torch.Tensor
            The computed MixUp loss.
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


    def forward(self, preds, targets_a, targets_b, lam):
        """
        Compute the MixUp loss during training.

        Parameters:
        -----------
        preds : torch.Tensor
            Model predictions of shape (batch_size, num_classes).
        targets_a : torch.Tensor
            First set of labels.
        targets_b : torch.Tensor
            Second set of shuffled labels.
        lam : float
            Mixing factor.

        Returns:
        --------
        torch.Tensor
            The computed loss.
        """
        loss_a = self.criterion(preds, targets_a)  # shape: (batch_size,)
        loss_b = self.criterion(preds, targets_b)  # shape: (batch_size,)
        loss = (lam * loss_a + (1 - lam) * loss_b).mean()
        return loss

    def visualize_mixup(self, dataset):
        """
        Generate and save a visualization of MixUp-augmented images.

        Parameters:
        -----------
        dataset : DataLoader
            The dataset to sample images from.

        Returns:
        --------
        None
        """
        images, labels = next(iter(dataset))
        mixed_images, _, _, _ = self.mixup_data(images, labels)
        script_dir = os.path.dirname(os.path.abspath(__file__))  
        save_path = os.path.join(script_dir, "mixup.png")
        save_image(mixed_images[:16].detach(), save_path, nrow=4, normalize=True)

if __name__ == '__main__':
    train_loader, test_loader, classes = load_data()
    mixup = MyMixUp(alpha=0.7)
    mixup.visualize_mixup(train_loader)
