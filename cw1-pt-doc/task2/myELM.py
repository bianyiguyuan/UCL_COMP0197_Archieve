import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

class MyExtremeLearningMachine(nn.Module):
    """
    Extreme Learning Machine (ELM) with a fixed-weight convolutional layer and a trainable fully connected layer.

    Methods:
    --------
    initialise_fixed_layers(shape: tuple, std: float) -> torch.Tensor
        Initializes fixed convolutional layer weights using a Gaussian distribution.

    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass through the network.

    fit_elm_sgd(train_loader: DataLoader, lr: float = 0.01, epochs: int = 100) -> None
        Trains the fully connected layer using stochastic gradient descent (SGD).

    fit_elm_ls(train_loader: DataLoader, lam: float) -> None
        Trains the fully connected layer using a least-squares closed-form solution.
    """

    def __init__(self, input_channels, num_feature_maps, num_classes, kernel_size=3, std=0.1, seed=42):
        """
        Initialize the ELM model.

        Parameters:
        -----------
        input_channels : int
            Number of input channels (e.g., 3 for RGB images).
        num_feature_maps : int
            Number of feature maps in the convolutional layer.
        num_classes : int
            Number of output classes.
        kernel_size : int, optional
            Size of the convolutional kernel. Default is 3.
        std : float, optional
            Standard deviation for initializing the fixed convolutional layer weights. Default is 0.1.
        seed : int, optional
            Random seed for reproducibility. Default is 42.
        """

        super(MyExtremeLearningMachine, self).__init__()

        self.num_feature_maps = num_feature_maps
        self.conv_layer = nn.Conv2d(input_channels, num_feature_maps, kernel_size, padding=1, bias=False)
        self.conv_layer.weight = nn.Parameter(self.initialise_fixed_layers(self.conv_layer.weight.shape, std), requires_grad=False)
        self.fc_layer = nn.Linear(num_feature_maps * 32 * 32, num_classes)  

    def initialise_fixed_layers(self, shape, std):
        """
        Initialize the fixed convolutional layer weights using a Gaussian distribution.

        Parameters:
        -----------
        shape : tuple
            Shape of the weight tensor.
        std : float
            Standard deviation for random initialization.

        Returns:
        --------
        torch.Tensor
            Initialized weight tensor.
        """
        return torch.randn(shape) * std
    
    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
        --------
        torch.Tensor
            Output logits of shape (batch_size, num_classes).
        """
        x = self.conv_layer(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)
        return x

    def fit_elm_sgd(self, train_loader, lr=0.01, epochs=100):
        """
        Train the fully connected layer using stochastic gradient descent (SGD).

        Parameters:
        -----------
        train_loader : DataLoader
            DataLoader containing the training dataset.
        lr : float, optional
            Learning rate for SGD. Default is 0.01.
        epochs : int, optional
            Number of training epochs. Default is 100.

        Returns:
        --------
        None
        """
        optimizer = optim.SGD(self.fc_layer.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')

    
    def fit_elm_ls(self, train_loader, lam):
        """
        Train the fully connected layer using a least-squares closed-form solution.

        Parameters:
        -----------
        train_loader : DataLoader
            DataLoader containing the training dataset.
        lam : float
            Regularization parameter for least-squares optimization.

        Returns:
        --------
        None
        """
        all_features = []
        all_targets = []
        for images, labels in train_loader:
            with torch.no_grad():
                h = self.conv_layer(images)
                h = F.relu(h)
                h = h.view(h.shape[0], -1)
            all_features.append(h)
            y_onehot = F.one_hot(labels, num_classes=self.fc_layer.out_features).float()
            all_targets.append(y_onehot)

        all_features = torch.cat(all_features, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        I = torch.eye(all_features.shape[1])
        A = all_features.T @ all_features + lam * I
        B = all_features.T @ all_targets

        W = torch.linalg.solve(A, B)

        with torch.no_grad():
            self.fc_layer.weight.copy_(W.T)
            if self.fc_layer.bias is not None:
                self.fc_layer.bias.zero_()
                
   