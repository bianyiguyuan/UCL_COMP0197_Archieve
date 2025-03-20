import torch
import torch.nn as nn
import random
import numpy as np
from myELM import MyExtremeLearningMachine

class MyEnsembleELM(nn.Module):
    """
    An ensemble of Extreme Learning Machines (ELMs) that aggregates predictions from multiple models.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor
        Computes the forward pass by aggregating predictions from multiple ELM models.
    """

    def __init__(self, n_models, input_channels, num_feature_maps, num_classes, seed=42, aggregation='mean'):
        """
        Initialize the ensemble of ELM models.

        Parameters:
        -----------
        n_models : int
            Number of individual ELM models in the ensemble.
        input_channels : int
            Number of input channels (e.g., 3 for RGB images).
        num_feature_maps : int
            Number of feature maps in the convolutional layer of each ELM model.
        num_classes : int
            Number of output classes.
        seed : int, optional
            Random seed for reproducibility. Default is 42.
        aggregation : str, optional
            Method for aggregating predictions from multiple models. Options:
            - 'mean' : Averages the outputs of all models.
            - 'vote' : Uses majority voting on the predicted class labels.
            - 'softmax_mean' : Averages the softmax probabilities from all models.
            Default is 'mean'.
        """

        super(MyEnsembleELM, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.n_models = n_models
        self.aggregation = aggregation
        self.models = nn.ModuleList([MyExtremeLearningMachine(input_channels, num_feature_maps, num_classes) for _ in range(n_models)])

    def forward(self, x):
        """
        Forward pass through the ensemble, aggregating predictions from all models.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
        --------
        torch.Tensor
            Aggregated output logits or predicted class labels, depending on the aggregation method.
        """
        
        outputs = torch.stack([model(x) for model in self.models], dim=0)
        if self.aggregation == 'mean':
            return outputs.mean(dim=0)
        elif self.aggregation == 'vote':
            mode_values, _ = torch.mode(outputs.argmax(dim=-1), dim=0)
            return mode_values
        elif self.aggregation == 'softmax_mean':
            return torch.softmax(outputs, dim=-1).mean(dim=0)


        
    
