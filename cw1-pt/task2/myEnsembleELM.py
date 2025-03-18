import torch
import torch.nn as nn
import random
import numpy as np
from myELM import MyExtrmeLearningMachine

class MyEnsembleELM(nn.Module):
    def __init__(self, n_models, input_channels, num_feature_maps, num_classes, seed=42, aggregation='mean'):
        super(MyEnsembleELM, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.n_models = n_models
        self.aggregation = aggregation
        self.models = nn.ModuleList([MyExtrmeLearningMachine(input_channels, num_feature_maps, num_classes) for _ in range(n_models)])

    def forward(self, x):
        outputs = torch.stack([model(x) for model in self.models], dim=0)
        if self.aggregation == 'mean':
            return outputs.mean(dim=0)
        elif self.aggregation == 'vote':
            return torch.mode(outputs.argmax(dim=-1), dim=0).values
        
    
