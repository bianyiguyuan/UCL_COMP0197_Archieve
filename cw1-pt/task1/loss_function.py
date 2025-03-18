import torch
import torch.nn as nn

class myCrossEntropy(nn.Module):
    def __init__(self):
        super(myCrossEntropy, self).__init__()
        self.loss = nn.BCELoss()


    def forward(self, output, target):
        return self.loss(output, target)
    
class myRootMeanSquare(nn.Module):
    def __init__(self):
        super(myRootMeanSquare, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, output, target):
        return torch.sqrt(self.loss(output, target))