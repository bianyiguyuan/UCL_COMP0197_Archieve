import torch
import torch.nn as nn

class myCrossEntropy(nn.Module):
    """
    Custom Cross-Entropy loss using Binary Cross-Entropy (BCE).

    Methods:
    --------
    forward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor
        Computes the BCE loss between output and target.

    """

    def __init__(self):
        """
        Initializes the loss function as Binary Cross-Entropy (BCE).
        """
        super(myCrossEntropy, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, output, target):
        """
        Computes the BCE loss.

        Parameters:
        -----------
        output : torch.Tensor
            Predicted probabilities of shape (N, *), where N is the batch size.
        target : torch.Tensor
            Ground-truth labels of the same shape as output.

        Returns:
        --------
        torch.Tensor
            Scalar loss value.
        """
        return self.loss(output, target)

class myRootMeanSquare(nn.Module):
    """
    Custom Root Mean Square Error (RMSE) loss.

    Methods:
    --------
    forward(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor
        Computes the RMSE loss between output and target.

    """

    def __init__(self):
        """
        Initializes the loss function as Mean Squared Error (MSE).
        """
        super(myRootMeanSquare, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, output, target):
        """
        Computes the RMSE loss.

        Parameters:
        -----------
        output : torch.Tensor
            Predicted values of shape (N, *), where N is the batch size.
        target : torch.Tensor
            Ground-truth labels of the same shape as output.

        Returns:
        --------
        torch.Tensor
            Scalar loss value.
        """
        return torch.sqrt(self.loss(output, target))
