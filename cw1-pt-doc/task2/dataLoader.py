import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def load_data(batch_size=32):
    """
    Load the CIFAR-10 dataset and split it into training and testing sets.

    Parameters:
    batch_size : int, optional
        Number of samples per batch in the DataLoader. Default is 32.

    Returns:
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the testing dataset.
    classes : tuple
        Tuple containing the names of the 10 CIFAR-10 classes.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))  
    test_size = len(dataset) - train_size 

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes


