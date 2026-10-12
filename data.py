# This file will hold everything about loading our dataset.
# In our case, we will be using MNIST

import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def get_device():
    """
    This is a helper function that returns the best
    Available device on Macbook. Apple GPU would probably
    be the best as my computer has an M2
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_dataloader(batch_size=100):
    """
    Here we download the MNIST training set,
    convert to tesnors and pass it to DataLoader to have
    ready for training
    """
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader

def show_grid(images, nrow=4):
    grid = make_grid(images, nrow=nrow)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    device = get_device()
    print(f"The device is {device}")

    loader = get_dataloader()
    # will report total number of samples in the dataset
    print(f"Size of dataset is {len(loader.dataset)}")

    # iter makes the loader iterable. Next pulls out the first batch
    # in order to inspect without a loop
    # Assuming batch_size of 100 from default, we get the following
    # images shape will be (100, 1, 28, 28)
    # labels will have shape (100,)
    images, labels = next(iter(loader))
    print(f"Batch shape is {images.shape}")
    print("Pixel range:", images.min().item(), "to", images.max().item())

    flat = images.view(images.size(0), -1)
    print(f"Flattened shape is {flat.shape}")

    show_grid(images[:16])
