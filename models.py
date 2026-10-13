import torch
import torch.nn as nn
from data import get_dataloader

NOISE_DIM = 100

# G will learn to map randomness to the distribution picked
# for the input images
def sample_noise(m, device):
    """
    Generates a batch of random noise vectors, used
    as input to a GAN generator.
    It is a source of randomness
    """
    return torch.rand(m, NOISE_DIM, device=device) * 2 - 1

# First dimension of batch never changes, each row is one sample
# nn.Linear(in_features, our_features) stores 2 learnable tensors
#   - weight matrix W of shape (out_features, in_features)
#   - a bias vector b of shape (out_features,)
# For single input vector z in R^{100}, first layer computes:
#   h = Wz + b
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, 1200),
            nn.LeakyReLU(0.2),
            nn.Linear(1200, 1200),
            nn.LeakyReLU(0.2),
            nn.Linear(1200, 784),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # using LeakyRelu because when you
        # train on G, the loss depends on D(G(z)) so gradient
        # goes backward through every layer of D.
        # Add dropout to prevent discriminator from getting too strong,
        # too fast
        self.net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(784, 240),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(240, 240),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5), 
            nn.Linear(240, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # keeps the batch dimension (x.size(0) = 100)
        # and merges everything else into one. If we have
        # (100, 1, 28, 28) then changes it to (100, 784)
        # if we already have (100, 784), it keeps it the same
        x = x.view(x.size(0), -1)
        return self.net(x)

if __name__ == "__main__":
    device = torch.device("cpu")
    G = Generator().to(device)
    D = Discriminator().to(device)

    real, _ = next(iter(get_dataloader()))
    real = real.to(device)
    fake = G(sample_noise(100, device))

    # z = sample_noise(100, device)
    # fake = G(z)
    # print("z shape:", z.shape)
    # print("z range:", z.min().item(), "to", z.max().item())
    # # should be (100, 784) since we are using a batch of 100 and producing
    # # 784 pixels
    # print("G(z) shape:", fake.shape)
    # # using sigmoid output function so should be between o and 1
    # print("G(z) range:", fake.min().item(), "to", fake.max().item())

    d_real = D(real)
    d_fake = D(fake)
    print("real shape:", real.shape, "-> D(x) shape:", d_real.shape)
    print("fake shape:", fake.shape, "-> D(G(z)) shape:", d_fake.shape)
    print("mean D(x):", d_real.mean().item())
    print("mean D(G(z)):", d_fake.mean().item())

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"G params: {g_params:,}   D params: {d_params:,}")