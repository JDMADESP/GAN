import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

from data import get_device, get_dataloader
from models import Generator, Discriminator, sample_noise

# How many times model will see the entire training dataset
EPOCHS = 50
# how many images are processed together at a time
BATCH_SIZE = 100
# learning rate, controls step size of optimizer when updating weights
LR = 2e-4
# two momentum coefficients used by the Adam optimizer
BETAS = (0.5, 0.999)

def main():
    torch.manual_seed(0)
    device = get_device()
    loader = get_dataloader(BATCH_SIZE)

    G = Generator().to(device)
    D = Discriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)
    bce = nn.BCELoss()

    fixed_z = sample_noise(64, device)
    os.makedirs("samples", exist_ok=True)

    for epoch in range(EPOCHS):
        for real, _ in loader:
            real = real.to(device)
            m = real.size(0)

            # Train D, want to maximize log D(x) + log(1 - D(G(z)))
            z = sample_noise(m, device)

            # Generate fakes, but detach so G is not updated yet
            fake = G(z).detach()
            d_real = D(real)
            d_fake = D(fake)
            loss_D = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))

            opt_D.zero_grad()
            # Compute gradients for D’s parameters only
            loss_D.backward()
            # Update D’s weights
            opt_D.step()

            # Train G, want to maximize log D(G(z))
            z = sample_noise(m, device)
            d_fake = D(G(z))
            loss_G = bce(d_fake, torch.ones_like(d_fake))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f" Epoch: {epoch} | Loss_D: {loss_D} | Loss_G: {loss_G} | D(x): {d_real.mean().item()} | D(G(z)): {d_fake.mean().item()}")

        with torch.no_grad():
            samples = G(fixed_z).view(-1, 1, 28, 28)
    #         print(f"  G out: min {samples.min():.3f} | max {samples.max():.3f} | "
    #   f"mean {samples.mean():.3f} | diversity {samples.std(dim=0).mean():.3f}")
        save_image(samples * 0.5 + 0.5, f"samples/epoch_{epoch}.png", nrow=8)

        torch.save({"G": G.state_dict(), "D": D.state_dict(),
                    "epoch": epoch}, "checkpoint.pt")

if __name__ == "__main__":
    main()