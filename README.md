# Generative Adversarial Nets — From Scratch in PyTorch

An implementation of the original GAN paper, [*Generative Adversarial Nets*](https://arxiv.org/abs/1406.2661) (Goodfellow et al., 2014), trained on MNIST.

This is the first project in a series where I read the landmark papers in AI and implement each model myself.

![Samples after 50 epochs](samples/epoch_49.png)

*64 samples from the generator after 50 epochs.*

## The idea

Two networks play a minimax game:

- **Generator $G$** takes random noise $z$ and produces a fake image $G(z)$.
- **Discriminator $D$** takes an image and outputs the probability that it's real.

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p_z}\big[\log(1 - D(G(z)))\big]
$$

$D$ tries to tell real from fake, and $G$ tries to fool it. The paper proves that with enough capacity, the game's global optimum is $p_g = p_{\text{data}}$: the generator reproduces the data distribution exactly, and $D$ is reduced to guessing $\tfrac{1}{2}$ everywhere. With an optimal discriminator, training $G$ is equivalent to minimizing the Jensen–Shannon divergence between the real and generated distributions.

## Architecture

| | Generator | Discriminator |
|---|---|---|
| Input | $z \sim \mathcal{U}(-1, 1)^{100}$ | image, flattened to 784 |
| Hidden layers | $1200 \to 1200$, LeakyReLU(0.2) | $240 \to 240$, LeakyReLU(0.2) |
| Output | 784 pixels, tanh | 1 probability, sigmoid |
| Regularization | — | Dropout: 0.2 on input, 0.5 on hidden |
| Parameters | 2,503,984 | 246,481 |

**Training:** Algorithm 1 from the paper with $k = 1$ (one $D$ step per $G$ step), batch size 100, 50 epochs, Adam with learning rate $2 \times 10^{-4}$ and $\beta = (0.5, 0.999)$. The generator uses the paper's non-saturating loss: it maximizes $\log D(G(z))$ instead of minimizing $\log(1 - D(G(z)))$, which gives much stronger gradients early in training.

### Differences from the paper

The layer sizes match the paper's MNIST config, but some training choices are modern defaults:

| Paper | This implementation |
|---|---|
| SGD + momentum, lr $0.1$ | Adam, lr $2 \times 10^{-4}$ |
| ReLU in $G$ | LeakyReLU in $G$ |
| Sigmoid output, data in $[0, 1]$ | tanh output, data in $[-1, 1]$ |
| Maxout units in $D$ | LeakyReLU in $D$ |

## What broke, and how I fixed it

The first versions failed, and debugging them taught me more than the working version did.

**Attempt 1: paper-style output (ReLU + sigmoid in $G$), no dropout.**
The discriminator won from the very first epoch. By epoch 10, $D(G(z)) \approx 10^{-5}$, and by epoch 49, $D(x) = 1.0$ and $D(G(z)) \approx 10^{-20}$, with $\mathcal{L}_G \approx 46 = -\log(10^{-20})$. Once $D$ is that confident, $G$ receives almost no gradient and can't recover.

**Attempt 2: added the paper's dropout to $D$.**
Barely helped. $D$ still won outright, even with half its network randomly disabled. That suggested the problem was on $G$'s side.

**Diagnosis: mode collapse.** I logged statistics of $G$'s output across a fixed batch of 64 noise vectors. The mean brightness matched real MNIST, but the per-pixel standard deviation across samples was $0.006$, versus about $0.3$ for real data. $G$ was producing nearly the same image for every $z$, the "Helvetica scenario" the paper warns about in Section 6. With every fake identical, $D$ only had to learn to reject one picture.

**Attempt 3: LeakyReLU + tanh in $G$, data scaled to $[-1, 1]$.** This fixed it. $D(x)$ settled around $0.6$ and $D(G(z))$ around $0.3$, with neither player running away with the game, and samples became diverse and recognizable.

- **LeakyReLU** keeps $G$'s hidden units from dying, so variation in $z$ survives through every layer to the output.
- **tanh** has a maximum slope of $1$ versus $0.25$ for sigmoid, so $G$'s output layer passes back stronger gradients.

Since both changed at once, I haven't yet isolated which one mattered more.

## Results

The final samples cover all ten digits with varied styles, comparable to the paper's Figure 2a. Remaining flaws: faint speckle in the backgrounds and some malformed digits, which is typical for fully connected GANs (convolutional architectures like DCGAN clean this up).

The `samples/` folder contains the same 64 noise vectors rendered after every epoch, so you can watch the digits form.

## Project structure

```
data.py      # MNIST loading, normalized to [-1, 1]
models.py    # Generator, Discriminator, noise sampler
train.py     # Training loop (Algorithm 1), saves samples and checkpoints
samples/     # Generated grid after each epoch
```

## Running it

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
python train.py
```

MNIST downloads automatically on the first run. Training takes a few minutes per epoch on an Apple Silicon MacBook (uses the `mps` backend if available, otherwise CPU).

## What I learned

- **GAN failures are silent.** Nothing crashes; the losses just drift to extremes. Logging $D(x)$, $D(G(z))$, and sample diversity was far more useful than watching the losses.
- **Balance matters more than strength.** $D$ needs to be good, but not so good that its feedback to $G$ vanishes.
- **Paper details exist for a reason.** Choices that look minor, like dropout and output activations, can decide whether training works at all.

## Next steps

- [ ] Ablation: test LeakyReLU and tanh separately to see which fixed the collapse
- [ ] Reproduce Figure 3: interpolate between two points in $z$-space
- [ ] Faithful replication with maxout, SGD, and the paper's hyperparameters

## Reference

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). *Generative Adversarial Nets.* [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)