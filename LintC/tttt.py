# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['axes.grid'] = False

import math
import numpy as np
from scipy import stats
import time
import itertools

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

from pathlib import Path

HOME = Path.home()
MNIST_PATH = HOME / 'data' / 'mnist'

NUM_CLASSES = 10
CHANNELS = 1
HEIGHT = 28
WIDTH = 28

Z_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_STEPS = 5000

# We're going to load the official MNIST train set and use
# *only* the images (not the labels).
official_mnist_train = torchvision.datasets.MNIST(str(MNIST_PATH), train=True, download=True)
train_images = official_mnist_train.train_data.numpy().astype(np.float32)

print(train_images.shape)

train_images -= train_images.mean(axis=(1, 2), keepdims=True)
train_images /= train_images.std(axis=(1, 2), keepdims=True)

print(train_images[:3].mean(axis=(1, 2)))
print(train_images[:3].std(axis=(1, 2)))

"""Let's view a 5 x 5 grid of the first 25 images."""

image_grid = train_images[:25].reshape(5, 5, HEIGHT, WIDTH)
image_grid = np.concatenate(image_grid, axis=1)
image_grid = np.concatenate(image_grid, axis=1)
plt.imshow(image_grid)
plt.grid(False)

"""Let's convert the images to vectors by flattening them, as we have done in the past:"""

train_vectors = train_images.reshape(-1, HEIGHT * WIDTH)

"""Now let's define a batch function for retrieving examples (which is a modified version of what you've written in the past):"""


def batch(batch_size):
    """Create a random batch of examples.

    Args:
      batch_size: An integer.

    Returns:
      input_batch: A Variable of floats with shape [batch_size, num_features]
    """
    random_ind = np.random.choice(train_vectors.shape[0], size=batch_size, replace=False)
    input_batch = train_vectors[random_ind]
    input_batch = Variable(torch.from_numpy(input_batch))
    return input_batch


"""**Modify the following cell to complete the decoder. It should**

1. Map from our low-dimensional $\zb$ to a hidden layer of 128 units
2. Apply a ReLU nonlinearity
3. Map to a hidden layer of 512 hidden units
4. Apply a ReLU nonlinearity
5. Map to $\mub_x$ (which resides in $\mathbb{R}^{784}$)
6. (In addition, we return a fixed $\sigmab_x$, but this is already in place for you.)
"""


class VAEDecoder(torch.nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        # TODO
        self.fc1 = torch.nn.Linear(self.z_dim, 128)
        self.fc2 = torch.nn.Linear(128, 512)
        self.fc3 = torch.nn.Linear(512, 784)

    def forward(self, z):
        # TODO
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        mu_x = self.fc3(x)
        sigma_x = Variable(mu_x.data.new(mu_x.shape).fill_(0.1))
        return mu_x, sigma_x

    def log_p_x_given_z(self, mu_x, sigma_x, x):
        dist = torch.distributions.Normal(mu_x, sigma_x)
        return dist.log_prob(x).sum()



class VAEEncoder(torch.nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.fc1 = torch.nn.Linear(784, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, z_dim)

        # TODO

    def forward(self, x):
        # TODO
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu_z = self.fc3(x)
        sigma_z = Variable(mu_z.data.new(mu_z.shape).fill_(0.1))
        sigma_z = F.softplus(sigma_z)
        return mu_z, sigma_z

    def rsample(self, mu_z, sigma_z):
        # If we truly sample, we break our (deterministic) computation graph.
        # This is the reparameterization trick: we sample from a standard
        # normal and transform it with mu_z and sigma_z so that we have
        # a well-defined computation graph (on which we can run backprop).
        standard_normal_samples = Variable(mu_z.data.new(mu_z.shape).normal_())
        scaled_samples = sigma_z * standard_normal_samples
        shifted_scaled_samples = scaled_samples + mu_z
        return shifted_scaled_samples

    def kl(self, mu_z, sigma_z):
        # This is the KL divergence KL( N(mu_z, diag(sigma_z)) || N(0, I) ) in closed form.
        # (It's not difficult to derive; try it!)
        kl_q_z_p_z = 0.5 * torch.sum(-2 * torch.log(sigma_z) - 1 + sigma_z ** 2 + mu_z.pow(2))
        return kl_q_z_p_z


"""We'll proceed as normal, creating our models and our optimizer..."""

encoder = VAEEncoder(Z_DIM)
decoder = VAEDecoder(Z_DIM)

params = list(itertools.chain(encoder.parameters(), decoder.parameters()))
optimizer = torch.optim.Adam(params, LEARNING_RATE)




def train(samples):
    mu_z, sigma_z = encoder(samples)
    z = encoder.rsample(mu_z, sigma_z)
    mu_x, sigma_x = decoder(z)
    neg_log_p_given_x = -decoder.log_p_x_given_z(mu_x, sigma_x, samples) / samples.shape[0]
    kl = encoder.kl(mu_z, sigma_z) / samples.shape[0]
    loss = neg_log_p_given_x + kl
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return neg_log_p_given_x.data[0], kl.data[0], loss.data[0]


"""Finally we can train; we'll keep track of the two components of our loss, their sum, and the ratio of the KL term to the log-prob term."""

loss_list = []

for step in range(NUM_STEPS):
    samples = batch(BATCH_SIZE)
    loss_list.append(train(samples))
    if step % 250 == 0:
        loss = loss_list[-1][2]
        print(f'Step {step:05d} / {NUM_STEPS:05d}. Loss: {loss:.2f}.')

neg_log_p, kl, loss = zip(*loss_list)
plt.figure(figsize=(6, 2))
plt.plot(loss, linewidth=5, label='sum')
plt.plot(neg_log_p, label='neg log p')
plt.plot(kl, label='kl')
plt.legend()

plt.figure(figsize=(6, 2))
plt.plot(np.array(kl) / np.array(neg_log_p))



samples = batch(10)

mu_z, sigma_z = encoder(samples)
z = encoder.rsample(mu_z, sigma_z)
mu_x, sigma_x = decoder(z)

samples = samples.view(-1, HEIGHT, WIDTH).data.cpu().numpy()
samples = np.concatenate(samples, axis=1)
mu_x = mu_x.view(-1, HEIGHT, WIDTH).data.cpu().numpy()
mu_x = np.concatenate(mu_x, axis=1)
plt.figure(figsize=(15, 5))
plt.set_cmap('gray')
plt.imshow(np.concatenate([samples, mu_x], axis=0))

"""**Modify the following cell to visualize generated images as we traverse $\zb$ space from an image to a 1 to an image of a 7. (You should concatenate the images produced by your interpolated $\zb$'s by concatenating their corresponding images horizontally.)**"""

z_1 = z[-1]  # TODO
z_7 = z[3]  # TODO
mixture_coefs = Variable(torch.arange(0.0, 1.0, 0.05)).view(-1, 1)
z_interp = (1.0 - mixture_coefs) ** .5 * z_1 + mixture_coefs ** .5 * z_7
# TODO
plt.figure(figsize=(15, 5))
plt.set_cmap('gray')
plt.imshow(mu_x)
