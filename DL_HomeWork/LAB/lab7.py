
import matplotlib.pyplot as plt
import numpy as np
import skimage.data
import scipy.misc
import scipy.signal
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from matplotlib import rcParams

rcParams['axes.grid'] = False

"""## Convolutions with SciPy

Let's start by loading a simple image of coffee using scikit-image, converting it to grayscale, and viewing it.

**You will likely get an error when you run the following line of code. This issue has to do with Google's Colaboratory environment. To fix it, just restart the runtime (`Runtime -> Restart Runtime`) and then run all of the code above again (`Runtime -> Run Before`).**
"""

image = skimage.color.rgb2gray(skimage.data.coffee()).astype(np.float32)

"""**In the following cell, write code to print this image's `dtype`, `shape`, and minimum and maximum values.**"""

# TODO
print(image.dtype)
print(image.shape)
print(image.max())
print(image.min())

"""Let's view the image:"""

plt.imshow(image)
plt.axis('image')
plt.set_cmap('gray')

"""Now let's create a 15 x 15 averaging filter:"""

kernel_shape = [15, 15]
kernel = np.ones(kernel_shape, dtype=np.float32) / np.prod(kernel_shape)

"""**In the following Markdown cell, answer: Why are we dividing by the product of `kernel_shape`'s elements here?**

![alt text](https://)
Maintain the scale of the intensity of pixels in image to prevent the image being "whitening out"
"""

image_smoothed = scipy.signal.convolve2d(image, kernel, mode='same')
kernel.shape

"""**Copy the previous line of code to the cell below and use IPython's `%timeit` magic to see how long this convolution takes.**"""

# %timeit image_smoothed = scipy.signal.convolve2d(image, kernel, mode='same')

"""**In the following Markdown cell, answer: Approximately how many milliseconds does it take for this 2-D convolution to complete?**

140 ms

**In the following Markdown cell, answer: We specified `mode='same'` so that the output image has the same size as the input image. If we instead retained only *valid* outputs – those computed using only values within `image` and `kernel` – what would the shape of the output image be?**

it would be (386, 586)

**In the following Markdown cell, answer: Expanding on the previous question, suppose you convolve an image of shape `[HEIGHT, WIDTH]` with a kernel of smaller shape `[K_HEIGHT, K_WIDTH]`, where `K_HEIGHT` and `K_WIDTH` are odd. Then what is the shape of the output of the convolution if only *valid* outputs are retained?**

[link text](https://)


It would be (HEIGHT - K_HEIGHT + 1, WIDTH - K_WIDTH + 1)

Let's visualize the output of this convolution.
"""

plt.imshow(image_smoothed)
plt.axis('image')

"""**In the following Markdown cell, answer: Why is there an artificial dark border surrounding this output image (which is not present in the original image above)?**

This is the result of using `mode='same'`. Here the original image is effectively padded with 0s so that a 'valid' convolution yields an output that has the same shape as the input image. These 0s are darker than the actual image, so when we include them in our averages, we see this artificial border.

## Convolutions with PyTorch (CPU only)
"""

image_ = Variable(torch.from_numpy(image))
kernel_ = Variable(torch.from_numpy(kernel))

"""**In the following Markdown cell, answer: Look up the documentation for `torch.nn.functional.conv2d`. What shape does it expect for `input`, and what shape does it expect for `weight`? (Note that in our usage, the argument `groups` is 1.)**

input: shape(minibatch x in_channels x H x W)

filter: shape (out_channels x in_channelsgroup x kH x kW)

**In the following cell, write code to reshape `image_` and `kernel_` so that they can be passed to `torch.nn.functional.conv2d`.**
"""

# TODO
# TODO

image_ = image_.view(1, 1, image_.shape[0], image_.shape[1])

print(image_.shape)
kernel_ = kernel_.view(1, 1, kernel_.shape[0], kernel_.shape[1])
print(kernel_.shape)

"""Now let's define appropriate padding (so that our output image again remains the same size at the input image) and use PyTorch's `conv2d` to perform the convolution."""

padding = (kernel_shape[0] // 2, kernel_shape[1] // 2)
image_smoothed_ = F.conv2d(image_, kernel_, padding=padding)

"""**Copy the previous cell's code to the cell below and use IPython's `%timeit` magic to see how long this convolution takes in PyTorch.**"""

# %timeit image_smoothed_ = F.conv2d(image_, kernel_, padding=padding)

"""**In the following Markdown cell, answer: Approximately how many milliseconds does it take for this 2-D convolution to complete?**

98.8

**In the following Markdown cell, answer: How much faster is PyTorch's implementation in comparison to SciPy's? (To answer this, just compute the ratio $T_\text{SciPy}$ / $T_\text{PyTorch}$.)**

TSciPy / TPyTorch = 140ms / 99ms = 1.41

**In the following Markdown cell, answer: Can you guess why PyTorch is faster here? (It's fine if you aren't sure; if so, just leave it blank.)**

Probably because the tensor computation is highly optimized on pytorch.

Again let's visualize the output to make sure it's what we expect.
"""

plt.imshow(image_smoothed_.data.numpy().squeeze())
plt.axis('image')

"""## Convolutions with PyTorch (GPU)

Now let's move on to using CUDA in PyTorch, to leverage GPUs. (If you haven't heard of CUDA, take a quick look at https://en.wikipedia.org/wiki/CUDA.)
"""


"""**If the above `assert` fails, hit `Edit -> Notebook Settings` and make sure GPU acceleration is enabled.**"""

image_ = image_
kernel_ = kernel_

image_smoothed_ = F.conv2d(image_, kernel_, padding=padding)

"""**Copy the above code to the cell below and use IPython's `%timeit` magic to see how long this convolution takes in PyTorch using our GPU.**"""

# %timeit F.conv2d(image_, kernel_, padding=padding)

"""**In the following Markdown cell, answer: Approximately how many milliseconds does it take for this 2-D convolution to complete?**

3 ms

**In the following Markdown cell, answer: How much faster is PyTorch's GPU implementation in comparison to SciPy's CPU implementation? And how much faster is PyTorch's GPU implementation than PyTorch's CPU implementation? (Answer these as done above, as $T_\text{PyTorch GPU}$ / $T_\text{SciPy}$ and $T_\text{PyTorch GPU}$ / $T_\text{PyTorch CPU}$.)**

𝑇PyTorch GPU  / 𝑇SciPy  = 3 / 140 = 0.021

𝑇PyTorch GPU / 𝑇PyTorch CPU = 3 / 99 = 0.030

Now let's go on to convolve an RGB image (height x width x 3) with a kernel that's 15 x 15 x 3.
"""

image = skimage.data.coffee().astype(np.float32)
image /= image.max()
plt.imshow(image)

"""**In the following cell, write code to print this image's `dtype`, `shape`, and minimum and maximum values.**"""

# TODO
print(image.dtype)
print(image.shape)
print(image.max())
print(image.min())

kernel_shape = [15, 15, 3]
kernel = np.ones(kernel_shape, dtype=np.float32) / np.prod(kernel_shape)

image_ = Variable(torch.from_numpy(image) )
kernel_ = Variable(torch.from_numpy(kernel) )

print(image_.shape)
print(kernel_.shape)

"""**In the following cell, write code to permute and reshape axes so that `image_` and `kernel_` have the shapes expected by `torch.nn.functional.conv2d`. (You can use `permute` and `unsqueeze` here.)**"""

# TODO
image_ = image_.unsqueeze(0)
image_ = image_.permute(0, 3, 1, 2)
print(image_.shape)

kernel_ = kernel_.unsqueeze(0)
kernel_ = kernel_.permute(0, 3, 1, 2)
# kernel_ = torch.cat((kernel_,)*3, 0)

print(image_.shape)
print(kernel_.shape)

# input: shape(minibatch x in_channels x H x W)

# filter: shape (out_channels x in_channelsgroup x kH x kW)

"""After the `permute`, we need to make our Variables contiguous. (`permute` changes the order in which we view memory, but avoids rearranging the order explicitly. Thus we need to explicitly reorder the memory so that future manipulations can operate as expected.)"""

image_ = image_.contiguous()
kernel_ = kernel_.contiguous()

"""**In the following cell, write code to print the shape of `image_` and `kernel_`, and confirm they're what you expect.**"""

# TODO
print(image_.shape)
print(kernel_.shape)

output_ = F.conv2d(image_, kernel_, padding=padding)

"""**In the following cell, write code to print the `type` and `shape` of `output_.data`.**"""

# TODO
print(type(output_.data))
print(output_.data.shape)

"""**In the following Markdown cell, answer: Why does the output have 1 output channel instead of 3?**

Each output pixel is the result of an average across *all three channels* of a 15 x 15 spatial window. Here the window moves only over spatial dimensions, not channels.

Finally, let's visualize the result.
"""

plt.imshow(output_.data.cpu().numpy().squeeze())
plt.axis('image')

"""## MNIST Classification with Extremely Simple CNNs"""

import torchvision

from pathlib import Path

HOME = Path.home()
MNIST_PATH = HOME / 'data' / 'mnist'

NUM_CLASSES = 10
CHANNELS = 1
HEIGHT = 28
WIDTH = 28

# We're going to load the official train set and never touch
# the true test set in these experiments, which consists of 10,000
# separate examples. We'll instead split our training set into
# a set for training and a set for validation.
official_mnist_train = torchvision.datasets.MNIST(str(MNIST_PATH), train=True, download=True)
official_train_images = official_mnist_train.train_data.numpy().astype(np.float32)
official_train_labels = official_mnist_train.train_labels.numpy().astype(np.int)

print(official_train_images.shape)
print(official_train_labels.shape)

"""Let's view a few examples:"""

example_images = np.concatenate(official_train_images[:10], axis=1)
example_labels = official_train_labels[:10]
print(example_labels)
plt.imshow(example_images)

"""Here we'll split our training set into 55000 for training and the rest for validation."""

train_images, val_images = np.split(official_train_images, [55000])
train_labels, val_labels = np.split(official_train_labels, [55000])

print(train_images.shape, train_labels.shape)
print(val_images.shape, val_labels.shape)

"""And we'll normalize our data in one of the simplest ways possible: centering and scaling on an image-by-image basis."""


def normalize_stats_image_by_image(images):
    mean = images.mean(axis=(1, 2), keepdims=True)
    stdev = images.std(axis=(1, 2), keepdims=True)
    return (images - mean) / stdev


train_images = normalize_stats_image_by_image(train_images)
val_images = normalize_stats_image_by_image(val_images)

print(train_images[:3].mean(axis=(1, 2)))
print(train_images[:3].std(axis=(1, 2)))
print(val_images[:3].mean(axis=(1, 2)))
print(val_images[:3].std(axis=(1, 2)))

"""As before, we'll define a function to return a batch of examples. However this time we'll assume we have a GPU available."""


def batch(batch_size, training=True):
    """Create a batch of examples.

    This creates a batch of input images and a batch of corresponding
    ground-truth labels. We assume CUDA is available (with a GPU).

    Args:
      batch_size: An integer.
      training: A boolean. If True, grab examples from the training
        set; otherwise, grab them from the validation set.

    Returns:
      A tuple,
      input_batch: A Variable of floats with shape
        [batch_size, 1, height, width]
      label_batch: A Variable of ints with shape
        [batch_size].
    """
    if training:
        random_ind = np.random.choice(train_images.shape[0], size=batch_size, replace=False)
        input_batch = train_images[random_ind]
        label_batch = train_labels[random_ind]
    else:
        input_batch = val_images[:batch_size]
        label_batch = val_labels[:batch_size]

    input_batch = input_batch[:, np.newaxis, :, :]

    volatile = not training
    input_batch = Variable(torch.from_numpy(input_batch) , volatile=volatile)
    label_batch = Variable(torch.from_numpy(label_batch) , volatile=volatile)

    return input_batch, label_batch


"""**Below, you will define a `SimpleCNN` with some significant restrictions on the model class: (1) Input to conv_final needs to be a single pixel (see comments where it is defined). (2) Only Convolutions and ReLUs can be used. In other words, do not use max pooling, do not use dropout, etc. (3) For full credit, achieve better than 2% error.**

**The purpose of this is to (1) gain competency with the basic settings for convolutions and (2) develop a practical sense for how important these basic settings are.**

Hint 1: You can use the `stride` argument in the convolutions.

Hint 2: This can easily be achieved in well under 5000 iterations using the same optimizer settings as below (Adam with a learning rate of 0.001).
"""


class SimpleCNN(torch.nn.Module):
    """A simple convolutional network.

    Map from inputs with shape [batch_size, 1, height, width] to
    outputs with shape [batch_size, 1].
    """

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=7, padding=7 // 2)  # feel free to change these parameters.
        # TODO
        # (You may also need to modify conv_final.)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=7, padding=7 // 2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=7, padding=7 // 2)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=7, padding=7 // 2)
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=7, padding=7 // 2)

        # Here the input to conv_final should be a single pixel, as can be obtained
        # by pooling spatially over all pixels. The goal of conv_final is to map
        # from some number of channels to 10, one for each possible class.

        # Here, in_channel = 128, but feel free to change that. All other parameters for conv_final should remain the same.
        self.conv_final = torch.nn.Conv2d(512, 10, kernel_size=1)
        self.out = torch.nn.Linear(28*28*10, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv_final(x)
        x = x.view(-1, 10*28*28)
        x = self.out(x)

        return F.log_softmax(x, dim=1)


"""And instantiate our model... notice again that we assume CUDA is available, and that moving all parameters to the GPU is as simple as running `model `."""

model = SimpleCNN()
model 

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_step(batch_size=128):
    model.train()

    input_batch, label_batch = batch(batch_size, training=True)
    output_batch = model(input_batch)

    loss = F.cross_entropy(output_batch, label_batch)
    _, pred_batch = torch.max(output_batch, dim=1)
    error_rate = 1.0 - (pred_batch == label_batch).float().mean()

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    return loss.data[0], error_rate.data[0]


def val():
    model.eval()
    input_batch, label_batch = batch(val_images.shape[0], training=False)
    output_batch = model(input_batch)

    loss = F.cross_entropy(output_batch, label_batch)
    _, pred_batch = torch.max(output_batch, dim=1)
    error_rate = 1.0 - (pred_batch == label_batch).float().mean()

    return loss.data[0], error_rate.data[0]


"""Finally, let's train, and also plot loss and error rate as a function of iteration."""

# Let's make sure we always start from scratch (that is,
# without starting from parameters from a previous run).
for module in model.children():
    module.reset_parameters()

info = []
fig, ax = plt.subplots(2, 1, sharex=True)
num_steps = 5000
num_steps_per_val = 50
best_val_err = 1.0
for step in range(num_steps):
    train_loss, train_err = train_step()
    if step % num_steps_per_val == 0:
        val_loss, val_err = val()
        if val_err < best_val_err:
            best_val_err = val_err
            print('Step {:5d}: Obtained a best validation error of {:.3f}.'.format(step, best_val_err))
        info.append([step, train_loss, val_loss, train_err, val_err])
        x, y11, y12, y21, y22 = zip(*info)
        ax[0].plot(x, y11, x, y12)
        ax[0].legend(['Train loss', 'Val loss'])
        ax[1].plot(x, y21, x, y22)
        ax[1].legend(['Train err', 'Val err'])
        ax[1].set_ylim([0.0, 0.25])
