"""
Colorization framed as a regression problem.
"""

from __future__ import print_function
import argparse
import os
import numpy as np
import numpy.random as npr
import scipy.misc
import torch
import torch.nn as nn
from load_data import load_cifar10

from colorization import process, get_batch, MyConv2d


class RegressionCNN(nn.Module):
    def __init__(self, kernel, num_filters):
        # first call parent's initialization function
        super(RegressionCNN, self).__init__()
        padding = kernel // 2

        self.downconv1 = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        # num_filters = 32
        # downconv1[0] weight.size [32,1,3,3]. bias.size [32]
        # downconv1[2] weight.size[32]. bias.size[32]
        self.downconv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())
        # downconv2[0] weight.size [64,32,3,3]. bias.size [64]
        # downconv1[2] weight.size[64]. bias.size[64]
        self.rfconv = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())
        # rfconv[0] weight.size[64,64,3,3]  bias.size [64]
        # rfconv[1] weight.size[64]  bias.size [64]
        self.upconv1 = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU())
        # upconv1[0] weight.size[32,64,3,3] bias.size[32]
        # upconv1[2] weight.size[32] bias.size[32]
        self.upconv2 = nn.Sequential(
            nn.Conv2d(num_filters, 3, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(3),
            nn.ReLU())
        # upconv2[0] weight.szie[3,32,3,3] bias.size[3]
        self.finalconv = MyConv2d(3, 3, kernel_size=kernel)
        # finalconv weight[3,3,3,3] bias[3]
    def forward(self, x):
        out = self.downconv1(x)
        out = self.downconv2(out)
        out = self.rfconv(out)
        out = self.upconv1(out)
        out = self.upconv2(out)
        out = self.finalconv(out)
        return out
# out.size[15,3,32,32]

######################################################################
# Training
######################################################################


def get_torch_vars(xs, ys, gpu=False):
    """
    Helper function to convert numpy arrays to pytorch tensors.
    If GPU is used, move the tensors to GPU.

    Args:
      xs (float numpy tensor): gray scale input
      ys (float numpy tensor): color output
      gpu (bool): whether to move pytorch tensor to GPU
    Returns:
      xs, ys
    """
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).float()
    if gpu:
        xs = xs.cuda()
        ys = ys.cuda()
    return xs, ys


def train(cnn, epochs=80, learn_rate=0.001, batch_size=100, gpu=True):
    """
    Train a regression CNN. Note that you do not need this function.
    Included for reference.
    """
    if gpu:
        cnn.cuda()

    # Set up L2 loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learn_rate)

    # Loading & transforming data
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    train_rgb, train_grey = process(x_train, y_train)
    test_rgb, test_grey = process(x_test, y_test)

    print("Beginning training ...")

    for epoch in range(epochs):
        # Train the Model
        cnn.train()  # Change model to 'train' mode
        for i, (xs, ys) in enumerate(get_batch(train_grey,
                                               train_rgb,
                                               batch_size)):
            images, labels = get_torch_vars(xs, ys, gpu)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, epochs, loss.data[0]))

        # Evaluate the model
        cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        losses = []
        for i, (xs, ys) in enumerate(get_batch(test_grey,
                                               test_rgb,
                                               batch_size)):
            images, labels = get_torch_vars(xs, ys, gpu)
            outputs = cnn(images)

            val_loss = criterion(outputs, labels)
            losses.append(val_loss.data[0])

        val_loss = np.mean(losses)
        print('Epoch [%d/%d], Val Loss: %.4f' % (epoch + 1, epochs, val_loss))

    # Save the Trained Model
    torch.save(cnn.state_dict(), 'regression_cnn_k%d_f%d.pkl' % (
        args.kernel, args.num_filters))


def plot(gray, gtcolor, predcolor, path):
    """
    Plot input, gt output and predicted output as an image.

    Args:
      gray: numpy tensor of shape Nx1xHxW
      gtcolor: numpy tensor of shape Nx3xHxW
      predcolor: numpy tensor of shape Nx3xHxW
      path: path to save the image
    """
    gray = np.transpose(gray, [0, 2, 3, 1])
    gtcolor = np.transpose(gtcolor, [0, 2, 3, 1])
    predcolor = np.transpose(predcolor, [0, 2, 3, 1])

    img = np.vstack([
        np.hstack(np.tile(gray, [1, 1, 1, 3])),
        np.hstack(gtcolor),
        np.hstack(predcolor)])
    scipy.misc.toimage(img, cmin=0, cmax=1).save(path)


######################################################################
# MAIN
######################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train colorization")
    parser.add_argument('--gpu', action='store_true', default=False,
                        help="Use GPU for training")
    parser.add_argument('-k', '--kernel', default=3,
                        help="Convolution kernel size")
    parser.add_argument('-f', '--num_filters', default=32,
                        help="Base number of convolution filters")
    args = parser.parse_args()

    npr.seed(0)

    cnn = RegressionCNN(args.kernel, args.num_filters)

    # Uncomment to train. You do not need this for the assignment.
    # Included for completeness
    # train(cnn); exit(0)

    print("Loading weights...")
    checkpoint = torch.load('weights/regression_cnn_k%d_f%d.pkl' % (args.kernel, args.num_filters),
                            map_location=lambda storage, loc: storage)
    cnn.load_state_dict(checkpoint)

    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    test_rgb, test_grey = process(x_test, y_test)

    # Create output folder if not created
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    print("Generating predictions...")
    grey = test_grey[:15]
    gtrgb = test_rgb[:15]
    predrgb = cnn(torch.from_numpy(grey).float())
    predrgb = predrgb.data.numpy()
    plot(grey, gtrgb, predrgb, "outputs/regression_output.png")
