# Importing all the libraries needed
import matplotlib.pyplot as plt
import imageio
import glob
import random
import os
import numpy as np
import math
import itertools
import time
import datetime
import cv2
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from discriminator import *
from generator import *
from datasets import *
from utils import *

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 200

"""### Training and evaluating models"""

# parameters
epoch = 0 #  epoch to start training from
n_epoch = 401  #  number of epochs of training
batch_size =10  #  size of the batches
lr = 0.0002 #  adam: learning rate
b1 =0.5  #  adam: decay of first order momentum of gradient
b2 = 0.999  # adam: decay of first order momentum of gradient
decay_epoch = 100  # epoch from which to start lr decay
img_height = 256  # size of image height
img_width = 256  # size of image width
channels = 3  # number of image channels
sample_interval = 500 # interval between sampling of images from generators
checkpoint_interval = -1 # interval between model checkpoints
cuda = True if torch.cuda.is_available() else False # do you have cuda ?

def main():
    # Configure dataloaders
    transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
                   transforms.ToTensor()]  # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

    dataloader = DataLoader(ImageDataset("facades", transforms_=transforms_),
                            batch_size=16, shuffle=True)

    val_dataloader = DataLoader(ImageDataset("facades", transforms_=transforms_, mode='val'),
                                batch_size=8, shuffle=False)

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Visualisation d'image
    #image, mask = next(iter(dataloader))
    #image = reverse_transform(image[0])
    #mask = reverse_transform(mask[0])
    #plot2x2Array(image, mask)
    
    # We take images that have 3 channels (RGB) as input and output an image that also have 3 channels (RGB)
    generator=U_Net(3,3)
    # Check that the architecture is as expected
    generator

    # We have 6 input channels as we concatenate 2 images (with 3 channels each)
    discriminator = PatchGAN(6,1)
    discriminator
    
    
    """Initialize our GAN"""
    # Calculate output of image discriminator (PatchGAN)
    patch = (1, img_height//2**3-2, img_width//2**3-2)

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()
        
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    # ----------
    #  Training
    # ----------

    losses = []
    num_epochs = 401

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    epoch_D = 0
    epoch_G = 0

    # train the network
    discriminator.train()
    generator.train()
    print_every = 400

    for epoch in range(epoch_G, num_epochs):
        for i, batch in enumerate(dataloader):

            # Model inputs
            real_A = Variable(batch[0].type(Tensor))
            real_B = Variable(batch[1].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_B.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_B.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # GAN loss
            fake_A = generator(real_B) # fake image generated based on mask
            pred_fake = discriminator(fake_A, real_B) # Real or fake image-mask pair ? tries to fool Discriminator
            GAN_loss = criterion_GAN(pred_fake, valid) # Loss computation
            # Pixel-wise loss
            pixel_loss = criterion_pixelwise(fake_A, real_A) # Pixel-wise loss

            # Total loss
            loss_G = GAN_loss + lambda_pixel*pixel_loss

            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_A, real_B)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_A.detach(), real_B)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()
            
            # Print some loss stats
            if i % print_every == 0:
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, num_epochs, loss_D.item(), loss_G.item()))
        ## AFTER EACH EPOCH##
        # append discriminator loss and generator loss
        losses.append((loss_D.item(), loss_G.item()))
        if epoch % 100 == 0:
            print('Saving model...')
            save_model(epoch, generator, optimizer_G, loss_G, discriminator, optimizer_D, loss_D)
    return losses
    
if __name__ == '__main__':
    losses = main()
    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.savefig('training.png')