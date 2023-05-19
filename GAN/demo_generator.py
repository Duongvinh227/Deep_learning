import argparse
import os

import cv2
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from gan import opt, Generator

os.makedirs("images_demo", exist_ok=True)


img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
generator.load_state_dict(torch.load("generator_new.pth"))

if cuda:
    generator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

image = cv2.imread("input.JPG")
# Resize ảnh thành kích thước (28, 28)
resized_image = cv2.resize(image, (28, 28))
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Chuyển đổi ảnh thành tensor
transform = transforms.Compose([
    transforms.ToTensor()
])
tensor = transform(gray_image)
# Nhân bản tensor nhiều lần để tạo tensor batch
batch = opt.n_epochs
batch_tensor = tensor.repeat(batch, 1, 1, 1)

for epoch in range(opt.n_epochs):
    # Adversarial ground truths
    val_custom = Variable(Tensor(batch_tensor.size(0), 1).fill_(1.0), requires_grad=False)
    #   Generator

    optimizer_G.zero_grad()

    # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (batch_tensor.shape[0], opt.latent_dim))))

    # Generate a batch of images
    gen_imgs = generator(z)

    optimizer_G.step()

    batches_done = epoch
    save_image(gen_imgs.data[:25], f"images/{batches_done}.png", normalize=True)
    print('save')

