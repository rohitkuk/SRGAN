# Imports
import torch.nn as nn
import torch
from models.discriminator import Discriminator
from models.generator import Generator
from models.VGG_Trucated import TruncatedVGG19

# Load the data

train_loader = Dataset_Loader()


# HyperParameters
CHECKPOINT = None
IMG_CHANNELS = 3
GEN_FEATURE = 64
DISC_FEATURE = 64
LEARNING_RATE = 10e-4

# Load the Models, Optimizers

if CHECKPOINT is None:
    generator = Generator(IMG_CHANNELS, GEN_FEATURE)
    discriminator = Discriminator(IMG_CHANNELS, DISC_FEATURE)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr = LEARNING_RATE)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr = LEARNING_RATE)







# Losses





# train Function


# save the check_points in between the function 



