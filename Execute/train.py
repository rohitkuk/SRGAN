# Imports
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.nn as nn
import torch
from torch import optim as optim 
from torch.utils.data import DataLoader
from models.discriminator import Discriminator
from models.generator import Generator
from models.VGG_Trucated import TruncatedVGG19
from Data.Prepare import SrGanDataset
from tqdm import tqdm 
import os 


# HyperParameters
CHECKPOINT = None
IMG_CHANNELS = 3
GEN_FEATURE = 64
DISC_FEATURE = 64
LEARNING_RATE = 10e-4
BETA1 = 0.5
BETA2 = 0.99
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BETA = 90
# will Verify the Mean and STD once in place

NORM_MEAN = [0.4262, 0.4133, 0.3996]
NORM_STD = [0.2174, 0.2022, 0.2092]

IMAGE_SHAPE = (1024, 1024)


# Load the data

train_dataset = SrGanDataset(dir_ = "Dataset/train", mean = NORM_MEAN, std = NORM_STD, hr_shape=IMAGE_SHAPE)
train_loader = DataLoader(train_dataset, shuffle = True, batch_size = 1)


# Load the Models, Optimizers
if CHECKPOINT is None:
    generator_ = Generator(IMG_CHANNELS, GEN_FEATURE).to(DEVICE)
    discriminator = Discriminator(IMG_CHANNELS, DISC_FEATURE).to(DEVICE)
    gen_optimizer = torch.optim.Adam(generator_.parameters(), lr = LEARNING_RATE, betas = (BETA1, BETA2))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr = LEARNING_RATE, betas = (BETA1, BETA2))



# print(discriminator)

# Losses
truncated_vgg = TruncatedVGG19().to(DEVICE)
content_loss_criterion = nn.MSELoss().to(DEVICE)
adverserial_loss_criterion = nn.BCEWithLogitsLoss().to(DEVICE)

truncated_vgg.eval()
generator_.train()
discriminator.train()

# train Function

print("Till the train Function")
def main(epoch):
    loop = tqdm(enumerate(train_loader), leave = False,desc="{}/{}".format(epoch,EPOCHS))
    for batch_idx , (img_lr, img_hr) in loop:
        img_lr, img_hr = img_lr.to(DEVICE), img_hr.to(DEVICE)        
        # train discrimenator 
        img_gen = generator_(img_lr)

        disc_real = discriminator(img_hr)
        disc_gen = discriminator(img_gen)

        disc_loss_real =  adverserial_loss_criterion(disc_real, torch.ones_like(disc_real))
        disc_loss_gen  = adverserial_loss_criterion(disc_gen, torch.zeros_like(disc_gen))
        disc_loss      = (disc_loss_real + disc_loss_gen)

        discriminator.zero_grad()
        disc_loss.backward()
        disc_optimizer.step

        # train generator
        img_gen = generator_(img_lr)
        disc_gen = discriminator(img_gen)

        # VGG Feature Maps

        img_gen_vgg19 = truncated_vgg(img_gen)
        img_hr_vgg19 = truncated_vgg(img_hr)
        content_loss = content_loss_criterion(img_gen_vgg19, img_hr_vgg19)
        adverserial_loss = adverserial_loss_criterion(disc_gen, torch.ones_like(disc_gen))

        perceptual_loss = content_loss + adverserial_loss * BETA
        
        generator_.zero_grad()
        perceptual_loss.backward()
        gen_optimizer.step

        loop.set_postfix(
        disc_loss = disc_loss.item(),
        gen_loss  = perceptual_loss.item()
            )


if __name__ == "__main__":
    for epoch in tqdm(EPOCHS):
        main(epoch)
