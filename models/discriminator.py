import torch.nn as nn
import torch


class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm ,*args, **kwargs):
        super(convblock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv =  nn.Conv2d(self.in_channels, self.out_channels, *args, **kwargs)
        self.batch_norms = nn.ModuleDict([
            ['Yes', nn.BatchNorm2d(self.out_channels)],
            ['No', nn.Identity()]
        ])

        self.l1 = nn.Sequential(
            self.conv,
            self.batch_norms[batch_norm],
            nn.LeakyReLU(0.2),
        )

    def forward(self,x):
        x = self.l1(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_channels, features):
        super(Discriminator, self).__init__()
        self.img_channels = img_channels
        self.features = features
        
        self.l1 = convblock(in_channels=self.img_channels, out_channels=self.features, batch_norm="No", kernel_size = 3, stride = 1, padding =1)
        self.l2 = convblock(in_channels=self.features, out_channels=self.features, batch_norm="Yes", kernel_size = 3, stride = 2, padding =1)

        self.l3 = convblock(in_channels=self.features, out_channels=self.features*2, batch_norm="Yes", kernel_size = 3, stride = 1, padding =1)
        self.l4 = convblock(in_channels=self.features*2, out_channels=self.features*2, batch_norm="Yes", kernel_size = 3, stride = 2, padding =1)

        self.l5 = convblock(in_channels=self.features*2, out_channels=self.features*4, batch_norm="Yes", kernel_size = 3, stride = 1, padding =1)
        self.l6 = convblock(in_channels=self.features*4, out_channels=self.features*4, batch_norm="Yes", kernel_size = 3, stride = 2, padding =1)

        self.l7 = convblock(in_channels=self.features*4, out_channels=self.features*8, batch_norm="Yes", kernel_size = 3, stride = 1, padding =1)
        self.l8 = convblock(in_channels=self.features*8, out_channels=self.features*8, batch_norm="Yes", kernel_size = 3, stride = 2, padding =1)


        self.l9  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1, kernel_size=1),
            # nn.Sigmoid() not needed as BCELogits does simoid
        )   
    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)
        out = self.l8(out)
        out = self.l9(out)
        return out



def test():
    disc = Discriminator(img_channels=3, features=64)
    x = torch.randn((5,3,256,256))
    result = disc(x)
    print(result.shape)
    pass


if __name__ == '__main__':
    test()