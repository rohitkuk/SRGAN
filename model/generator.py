import torch.nn as nn
import torch


nn.PixelShuffle


class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, activation , batch_norm ,*args, **kwargs):
        super(convblock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.activations =  nn.ModuleDict([ 
            ['prelu', nn.PReLU()],
            ['relu', nn.ReLU()],
            ['tanh', nn.Tanh()]
        ])
        self.conv =  nn.Conv2d(self.in_channels, self.out_channels, *args, **kwargs)
        self.batch_norms = nn.ModuleDict([
            ['Yes', nn.BatchNorm2d(self.out_channels)],
            ['No', nn.Identity()]
        ])

        self.l1 = nn.Sequential(
            self.conv,
            self.batch_norms[batch_norm],
            self.activations[activation],
        )

    def forward(self,x):
        x = self.l1(x)
        return x
    


class Generator(nn.Module):
    def __init__(self):
        super(Generator).__init__()



def test():
    pass


if __name__ == '__main__':
    test()