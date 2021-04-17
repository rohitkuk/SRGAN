import torch.nn as nn
import torch


class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, activation , batch_norm ,*args, **kwargs):
        super(convblock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.activations =  nn.ModuleDict([ 
            ["None", nn.Identity()],
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
    

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale, activation = "None" , *args, **kwargs):
        super(UpsampleBLock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = self.in_channels * up_scale ** 2

        self.activations =  nn.ModuleDict([ 
            ["None", nn.Identity()],
            ['prelu', nn.PReLU()],
            ['relu', nn.ReLU()],
            ['tanh', nn.Tanh()]
        ])
        self.conv =  nn.Conv2d(self.in_channels, self.out_channels, *args, **kwargs)

        self.l1 = nn.Sequential(
            self.conv,
            nn.PixelShuffle(up_scale),
            self.activations[activation],
        )

    def forward(self,x):
        x = self.l1(x)
        return x


class Reisdual_block(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(Reisdual_block,self).__init__()

        self.res_block = nn.Sequential(
            
            convblock(in_channels, out_channels, kernel_size = 3, activation="prelu", batch_norm="Yes", 
            stride = 1, padding = 1),

            convblock(in_channels, out_channels, kernel_size = 3, activation="None", batch_norm="Yes",
            stride = 1, padding = 1)
            
            )
        
    def forward(self, x):
        residual = self.res_block(x)
        return x + residual



class Generator(nn.Module):

    def __init__(self, img_channels, features, total_res_blocks= 5):
        super(Generator, self).__init__()
        self.img_channels = img_channels
        self.features = features

        # Iniitial Block IN --> Conv-->PRelU , K9n64s1
        self.initial_block = convblock(
                in_channels = self.img_channels , out_channels = self.features, kernel_size = 9, 
                
                activation="prelu", batch_norm="No", stride = 1, padding = 4
                )

        # 5 Resblocks as per the paper some people use 16
        self.residual_blocks = nn.Sequential(*[
            Reisdual_block(in_channels = self.features, out_channels = self.features) for _ in range(total_res_blocks)
        ])

        # K3n64s1 Conv--> BN
        self.after_res_block  = convblock(
            in_channels=self.features, out_channels=self.features,
            activation = "None", batch_norm="Yes" ,kernel_size = 3, stride  = 1, padding=1)

        # UP Sampling Blocks
        self.up_blocks = nn.ModuleList([
            UpsampleBLock(in_channels=features, up_scale=2,activation='prelu', kernel_size = 3, stride = 1, padding = 1) for _ in range(2)
        ])

        # Final Layer
        self.final_layer = convblock(
            in_channels=self.features, out_channels=self.img_channels, activation="tanh", batch_norm = "No",
            kernel_size = 9, stride = 1, padding = 4                        
                                    )
    def forward(self, x):
        out = self.initial_block(x)
        residual = out
        out = self.residual_blocks(out)
        out = self.after_res_block(out)
        out = self.final_layer(residual+out)
        # return nn.Tanh(out) #Convention to use Tanh activateion to Generator, some people user (Tanh +1)/2 dont know why but worth trying
        return (out +1)/2 #Dont know why we do this
        




def test():
    gen = Generator(img_channels=3, features=64, total_res_blocks=5)
    x = torch.randn((5,3,256,256))
    result = gen(x)
    print(result.shape)
    pass


if __name__ == '__main__':
    test()