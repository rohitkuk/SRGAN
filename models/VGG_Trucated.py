import torch.nn as nn
import torchvision

class TruncatedVGG19(nn.Module):
    """
    A truncated VGG19 network, such that its output is the 'feature map obtained by the j-th convolution (after activation)
    before the i-th maxpooling layer within the VGG19 network', as defined in the paper.
    Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss.
    """
    def __init__(self):

        super(TruncatedVGG19, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)
        
        # As pe rthe paper take forst 36 layers of vgg truncated.
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:35 + 1])
    
    def forward(self, input):
        """
        Forward propagation
        :param input: high-resolution or super-resolution images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: the specified VGG19 feature map, a tensor of size (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)
        return output