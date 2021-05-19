# Importing Modules
import numpy as np
from math import log10, sqrt
import cv2
from skimage.metrics import structural_similarity

# PSNR Function
def PSNR(original, compressed):

    # could also use scikit-image 
    # scikit-image was giving a different answer because of dtype and normalization
    # solved the above by implementing dtype float here 

    original = np.asarray(original, dtype=np.float32)
    compressed = np.asarray(compressed, dtype=np.float32)
    mse = np.mean((original-compressed)**2)
    # MSE 0 means no noise
    if mse == 0:
        return 100
    max_pixel = 255
    psnr = 20 * log10(max_pixel/sqrt(mse))
    return psnr


def SSIM(original, compressed):
    # use this only as of now will implement in future
    """
    SSIM is structural similartiy Index metric to calculate the similartiy between two images.
    
    """
    return structural_similarity(original, compressed, multichannel=True)


def psnr_test(original_image, compressed_image):
    print("Read Done")
    print(PSNR(original_image, compressed_image))


def ssim_test(original_image, compressed_image):
    print(SSIM(original_image, compressed_image))

if __name__ == '__main__':
    original_image = cv2.imread('../testing/original.jpeg')
    compressed_image = cv2.imread('../testing/compressed.jpeg')
    psnr_test(original_image, compressed_image)
    ssim_test(original_image, compressed_image)


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += self.val * n 
        self.count += n 
        self.avg = self.sum/self.count