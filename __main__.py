


"""
ToDos:
- Generator :
    -- Read about PReLU (Parameteric ReLU)
- Disriminator:

- Reading about below:
    -- Pixel shuffle ? - Done 
            Basically it rearranges channels to dimensions
            (B, C, H, W ) --> C = C x r x --> = (B, C, H x r, W x R)
            https://github.com/pytorch/pytorch/pull/6340/files
            https://github.com/microsoft/CNTK/issues/2501
            https://github.com/pytorch/pytorch/issues/1684

    -- math.log and math.log2 #loge euler natural growth -
        math.log takes 2 arguments (x, base) default is e 
        math.log2 is log base 2 

    -- SSIM Structuram Similarity Index - Done
    -- PSNR Peak signal noise ratio- Done 
    -- AvgMeter
    -- Grad Clipping - Not cleared really but will get back to it again.
    
"""

from Execute import train



def main():
    for epoch in range(10):
        train.main(epoch)



if __name__ == "__main__":
    main()

