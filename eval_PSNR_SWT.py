
import argparse
import os 
import cv2
import numpy as np
import glob
from scipy import misc
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from skimage.color import rgb2ycbcr, rgb2yuv


def luminance(image):
    # Get luminance
    lum = rgb2ycbcr(image)[:, :, 0]
    # Crop off 4 border pixels
    lum = lum[8:lum.shape[0] - 8, 8:lum.shape[1] - 8]
    # lum = lum.astype(np.float64)
    return lum

def PSNR(gt, pred):
    return compare_psnr(gt, pred, data_range=255)

def SSIM(gt, pred):
    ssim = compare_ssim(gt, pred, data_range=255, gaussian_weights=True)
    return ssim

def test_images(gt, pred):
    avg_psnr = 0
    avg_ssim = 0
    individual_psnr = []
    individual_ssim = []

    # gt = cv2.cvtColor(cv2.imread(gt), cv2.COLOR_BGR2RGB)
    gt = misc.imread(gt, mode='RGB')
    # for i in range(len(pred)):
    # compare to gt
    # pred_img = cv2.cvtColor(cv2.imread(pred[0]), cv2.COLOR_BGR2RGB)
    pred_img = misc.imread(pred[0], mode='RGB')
    psnr = PSNR(luminance(gt), luminance(pred_img))
    ssim = SSIM(luminance(gt), luminance(pred_img))

    return psnr, ssim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path Of Directory')
    parser.add_argument('--ref', type=str, help='Path Of Directory')
    parser.add_argument('--name', type=str, help='Result Of Directory')

    args = parser.parse_args()

    file_list =  glob.glob(args.path+'/*.png')
    print(file_list)    

    ref_image = args.ref+'/'+args.name+'.png'

    psnr, ssim = test_images(ref_image, file_list)

    print(psnr, ssim)