
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
    # lum = lum[8:lum.shape[0] - 8, 8:lum.shape[1] - 8]
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

    gt = misc.imread(gt, mode='RGB')
    # gt = cv2.cvtColor(cv2.imread(gt), cv2.COLOR_BGR2RGB)

    for i in range(len(pred)):
    # compare to gt
        # pred_img = cv2.cvtColor(cv2.imread(pred[i]), cv2.COLOR_BGR2RGB)
        pred_img = misc.imread(pred[i], mode='RGB')

        psnr = PSNR(luminance(gt), luminance(pred_img))
        ssim = SSIM(luminance(gt), luminance(pred_img))
        # save results to log_path ex: 'results/experiment1/Set5/baby/1000.png'
        # if save_images:
        #  path = os.path.join(log_path, self.name, self.names[i])
        # gather results
        individual_psnr.append(psnr)
        individual_ssim.append(ssim)
        avg_psnr += psnr
        avg_ssim += ssim


    return individual_psnr, individual_ssim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path Of Directory')
    parser.add_argument('--ref', type=str, help='Path Of Directory')
    parser.add_argument('--name', type=str, help='Result Of Directory')

    args = parser.parse_args()

    file_list =  glob.glob(args.path+'/*.png')
    print(file_list)    

    algm_list = []
    for item in file_list:
        print(item.split('_')[-1].replace('.png',''))
        algm_list.append(item.split('_')[-1].replace('.png',''))


    ref_image = args.ref+'/'+args.name+'.png'

    psnr_li, ssim_li = test_images(ref_image, file_list)

    print(args.path +'/PSNR & SSIM.csv')
    with open(args.path +'/PSNR & SSIM.csv', 'w+') as f:
        for n_name in algm_list:
            f.write("%s, " % n_name)
        f.write("\n")
        for item in psnr_li:
            f.write("%.7f, " % item)
        f.write("\n")
        for item in ssim_li:
            f.write("%.7f, " % item)
    # with open(log_path + '/PSNR.csv', 'a') as f:
    #     f.write(
    #         'iteration, set5_psnr, set5_ssim, set14_psnr, set14_ssim, bsd100_psnr, bsd100_ssim,UCMerced_LandUse_psnr, UCMerced_LandUse_ssim,RSSCN7_psnr, RSSCN7_ssim\n'
    #      )
    #     f.write('%d,%s\n' % (iteration, log_line))