from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import cv2
import argparse
import os 
import datetime
import sys
import glob

def build_log_dir(name):
    """Set up a timestamped directory for results and logs for this training session"""
    if name:
        log_path = name  # (name + '_') if name else ''
    else:
        log_path = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join('results', log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(log_path+'_crop')
    print('Logging results for this session in folder "%s".' % log_path)

    with open(log_path + '/args.txt', 'w+') as f:
        f.write(' '.join(sys.argv))
    return log_path, log_path+'_crop'

def process_crop(path, file_list, log_path, log_path_crop, posi_list):
    for img_name in file_list:

        print(log_path+'/'+img_name.split('_')[-1])

        img = cv2.cvtColor(cv2.imread(img_name),cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = posi_list # butterfly & bird
        pure = img.copy()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        crop = pure[y1:y1+(y2-y1),x1:x1+(x2-x1),:]
        crop= cv2.resize(crop,None,fx=2.0,fy=2.0,interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(log_path+'/'+img_name.split('_')[-1], cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(log_path_crop+'/'+img_name.split('_')[-1], cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        # with open(log_path + '/position.txt', 'w') as f:
        #     f.write(posi_list)
        with open(log_path + '/position.txt', 'w+') as f:
            for item in posi_list:
                f.write("%s " % item)
        # return log_path, log_path+'_crop'
def mouse_select(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
 
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
 
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
 
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path Of Directory')
    parser.add_argument('--ref', type=str, help='Path Of Directory')
    parser.add_argument('--name', type=str, help='Result Of Directory')
    parser.add_argument('--mouse', action='store_true', help='')
    parser.add_argument('--x', type=int, help='Crop x point')
    parser.add_argument('--y', type=int, help='Crop y point')
    parser.add_argument('--h', type=int, help='Crop Hight')
    parser.add_argument('--w', type=int, help='Crop Width')

    args = parser.parse_args()

    cropping = False
    end_crop = False 
    x_start, y_start, x_end, y_end = 0, 0, 0, 0

    if args.mouse:
        ref_path = args.ref+'/'+args.name+'.png'
        print(ref_path)
        image = cv2.imread(ref_path)
        oriImage = image.copy()

        cv2.namedWindow("Crop Image")
        cv2.setMouseCallback("Crop Image", mouse_select)

        while True:
         
            i = image.copy()

            if not cropping:
                cv2.imshow("Crop Image", image)
                if end_crop:
                    break
            elif cropping:
                cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                cv2.namedWindow("Crop Image")

                cv2.imshow("Crop Image", i)

                end_crop = True

            cv2.waitKey(1)


        print(x_start, y_start, x_end, y_end )
        posi_list = [x_start, y_start, x_end, y_end]

    else:
        posi_list = [args.x, args.y,args.x+args.w, args.y+args.h]

    file_list = glob.glob(args.path+'/*.png')
    print(file_list)

    log_path, log_path_crop = build_log_dir(args.name)
    # img = cv2.cvtColor(cv2.imread('build_BICUBIC.png'),cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img,None,fx=4.0,fy=4.0,interpolation=cv2.INTER_CUBIC)
    # posi_list = [x_start, y_start, x_end, y_end]

    process_crop(args.path, file_list, log_path, log_path_crop, posi_list)

