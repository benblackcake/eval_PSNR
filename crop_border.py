import cv2
import numpy as np


if __name__=='__main__':
	img = cv2.imread('baboon_swt-lamb003-alpha5.png')

	h,w,_ = img.shape
	print(h,w)
	crop = img[:,2:w-2,:]

	print(crop.shape)
	cv2.imshow('testing', crop)
	cv2.imwrite('baboon_swt-lamb003-alpha5-out.jpg', crop)
	cv2.waitKey(0)
