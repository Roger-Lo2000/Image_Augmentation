import sys
import os
import cv2
import numpy as np
from numpy.lib.utils import deprecate
from tqdm import tqdm, trange
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-k","--kernel_size",type=int,default = 9)
parser.add_argument("-a","--alpha",type=float,default = 0.6)
parser.add_argument("-m","--mean",type=float,default = 0)
parser.add_argument("-v","--variance",type=float,default = 0.1)

args = parser.parse_args()

image_list = []
path = "./image"
os.makedirs("augmentation")

def readfile(path):
    for pic in os.listdir(path):
        image_list.append(pic) 

def vertical_write(kernel_size):
    os.makedirs("./augmentation/vertical_motion_aug")
    print("Vertical motion proccessing:")
    f_v = np.zeros((kernel_size,kernel_size))
    f_v[:,int((kernel_size - 1)/2)] = 1
    f_v /= kernel_size

    with tqdm(total=len(image_list)) as pbar:
        for image in image_list:
            pic = cv2.imread("./image/"+image)
            image_v = cv2.filter2D(pic,-1,f_v)
            cv2.imwrite("./augmentation/vertical_motion_aug/"+image,image_v)
            pbar.update(1)

def horizontal_write(kernel_size):
    os.makedirs("./augmentation/horizontal_motion_aug")
    print("Horizontal motion proccessing:") 
    f_h = np.zeros((kernel_size,kernel_size))
    f_h[int((kernel_size - 1)/2),:] = 1
    f_h /= kernel_size
    with tqdm(total=len(image_list)) as pbar:
        for image in image_list:
            pic = cv2.imread("./image/"+image)
            image_v = cv2.filter2D(pic,-1,f_h)
            cv2.imwrite("./augmentation/horizontal_motion_aug/"+image,image_v)
            pbar.update(1)

def darker_write(alpha):
    os.makedirs("./augmentation/darker_aug")
    print("Image darker proccessing:")
    with tqdm(total=len(image_list)) as pbar:
        for image in image_list:
            pic = cv2.imread("./image/"+image)
            image_d = pic * alpha
            image_d.astype('float')/255.0
            cv2.imwrite("./augmentation/darker_aug/"+image,image_d)
            pbar.update(1)     

def gaussian_noise_write(mean,variance):
    os.makedirs("./augmentation/gaussian_noise_aug")
    print("Gaussian noise proccessing:")
    with tqdm(total=len(image_list)) as pbar:
        for image in image_list:
            pic = cv2.imread("./image/"+image)
            noise = np.random.normal(mean, variance, pic.shape)
            image_g = pic + noise * 255
            cv2.imwrite("./augmentation/gaussian_noise_aug/"+image,image_g)
            pbar.update(1)   
            
readfile(path)
horizontal_write(args.kernel_size)
vertical_write(args.kernel_size)
darker_write(args.alpha)
gaussian_noise_write(args.mean,args.variance)


