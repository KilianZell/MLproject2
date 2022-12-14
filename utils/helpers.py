import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import transforms

import matplotlib.image as mpimg

import numpy as np

import time
import os,sys
import random

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

# Function that implements random cropping

def uniform(a,b):
    return(a+(b-a)*random.random())

def img_rnd_crop(im, w, h, i = -1, j = -1):
    is_2d = len(im.shape) < 3
    imgwidth = im.shape[len(im.shape)-2]
    imgheight = im.shape[len(im.shape)-1]
    if (i == -1 and j == -1): # random center for the crop
        i = int(uniform(0, imgwidth-w-1))
        j = int(uniform(0, imgheight-h-1))
    if is_2d:
        im_patch = im[i:i+w, j:j+h]
    else:
        im_patch = im[:, i:i+w, j:j+h]
    return im_patch, i, j


def rotated_expansion(imgs):
    shape = [imgs.shape[i] for i in range(len(imgs.shape))]
    shape[0] = shape[0]*4 # there will be 4 times as many images after we rotate in each direction
    shape = tuple(shape)
    rotated_imgs = np.empty(shape)
    
    for index in range(int(shape[0]/4)):
        img = imgs[index]
        if(len(np.shape(img))>2):
            img90 = np.rot90(img.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
            img180 = np.rot90(img90.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
            img270 = np.rot90(img180.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
        else:
            img90 = np.rot90(img)
            img180 = np.rot90(img90)
            img270 = np.rot90(img180)
        
        rotated_imgs[index*4] = img
        rotated_imgs[index*4+1] = img90
        rotated_imgs[index*4+2] = img180
        rotated_imgs[index*4+3] = img270
    
    return rotated_imgs

def flipped_expansion(imgs):
    shape = [imgs.shape[i] for i in range(len(imgs.shape))]
    shape[0] = shape[0]*3 # there will be 4 times as many images after we rotate in each direction
    shape = tuple(shape)
    flipped_imgs = np.empty(shape)
    
    for index in range(int(shape[0]/3)):
        img = imgs[index]
        if(len(np.shape(img))>2):
            imgup = np.flipud(img.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
            imglr = np.fliplr(img.swapaxes(1,2).swapaxes(0,2)).swapaxes(0,2).swapaxes(1,2)
        else:
            imgup = np.flipud(img)
            imglr = np.fliplr(img)
        
        flipped_imgs[index*3] = img
        flipped_imgs[index*3+1] = imgup
        flipped_imgs[index*3+2] = imglr
    
    return flipped_imgs

def load_images_and_grounds(imgs_path,gt_path,nb_imgs):

    files = os.listdir(imgs_path)
    n = min(nb_imgs, len(files)) # Load maximum nb_images
    imgs = np.array([load_image(imgs_path + files[i]) for i in range(n)]).swapaxes(1,3).swapaxes(2,3)

    files = os.listdir(gt_path)
    n = min(nb_imgs, len(files))
    gt_imgs = [load_image(gt_path + files[i]) for i in range(n)]

    imgs = np.array(imgs)
    gt_imgs = np.array(gt_imgs)

    return imgs, gt_imgs


def crop_images_train(ratio_train_val, imgs, grounds,w,h,nb_train):

    # crop images to their w*h counterparts
    cropped_imgs = []
    cropped_targets = []
    end_train = int(ratio_train_val*nb_train)
    
    for q in range(nb_train // len(imgs)):
        # on crop plusieurs fois pour avoir le bon nombre d'??chantillons
        for i in range(len(imgs)): 
            cropped_img, k, l = img_rnd_crop(imgs[i], w, h)
            cropped_target, _, _ = img_rnd_crop(grounds[i], w, h, k, l)
            cropped_imgs.append(cropped_img)
            cropped_targets.append(cropped_target)
    # puis on compl??te
    for i in range(nb_train % len(imgs)):
        cropped_img, k, l = img_rnd_crop(imgs[i], w, h)
        cropped_target, _, _ = img_rnd_crop(grounds[i], w, h, k, l)
        cropped_imgs.append(cropped_img)
        cropped_targets.append(cropped_target)
        

    x = list(range(nb_train))
    random.shuffle(x)

    train_input = [cropped_imgs[i] for i in x[:end_train]]
    validation_input = [cropped_imgs[i] for i in x[end_train:]]

    train_target = [cropped_targets[i] for i in x[:end_train]]
    validation_target = [cropped_targets[i] for i in x[end_train:]]

    return train_input, validation_input, train_target, validation_target


def load_test():
    if os.path.exists("ResNet"):
        root_dir = "ResNet/data/test_set_images/"
    else:
        root_dir = "data/test_set_images/"
    test_images=[]
    for i in range(1, 51):
        image_filename = root_dir + "test_" + str(i) + "/test_" + str(i) + '.png'
        test_images.append(np.array(load_image(image_filename)).swapaxes(0,2).swapaxes(1,2))

    return test_images


def uncrop_256_to_608(imgs):
    output_shape = (3,608,608)
    
    interval_tl = slice(0,256) #interval corresponding to left of x axis and top of y axis
    interval_c = slice(176,432) #interval corresponding to center of both axis
    interval_br = slice(352,608) #interval corresponding to right of x axis and bottom of y axis
    
    top_left = np.zeros(output_shape)
    top_center = np.zeros(output_shape)
    top_right = np.zeros(output_shape)
    center_left = np.zeros(output_shape)
    true_center = np.zeros(output_shape)
    center_right = np.zeros(output_shape)
    bottom_left = np.zeros(output_shape)
    bottom_center = np.zeros(output_shape)
    bottom_right = np.zeros(output_shape)
    
    top_left[:,interval_tl, interval_tl] = imgs[0]
    top_center[:,interval_tl, interval_c] = imgs[1]
    top_right[:,interval_tl, interval_br] = imgs[2]
    center_left[:,interval_c, interval_tl] = imgs[3]
    true_center[:,interval_c, interval_c] = imgs[4]
    center_right[:,interval_c, interval_br] = imgs[5]
    bottom_left[:,interval_br, interval_tl] = imgs[6]
    bottom_center[:,interval_br, interval_c] = imgs[7]
    bottom_right[:,interval_br, interval_br] = imgs[8]
    
    output = np.max([top_left, top_center, top_right, 
                     center_left, true_center, center_right,
                     bottom_left, bottom_center, bottom_right], axis=0).astype(np.uint8)
    return output