import numpy as np 
from matplotlib import pyplot as plt 
from skimage .transform import AffineTransform, warp 
from skimage import io, img_as_ubyte 
import random 
import os 
from scipy.ndimage import rotate 
import albumentations as A 

images_to_generate=2000 


images_path= "training/images"
masks_path= "training/groundtruth"
images_aug_path = "training/aug_images"
masks_aug_path = "training/aug_groundtruth"
images=[]
masks=[]

for im in os.listdir(images_path):
    images.append(os.path.join(images_path,im))
for msk in os.listdir(masks_path):
    masks.append(os.path.join(masks_path,msk))

aug = A.Compose([
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=1),
    A.Transpose(p=1),
    A.GridDistortion(p=1)
    ]
)


i=1 # variable to iterate tiLL images_to_generate 
while i<=images_to_generate: 
    number = random.randint(0, len(images)-1) #PIck a number to seLect an i 
    image = images[number] 
    mask = masks[number] 
    print(image, mask)
    original_image = io.imread(image) 
    original_mask = io.imread(mask) 
    
    augmented = aug(image=original_image, mask=original_mask) 
    transformed_image = augmented['image'] 
    transformed_mask = augmented['mask'] 
    
    new_image_path= "%s/augmented_image_%s.png" %(images_aug_path,i)
    new_mask_path = "%s/augmented_mask_%s.png" %(masks_aug_path, i)
    io.imsave(new_image_path, transformed_image) 
    io.imsave(new_mask_path, transformed_mask) 
    i = i+1 