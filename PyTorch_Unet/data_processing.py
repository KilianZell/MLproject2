import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

class RoadSegDataset(Dataset):
    def __init__(self, image_dir, groundtruth_dir, function=None):                            #Set variables
        self.image_dir = image_dir                                                            #Set sat images directory
        self.groundtruth_dir = groundtruth_dir                                                #Set grountruth directory
        self.function = function                                                              #Set data transformation function (augmentation, normalization, ect..)
        self.images = os.listdir(image_dir)                                                   #List containing the names of the entries in the sat images directory (must be the same names in groundtruth_dir)

    def __len__(self):
        return len(self.images)                                                               #Returns number of images in data set

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])                           #Path to sat images
        groundtruth_path = os.path.join(self.groundtruth_dir, self.images[index])             #Path to groundtruth
        image = np.array(Image.open(img_path).convert("RGB"))                                 #Array of sat images (RGB)
        groundtruth = np.array(Image.open(groundtruth_path).convert("L"), dtype=np.float32)   #Array of groudtruth (Gray scale)
        groundtruth[groundtruth > 0.0] = 1.0                                                  #Set mask to binary of groudtruth (white or black, no gray)

        if self.function is not None:
            transformations = self.function(image=image, mask=groundtruth)                     #Transform image and mask according to function (ex: normalization)
            image = transformations["image"]
            groundtruth = transformations["mask"]

        return image, groundtruth