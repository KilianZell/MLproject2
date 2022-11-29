import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

class RoadSegDataset(Dataset):
    """
    Class of the dataset used.
    """
    def __init__(self, image_dir, groundtruth_dir, function=None):
        """
        Initial function that sets the variable of a RoadSegDataset
        Input:
            - self: the RoadSegDataset that is initialized
            - image_dir: path to the folder containing the sat images
            - groundtruth_dir: path to the folder containing the groundtruth images
            - function: transformation functions applied to the images (NONE by default)
        """                 
        self.image_dir = image_dir                                                            #Set sat images directory
        self.groundtruth_dir = groundtruth_dir                                                #Set grountruth directory
        self.function = function                                                              #Set data transformation function (augmentation, normalization, ect..)
        self.images = os.listdir(image_dir)                                                   #List containing the names of the entries in the sat images directory (must be the same names in groundtruth_dir)

    def __len__(self):
        """
        Function that returns the length of the RoadSegDataset
        Imput:
            - self: the RoadSegDataset in question
        Output:
            - the length of the RoadSegDataset
        """
        return len(self.images)                                                              

    def __getitem__(self, index):
        """
        Function that returns a given image and groundtruth pair after binairization 
        of groundtruth and possible transformation (NONE by default).
        Input:
            - self: the RoadSegDataset in question
            - index: the pair of sat image/groundtruth
        Output:
            - image: the sat image in RGB after transformation
            - groundtruth: the groundtruth in black and white after transformation
        """
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