import torch
import torchvision
from data_processing import RoadSegDataset
from torch.utils.data import DataLoader
import albumentations as A #data augmentation
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm #progress bar
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_trained_model,
)

DEVICE = "cpu"
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
test_number = 100

if __name__ == '__main__':

    val_transforms = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_trained_model(torch.load("/content/models/my_checkpoint.pth.tar"), model)

    model.eval()

    device="cpu"

    for i in range(test_number):
        test_dir =      "/content/post_processing/data_post/sat_images/sat_test/test_" + str(i+1)
        test_maskdir =  "/content/post_processing/data_post/sat_images/sat_test/test_" + str(i+1)

        print(test_maskdir, end="\r")

        val_ds = RoadSegDataset(
            image_dir=test_dir,
            groundtruth_dir=test_maskdir,
            function=val_transforms,
        )

        test_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        for idx, (x, y) in enumerate(test_loader): 
            x = x.to(device=device)
            with torch.no_grad():
                pred = torch.sigmoid(model(x)) 
                pred = (pred > 0.5).float()
            
            j = i + 1
            torchvision.utils.save_image(
            pred, f'/content/post_processing/data_post/images/satImage_' + '%.3d' % j + '.png')

            