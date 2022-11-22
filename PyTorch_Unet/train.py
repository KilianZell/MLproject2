import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A #data augmentation
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm #progress bar library
from model import UNET
from utils import (
    load_trained_model,
    save_trained_model,
    get_loaders,
    check_accuracy,
    save_predictions_as_masks,
)

#------Hyperparameters--------
LEARNING_RATE = 1e-4
DEVICE = "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT = 400  # 400 originally
IMAGE_WIDTH = 400  # 400 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR =     "data/aug_data/train/images/"
TRAIN_MASK_DIR =    "data/aug_data/train/masks/"
VAL_IMG_DIR =       "data/aug_data/val/images/"
VAL_MASK_DIR =      "data/aug_data/val/masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):    
    #Training of 1 epoch
    loop = tqdm(loader)                                     #Initiate the progress bar

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_functions = A.Compose(   
    #Set the transformations for train dataset       
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_functions = A.Compose(
    #Set the transformations for validation dataset 
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

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)          #Declare model from class UNET
    loss_fn = nn.BCEWithLogitsLoss()                                #Calls binary crossentropy from torch library
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)    #Calls adam optimizer from torch library

    train_loader, val_loader = get_loaders(                         #Prepare data to load in the model
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_functions,
        val_functions,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    #if LOAD_MODEL:
        #load_trained_model(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)                #Check accuracy at every step
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
    #Train model for n epochs
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        #Save model at every epoch
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_trained_model(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)            #Check accuracy

                                                                    
        save_predictions_as_masks(                                  #Get masks images from our predictions
            val_loader, model, folder="saved_predictions/", 
            device=DEVICE
        )


if __name__ == "__main__":                                          #Needed for NUM_WORKERS
    main()