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
REGULARIZATION = 0
DEVICE = torch.device("cpu")
BATCH_SIZE = 5
NUM_EPOCHS = 50 
NUM_WORKERS = 8
PIN_MEMORY = True
LOAD_MODEL = False #False

IMAGE_HEIGHT = 128  # 256 originally, please decomment in train_function if modified
IMAGE_WIDTH = 128   # 256 originally, please decomment in train_function if modified

TRAIN_IMG_DIR =     "data/processing/train/images/"
TRAIN_MASK_DIR =    "data/processing/train/masks/"
VAL_IMG_DIR =       "data/processing/val/images/"
VAL_MASK_DIR =      "data/processing/val/masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    Function that trains one epoch.
    Input:
        - loader: the Dataloader that will be trained
        - model: the UNET used
        - optimizer: the optimizer used to train the model
        - loss_fn: the loss function used to evaluate the model
        - scaler: will help perform the steps of gradient scaling conveniently
    """    
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
            A.Rotate(limit=90, p=0.2),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
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
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)          #Declare model from class UNET
    
    loss_fn = nn.BCEWithLogitsLoss()                                #Calls binary crossentropy from torch library, This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, 
                            weight_decay=REGULARIZATION)            #Calls adam optimizer from torch library (slowly decreses the learning rate)

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

    if LOAD_MODEL:
        load_trained_model(torch.load("my_checkpoint.pth.tar"), model)

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