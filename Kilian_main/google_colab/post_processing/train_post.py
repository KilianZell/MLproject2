import torch
import torch.nn as nn
import torch.optim as optim
from plot import plot_loss
from plot import plot_accuracy
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
LEARNING_RATE = 1e-7
REGULARIZATION = 0
DEVICE = "cuda"
BATCH_SIZE = 10
NUM_EPOCHS = 50
NUM_WORKERS = 1
PIN_MEMORY = False
LOAD_MODEL = False #False

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

TRAIN_IMG_DIR =     "/content/post_processing/data_post/post_processing/train/images/"
TRAIN_MASK_DIR =    "/content/post_processing/data_post/post_processing/train/masks/"
VAL_IMG_DIR =       "/content/post_processing/data_post/post_processing/val/images/"
VAL_MASK_DIR =      "/content/post_processing/data_post/post_processing/val/masks/"

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

    losses = []
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    return sum(losses)/len(losses) 


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
        load_trained_model(torch.load("models/my_checkpoint_post.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)                #Check accuracy at every step
    scaler = torch.cuda.amp.GradScaler()


    losses = []
    accuracies_val = []
    accuracies_train = []
    for epoch in range(NUM_EPOCHS):
    #Train model for n epochs
        print("Epoch: ", epoch+1)
        losses.append(train_fn(train_loader, model, optimizer, loss_fn, scaler))

        #Save model at every epoch
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_trained_model(checkpoint)

        accuracies_train.append(check_accuracy(train_loader, model, device=DEVICE))
        accuracies_val.append(check_accuracy(val_loader, model, device=DEVICE))

                                                                    
        save_predictions_as_masks(                                  #Get masks images from our predictions
            val_loader, model, folder="/content/post_processing/saved_predictions/", 
            device=DEVICE
        )

        plot_loss(losses, losses, "/content/post_processing/plots/loss.png", epoch)
        plot_accuracy(accuracies_train, accuracies_val, "/content/post_processing/plots/accuracy.png", epoch)


if __name__ == "__main__":                                          #Needed for NUM_WORKERS
    main()