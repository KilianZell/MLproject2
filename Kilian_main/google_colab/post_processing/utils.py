import torchvision
import torch
from torch.utils.data import DataLoader
from data_processing import RoadSegDataset


def save_trained_model(state, filename="models/my_checkpoint.pth.tar"): 
    """
    Function that saves current model. It will be called at the end of each epoch
    so that if the program stops unexpectively, current mode, can be retreive.
    Input:
        - state: current epoch state
        - filename: path/name of the model
    """
    print("=> Saving model")
    torch.save(state, filename)
    print("=> Model saved at: ", filename)

def load_trained_model(checkpoint, model):
    """
    Function that loads a model that was previously saved. It will allow users to 
    use pretrained models.
    Input:
        - checkpoint: the path to the saved model.
        - model: the UNET used
    """
    print("=> Loading model")
    model.load_state_dict(checkpoint["state_dict"])
    print("=> Model loaded")

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, 
                batch_size, train_transform, val_transform, 
                num_workers=4, pin_memory=True):
    """
    Function that formats the data, in order to load it to the model and 
    initiate training. (Initiate the torch.utils.data.DataLoader)
    Input:
        - train_dir: path to folder with sat images dedicated to training
        - train_maskdir: path to folder with groundtruth images dedicated to training
        - val_dir: ath to folder with sat images dedicated to validation
        - val_maskdir: path to folder with groundtruth images dedicated to validation
        - batch_size: number of image pairs per batch
        - train_transform: function applied to train dataset before training
        - val_transform: function applied to validation dataset before training
        - num_workers: tells the data loader instance how many sub-processes to use for data loading 
            (default 4)
        - pin_memory: Pinned memory is used to speed up a CPU to GPU memory copy operation
            (default True)
    Output:
        - train_DataLoader: train data ready to be loaded
        - val_DataLoader: validation data ready to be loaded
    """
    
    train_data = RoadSegDataset(    image_dir=train_dir,
                                    groundtruth_dir=train_maskdir,
                                    function=train_transform   )

    train_DataLoader = DataLoader(  train_data,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    drop_last = True,
                                    shuffle=True   )

    val_data = RoadSegDataset(      image_dir=val_dir,
                                    groundtruth_dir=val_maskdir,
                                    function=val_transform    )

    val_DataLoader = DataLoader(    val_data,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    drop_last = True,
                                    shuffle=False   )

    return train_DataLoader, val_DataLoader

def check_accuracy(loader, model, device="cpu"):
    """
    Function that evalute and print the accurucy of the model. 
    (Will be called at the end of each epoch)
    Input: 
        - loader: the Dataloader that we want to evaluate (val_DataLoader in most cases)
        - model: the UNET used
        - device: Componant that will recieve the data (default "cpu")
    """
    #checks accuracy. Will be called at the end of each epoch.
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))                 #Compute the sigmoid of the model output
            preds = (preds > 0.5).float()                   #Evalute the sigmoid of the model output with a threshold (output 1 or 0)
            num_correct += (preds == y).sum()               #Get number of correct pixels 
            num_pixels += torch.numel(preds)                #Get total number of pixels

    print( f"Accuracy: {num_correct/num_pixels*100:.2f}" )  #Output the scores
    
    model.train()                                           #Continue to train the model

    return (num_correct/num_pixels*100).cpu()

def save_predictions_as_masks(DataLoader, model, folder="saved_predictions", device="cpu"):
    """
    Function that saves predictions (of validation images) as binary mask images.
    Input:
        - DataLoader:
        - model: the UNET used
        - folder: path to were predictions are saved (default: "saved_predictions")
        - device: Componant that will recieve the data (default "cpu")
    """
    model.eval()

    n = 1
    for idx, (x, y) in enumerate(DataLoader):
        x = x.to(device=device)

        with torch.no_grad():                               #Saves memory and time
            preds = torch.sigmoid(model(x))                 #Compute the sigmoid of the model output
            preds = (preds > 0.5).float()                   #Evalute the sigmoid of the model output with a threshold (output 1 or 0)

        for pred in preds:                                  #Generate and save predictions as mask images
            torchvision.utils.save_image(
            pred, f"{folder}/pred_{n}.png"
            )
            n += 1

        torchvision.utils.save_image(                       #Concatanated image of the validation masks
            y.unsqueeze(1), f"{folder}{idx}.png"
        )

    model.train()                                           #Continue training