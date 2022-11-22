import torchvision
import torch
from torch.utils.data import DataLoader
from data_processing import RoadSegDataset


def save_trained_model(state, filename="models/my_checkpoint.pth.tar"): 
    #Saving current model (will be called at the end of every epoch)
    print("=> Saving model")
    torch.save(state, filename)
    print("=> Model saved at: ", filename)

def load_trained_model(checkpoint, model):
    #Load model to get predictions
    print("=> Loading model")
    model.load_state_dict(checkpoint["state_dict"])
    print("=> Model loaded")

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, 
                batch_size, train_transform, val_transform, 
                num_workers=4, pin_memory=True):
    #Prepare data to load in the model for training
    
    train_data = RoadSegDataset(    image_dir=train_dir,
                                    groundtruth_dir=train_maskdir,
                                    function=train_transform   )

    train_DataLoader = DataLoader(  train_data,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    shuffle=True   )

    val_data = RoadSegDataset(      image_dir=val_dir,
                                    groundtruth_dir=val_maskdir,
                                    function=val_transform    )

    val_DataLoader = DataLoader(    val_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                shuffle=False   )

    return train_DataLoader, val_DataLoader

def check_accuracy(loader, model, device="cpu"):
    #checks accuracy. Will be called at the end of each epoch.
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))                 #Compute the sigmoid of the model output
            preds = (preds > 0.5).float()                   #Evalute the sigmoid of the model output with a threshold (output 1 or 0)
            num_correct += (preds == y).sum()               #Get number of correct pixels 
            num_pixels += torch.numel(preds)                #Get total number of pixels
            dice_score += (2 * (preds * y).sum()) / (       #Compute the dice score -> to be replace by F1 score
                (preds + y).sum() + 1e-8
            )

    print( f"Accuracy: {num_correct/num_pixels*100:.2f}" )  #Output the scores
    print(f"Dice score: {dice_score/len(loader)}")
    
    model.train()                                           #Continue to train the model

def save_predictions_as_masks(DataLoader, model, folder="saved_predictions", device="cpu"):
    #Save predictions as binary mask images
    model.eval()

    for idx, (x, y) in enumerate(DataLoader):
        x = x.to(device=device)

        with torch.no_grad():                               #Saves memory and time
            preds = torch.sigmoid(model(x))                 #Compute the sigmoid of the model output
            preds = (preds > 0.5).float()                   #Evalute the sigmoid of the model output with a threshold (output 1 or 0)

        n = 1
        for pred in preds:                                  #Generate and save predictions as mask images
            torchvision.utils.save_image(
            pred, f"{folder}/pred_{n}.png"
            )
            n += 1

        torchvision.utils.save_image(                       #Concatanated image of the validation masks
            y.unsqueeze(1), f"{folder}{idx}.png"
        )

    model.train()                                           #Continue training