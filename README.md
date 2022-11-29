# MLproject2

Kilian_______
New version of the pythorch Unet in the folder "Kilian_main". Updates:

- Automatization of data augmentation (augmented images goes in the correct folder automatically)
- Revised data augmentation, for each image:
    - the original image is cropped at each corner (size: 256x256)
    - the original image is cropped in its center (size: 256x256)
    - the original image is resized (size: 256x256)
    
   -> 6 augmented images are generated from one original image
 NOTE: During training, each training image will additionnaly undergo random rotations and flips based on probalities (20%)
 - Split is automatically performed in data_augmentation.py
 - Minor changes in training to enhance performance
 - Comments were added
