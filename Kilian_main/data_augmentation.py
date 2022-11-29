
from PIL import Image
import glob
from skimage import io
import splitfolders


images_path=        "your_path/data/images"             #Path to the folder with the satelite images
masks_path=         "your_path/data/groundtruth"        #Path to the folder with the groundtruth images
images_aug_path =   "your_path/data/aug_data/images"    #Path to the augmented sat images
masks_aug_path =    "your_path/data/aug_data/masks"     #Path to the augmented groundtruth images (called masks)


def trasnform_image(images_path, images_aug_path):
    """
    A function that transform all images of a given folder into 6 different 
    images by croping each image in its corners (256x256),cropping the image 
    at its center (256x256) and finnaly resizing the entire image (256x256). 
    Output images are saved in a given folder.

    input: 
        - images_path: path to the images that need to be augmented
        - images_aug_path: path where augmented images should be stored
    """

    i = 1
    for filename in glob.glob(images_path + '/*.png'):
        im=Image.open(filename)

        im_crop1 = im.crop((0, 0, 256, 256))                            #upper left corner
        new_image_path= "%s/aug_img_%s.png" %(images_aug_path, i)
        im_crop1 = im_crop1.save(new_image_path)
        i += 1
        print(new_image_path, end="\r")

        im_crop2 = im.crop((0, 144, 256, 400))                          #lower left corner
        new_image_path= "%s/aug_img_%s.png" %(images_aug_path, i)
        im_crop2 = im_crop2.save(new_image_path)
        i += 1
        print(new_image_path, end="\r")

        im_crop3 = im.crop((144, 0, 400, 256))                          #upper right corner
        new_image_path= "%s/aug_img_%s.png" %(images_aug_path, i)
        im_crop3 = im_crop3.save(new_image_path)
        i += 1
        print(new_image_path, end="\r")

        im_crop4 = im.crop((144, 144, 400, 400))                        #lower right corner
        new_image_path= "%s/aug_img_%s.png" %(images_aug_path, i)
        im_crop4 = im_crop4.save(new_image_path)
        i += 1
        print(new_image_path, end="\r")

        im_crop5 = im.crop((72, 72, 328, 328))                          #center
        new_image_path= "%s/aug_img_%s.png" %(images_aug_path, i)
        im_crop5 = im_crop5.save(new_image_path)
        i += 1
        print(new_image_path, end="\r")

        im_crop6 = im.resize((256, 256))                                #resize entire imge (256x256)
        new_image_path= "%s/aug_img_%s.png" %(images_aug_path, i)
        im_crop6 = im_crop6.save(new_image_path)
        i += 1
        print(new_image_path, end="\r")


def data_aug(images_path, masks_path, images_aug_path, masks_aug_path):
    """
    A function that apply data augmentation given input and output folders.

    input: 
        - images_path: path to the sat images that need to be augmented
        - masks_path: path to the groudtruth images that need to be augmented
        - images_aug_path: path where augmented sat images should be stored
        - masks_aug_path: path where augmented groundtruth images should be stored
    """
    trasnform_image(images_path, images_aug_path)
    trasnform_image(masks_path, masks_aug_path)


if __name__ == "__main__": 
    data_aug(images_path, masks_path, images_aug_path, masks_aug_path)                  #data augmentation

    input_folder = "your_path/data/aug_data"
    output =       "your_path/data/processing"

    splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.95, 0.05, 0.0))    #Split the data between train and validation dataset
