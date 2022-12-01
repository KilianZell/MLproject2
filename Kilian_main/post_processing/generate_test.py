import os
from PIL import Image  
import PIL  



parent_dir = "/content/post_processing/data_post/test_set_images"
image_dir = "/content/submission/"
n_tests = 50

for i in range (1, n_tests+1):
    dir = parent_dir + "/test_" + str(i)

    os.mkdir(dir)

    picture = Image.open(image_dir + "pred_" + str(i) + ".png")  
    picture = picture.save(dir + "/test_" + str(i) + ".png") 

    print(dir)