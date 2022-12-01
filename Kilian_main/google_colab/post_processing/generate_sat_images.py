import os
from PIL import Image  
import PIL  



parent_dir = "/content/post_processing/data_post/sat_images/sat_test"
image_dir = "/content/data/images/"
n_tests = 100

for i in range (1, n_tests+1):
    dir = parent_dir + "/test_" + str(i)

    os.mkdir(dir)

    picture = Image.open(image_dir + 'satImage_' + '%.3d' % i + '.png')  
    picture = picture.save(dir + "/test_" + str(i) + ".png") 

    print(dir)

    'satImage_' + '%.3d' % i + '.png'