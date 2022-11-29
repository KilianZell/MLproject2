How to run:
(0) Download all the recquired packages

(1) Creat the same folder steucture as displayed in "folder_struct.png"

(2) Load sat images (100 images) in  data/images

(3) Load groundtruth images (100 images) in  data/groundtruth

(4) Load the test images (50 images) in data/test_set_images

(5) Prepare the data set by running the following command:
    
    -> python3 data_augmentation.py

(6) Train your model with the following command:
    
    -> python3 train.py
    
    (validation images are displayed in "saved_prediction)

(7) Generate submission masks with the following command:
    
    -> python3 submission.py
    
    (masks submission sare generated in "submission")
    
(8) Get submission file with the following command:
    
    -> python3 mask_to_submission.py
    
    (you can now submit "submission.csv" on aicrownd)
