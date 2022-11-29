How to run:
1. Download all the recquired packages

2. Creat the same folder steucture as displayed in "folder_struct.png"

3. Load sat images (100 images) in  data/images

4. Load groundtruth images (100 images) in  data/groundtruth

5. Load the test images (50 images) in data/test_set_images

6. Prepare the data set by running the following command:
    
    -> python3 data_augmentation.py

7. Train your model with the following command:
    
    -> python3 train.py
    
    (validation images are displayed in "saved_prediction)

8. Generate submission masks with the following command:
    
    -> python3 submission.py
    
    (masks submission sare generated in "submission")
    
9. Get submission file with the following command:
    
    -> python3 mask_to_submission.py
    
    (you can now submit "submission.csv" on aicrownd)
