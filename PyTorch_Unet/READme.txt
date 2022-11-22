Comand lines:
(1) python3 train.py
        -> train the Unet model
                (with BATCH_SIZE = 16, NUM_EPOCHS = 3 and 
                IMAGE_HEIGHT/WIDTH = 100 it takes arround 10 minutes)

(2) python3 submission.py
        -> Generate masks with trained model from test sat images
                (takes ~3minutes)

(3) python3 mask_to_submission
        -> Generate submission file (submission.csv)

*Data folders needs to be filled with masks and sat images following the folder structure