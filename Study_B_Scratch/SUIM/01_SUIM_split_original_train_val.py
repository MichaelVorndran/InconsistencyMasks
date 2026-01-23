
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from sklearn.model_selection import train_test_split
import paths




def split_and_resize_dataset(images_dir, masks_dir, output_dir, subset_names, test_size=0.1):

     # Check that subset_names is a list of exactly two strings
    assert isinstance(subset_names, list), "subset_names must be a list"
    assert len(subset_names) == 2, "subset_names must have exactly two elements"
    assert all(isinstance(item, str) for item in subset_names), "all elements in subset_names must be strings"
    

    image_files = os.listdir(images_dir)
    mask_files = os.listdir(masks_dir)

    # Remove file extensions to match image and mask pairs
    image_files_no_ext = [os.path.splitext(f)[0] for f in image_files]
    mask_files_no_ext = [os.path.splitext(f)[0] for f in mask_files]

    # Assert that all image files have corresponding mask files
    assert set(image_files_no_ext) == set(mask_files_no_ext), "Images and masks don't match"

    # Split into train_full, val, test
    train_files, val_files = train_test_split(image_files_no_ext, test_size=test_size, random_state=42)

    subsets = {subset_names[0]: train_files, subset_names[1]: val_files}
    for subset, files in subsets.items():
        print(f'Processing {subset} data...')
        img_subset_dir = os.path.join(output_dir, subset, 'images')
        mask_subset_dir = os.path.join(output_dir, subset, 'masks')
        os.makedirs(img_subset_dir, exist_ok=True)
        os.makedirs(mask_subset_dir, exist_ok=True)
        for f in files:

            img = cv2.imread(os.path.join(images_dir, f + '.jpg'))
            cv2.imwrite(os.path.join(img_subset_dir, f + '.jpg'), img)

            mask = cv2.imread(os.path.join(masks_dir, f + '.png'))
            cv2.imwrite(os.path.join(mask_subset_dir, f + '.png'), mask)




# split train_val
split_and_resize_dataset(paths.SUIM_ORG_TRAIN_VAL_IMAGES_DIR, 
                         paths.SUIM_ORG_TRAIN_VAL_MASKS_PNG_DIR, 
                         paths.SUIM_ORG_DATA_DIR, 
                         ['train_full', 'val'], 
                         0.1) 

# split train in labeled and unlabeled
split_and_resize_dataset(paths.SUIM_ORG_TRAIN_FULL_IMAGES_DIR, 
                         paths.SUIM_ORG_TRAIN_FULL_MASKS_DIR, 
                         paths.SUIM_ORG_DATA_DIR, 
                         ['train_unlabeled', 'train_labeled'], 
                         0.1) 


