import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
import cv2
import paths
import configparser

config = configparser.ConfigParser()
config.read(os.path.join('InconsistencyMasks', 'config.ini'))

SEED = int(config['DEFAULT']['SEED'])


def split_and_resize_dataset(images_dir, masks_dir, output_dir, subset_names, split_ratio, seed=SEED):

    image_files = os.listdir(images_dir)
    mask_files = os.listdir(masks_dir)
    
    # Assert that all image files have corresponding mask files
    assert set(image_files) == set(mask_files), "Images and masks don't match"

    # Split into train_full, val, test
    part_a, part_b = train_test_split(image_files, test_size=split_ratio, random_state=seed)

    subsets = {subset_names[0]: part_a, subset_names[1]: part_b}
    for subset, files in subsets.items():
        print(f'Processing {subset} data...')
        img_subset_dir = os.path.join(output_dir, subset, 'images')
        mask_subset_dir = os.path.join(output_dir, subset, 'masks')
        os.makedirs(img_subset_dir, exist_ok=True)
        os.makedirs(mask_subset_dir, exist_ok=True)
        for f in files:

            image = cv2.imread(os.path.join(images_dir, f))
            cv2.imwrite(os.path.join(img_subset_dir, f), image)

            mask = cv2.imread(os.path.join(masks_dir, f))
            cv2.imwrite(os.path.join(mask_subset_dir, f), mask)


subset_names_train = ['train_labeled', 'train_unlabeled']


split_and_resize_dataset(paths.ISIC_2018_TRAIN_FULL_IMAGES_DIR, 
                         paths.ISIC_2018_TRAIN_FULL_MASKS_DIR,  
                         paths.ISIC_2018_BASE_DIR, 
                         subset_names_train, 
                         0.9,
                         SEED)




