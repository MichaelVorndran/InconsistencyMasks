import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import paths
from tqdm import tqdm
import configparser

config = configparser.ConfigParser()
config.read(os.path.join('InconsistencyMasks', 'config.ini'))

IMAGE_WIDTH = int(config['ISIC_2018']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['ISIC_2018']['IMAGE_HEIGHT'])

image_dirs = [
    (paths.ISIC_2018_ORG_TRAIN_IMAGES_DIR, paths.ISIC_2018_TRAIN_FULL_IMAGES_DIR),
    (paths.ISIC_2018_ORG_VAL_IMAGES_DIR, paths.ISIC_2018_VAL_IMAGES_DIR),
    (paths.ISIC_2018_ORG_TEST_IMAGES_DIR, paths.ISIC_2018_TEST_IMAGES_DIR)
]

mask_dirs = [
    (paths.ISIC_2018_ORG_TRAIN_MASKS_DIR, paths.ISIC_2018_TRAIN_FULL_MASKS_DIR),
    (paths.ISIC_2018_ORG_VAL_MASKS_DIR, paths.ISIC_2018_VAL_MASKS_DIR),
    (paths.ISIC_2018_ORG_TEST_MASKS_DIR, paths.ISIC_2018_TEST_MASKS_DIR)
]

def preprocess_data(dir_pairs, is_mask=False):
    for in_dir, out_dir in dir_pairs:
        if os.path.exists(in_dir):
            os.makedirs(out_dir, exist_ok=True)

            for filename in tqdm(os.listdir(in_dir)):
                file_path = os.path.join(in_dir, filename)
                
                try:
                    image = cv2.imread(file_path)
                    if image is None:
                        print(f'Failed to load {"mask" if is_mask else "image"}: {file_path}')
                        continue

                    resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

                    if is_mask:
                        out_path = os.path.join(out_dir, f'{filename[:-17]}.png')   # remove _segmentation from mask name
                    else:
                        out_path = os.path.join(out_dir, f'{filename[:-4]}.png')

                    cv2.imwrite(out_path, resized_image)
                except Exception as e:
                    print(f'Failed to process {"mask" if is_mask else "image"}: {file_path}. Error: {str(e)}')

preprocess_data(image_dirs)
preprocess_data(mask_dirs, is_mask=True)
