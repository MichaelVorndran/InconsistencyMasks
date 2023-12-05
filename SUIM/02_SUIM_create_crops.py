
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import cv2
from tqdm import tqdm
import paths


def random_crop(img, mask, target_crop_size=256, min_crop_size=256, max_crop_size=512):
    height, width = img.shape[:2]

    # Randomly select a value between min_crop_size and either the max_crop_size or the maximum dimension of the image.
    crop_size = np.random.randint(min_crop_size, min(max_crop_size, max(height, width)))

    if height >= crop_size and width >= crop_size:
        x = random.randint(0, width - crop_size)
        y = random.randint(0, height - crop_size)
        
        img_crop = img[y:y + crop_size, x:x + crop_size]
        mask_crop = mask[y:y + crop_size, x:x + crop_size]
        
        # Resize to target_crop_size
        img_crop = cv2.resize(img_crop, (target_crop_size, target_crop_size))
        # Use nearest neighbor interpolation for the mask
        mask_crop = cv2.resize(mask_crop, (target_crop_size, target_crop_size), interpolation=cv2.INTER_NEAREST)
        
        return img_crop, mask_crop
    else:
        # Resize the images if they are smaller than the crop_size
        img = cv2.resize(img, (target_crop_size, target_crop_size))
        mask = cv2.resize(mask, (target_crop_size, target_crop_size), interpolation=cv2.INTER_NEAREST)
        
        return img, mask


def create_random_crops(image_folder, mask_folder, main_output_path, num_crops_per_image):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')

    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)


    for i, image_file in tqdm(enumerate(image_files)):

        # Open the image and its corresponding mask
        image = cv2.imread(os.path.join(image_folder, image_file))
        mask = cv2.imread(os.path.join(mask_folder, f'{image_file[:-4]}.png'), cv2.IMREAD_GRAYSCALE)  # assume the masks are grayscale images
        
        # Take a random crop from the image and the mask
        for j in range(0,num_crops_per_image):
            image_crop, mask_crop = random_crop(image, mask)
            
            cv2.imwrite(os.path.join(images_path_out, f'{image_file[:-4]}_{i}_{j}.png'), image_crop)
            cv2.imwrite(os.path.join(masks_path_out, f'{image_file[:-4]}_{i}_{j}.png'), mask_crop)





create_random_crops(paths.SUIM_ORG_TRAIN_FULL_IMAGES_DIR, 
                    paths.SUIM_ORG_TRAIN_FULL_MASKS_DIR, 
                    paths.SUIM_TRAIN_FULL_MAIN_DIR, 
                    2)

create_random_crops(paths.SUIM_ORG_TRAIN_LABELED_IMAGES_DIR, 
                    paths.SUIM_ORG_TRAIN_LABELED_MASKS_DIR, 
                    paths.SUIM_TRAIN_LABELED_MAIN_DIR, 
                    2)

create_random_crops(paths.SUIM_ORG_TRAIN_UNLABELED_IMAGES_DIR, 
                    paths.SUIM_ORG_TRAIN_UNLABELED_MASKS_DIR, 
                    paths.SUIM_TRAIN_UNLABELED_MAIN_DIR, 
                    2)

create_random_crops(paths.SUIM_ORG_VAL_IMAGES_DIR, 
                    paths.SUIM_ORG_VAL_MASKS_DIR, 
                    paths.SUIM_VAL_MAIN_DIR, 
                    2)

create_random_crops(paths.SUIM_ORG_TEST_IMAGES_DIR, 
                    paths.SUIM_ORG_TEST_MASKS_PNG_PATH, 
                    paths.SUIM_TEST_MAIN_DIR, 
                    2)





