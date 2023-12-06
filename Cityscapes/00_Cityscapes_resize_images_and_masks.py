
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import paths
import configparser

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
config_dir = os.path.join(parent_dir, 'config.ini')

config = configparser.ConfigParser()
config.read(config_dir)

RESIZE_FACTOR = float(config['CITYSCAPES']['RESIZE_FACTOR'])


def resize_image(image_path, factor, base=16, is_mask=False):
    image = cv2.imread(image_path)

    # Calculate the new size
    new_size = (int(image.shape[1] * factor), int(image.shape[0] * factor))

    # Adjust the new size to the next multiple of 'base'
    new_size = (base * np.ceil(new_size[0] / base).astype(int),
                base * np.ceil(new_size[1] / base).astype(int))

    if is_mask:
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
    else:
        image = cv2.resize(image, new_size)  # OpenCV uses bilinear interpolation by default

    return image

def process_images(img_dir, mask_dir, save_img_dir, save_mask_dir, factor, base=16):

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    for root, _, files in os.walk(img_dir):
        _, city = os.path.split(root)
        print(f'processing {city}')

        for file in files:
            if file.endswith('.png'):  # adjust this if your images are not png
                common_name = "_".join(file.split("_")[:-1])
                mask_name = common_name + '_gtFine_labelIds.png'  # adjust this if your masks are not png
                mask_path = os.path.join(mask_dir, city, mask_name)

                if os.path.exists(mask_path):
                    # resize and save image
                    img_path = os.path.join(root, file)
                    resized_img = resize_image(img_path, factor, base, is_mask=False)
                    save_img_name = common_name + '.png'
                    cv2.imwrite(os.path.join(save_img_dir, save_img_name), resized_img)

                    # resize and save mask
                    resized_mask = resize_image(mask_path, factor, base, is_mask=True)
                    # Increase the pixel values by 1
                    resized_mask = np.where(resized_mask > 0, resized_mask + 1, resized_mask)

                    save_mask_name = common_name + '.png'
                    cv2.imwrite(os.path.join(save_mask_dir, save_mask_name), resized_mask)


process_images(paths.CITYSCAPES_ORG_TRAIN_IMAGES_DIR, 
               paths.CITYSCAPES_ORG_TRAIN_MASKS_DIR, 
               paths.CITYSCAPES_TRAIN_FULL_IMAGES_DIR, 
               paths.CITYSCAPES_TRAIN_FULL_MASKS_DIR, 
               RESIZE_FACTOR)

process_images(paths.CITYSCAPES_ORG_VAL_IMAGES_DIR, 
               paths.CITYSCAPES_ORG_VAL_MASKS_DIR, 
               paths.CITYSCAPES_ORG_VAL_TEST_IMAGES_DIR, 
               paths.CITYSCAPES_ORG_VAL_TEST_MASKS_DIR, 
               RESIZE_FACTOR)



