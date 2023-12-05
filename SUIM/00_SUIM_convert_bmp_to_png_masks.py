
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SUIM_class_mapping import COLOR_TO_CLASS_MAPPING_SUIM
import cv2
import numpy as np
from tqdm import tqdm
import paths




def convert_color_bmp_to_class_mask_png(bmp_file_path, output_path, color_to_class_mapping):
    image = cv2.imread(bmp_file_path)

    # Convert color space from BGR to RGB because OpenCV uses BGR as default
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = np.where(image < 128, 0, 255)

    # Create a new array filled with zeros to store the class IDs
    class_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Map the colors to classes
    for rgb, class_id in color_to_class_mapping.items():
        mask = np.all(image == rgb, axis=-1)
        class_image[mask] = class_id

    # Save the class image as a PNG
    cv2.imwrite(output_path, class_image)



def convert_all_masks_in_folder(input_folder, output_folder, color_to_class_mapping):

    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all .bmp files in the input folder
    bmp_files = [f for f in os.listdir(input_folder) if f.endswith('.bmp')]

    # Process each .bmp file
    for bmp_file in tqdm(bmp_files):
        # Determine the full input path and the output path
        bmp_file_path = os.path.join(input_folder, bmp_file)
        output_path = os.path.join(output_folder, f'{os.path.splitext(bmp_file)[0]}.png')

        # Convert the .bmp image
        convert_color_bmp_to_class_mask_png(bmp_file_path, output_path, color_to_class_mapping)



convert_all_masks_in_folder(paths.SUIM_ORG_TRAIN_VAL_MASKS_BMP_DIR, paths.SUIM_ORG_TRAIN_VAL_MASKS_PNG_DIR, COLOR_TO_CLASS_MAPPING_SUIM)
convert_all_masks_in_folder(paths.SUIM_ORG_TEST_MASKS_BMP_PATH, paths.SUIM_ORG_TEST_MASKS_PNG_PATH, COLOR_TO_CLASS_MAPPING_SUIM)
