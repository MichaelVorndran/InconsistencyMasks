import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from tqdm import tqdm
import paths
from functions import get_pos_contours, get_min_dist
import configparser

config = configparser.ConfigParser()
config.read(os.path.join('IBAs', 'config.ini'))


class ImagePreprocessor:
    def __init__(self, img_path, crop_path, crop_size, overlap, use_mod_pos_size=True):
        self.img_path = img_path
        self.crop_path = crop_path
        self.crop_size = crop_size
        self.overlap = overlap
        self.use_mod_pos_size = use_mod_pos_size
        self.channels = ['brightfield', 'alive', 'dead']
        if use_mod_pos_size:
            self.channels.append('mod_position')
        else:
            self.channels.append('position')

    def get_positions(self, img_height, img_width):
        """
        Calculate positions for cutouts based on given image dimensions, crop size, and overlap.
        
        Parameters:
        -----------
        img_height : int
            The height of the image.
        img_width : int
            The width of the image.
            
        Returns:
        --------
        positions : list of tuple
            A list of positions for cutouts in the form [(x1, y1), (x2, y2), ...]
        
        Raises:
        -------
        AssertionError
            If image dimensions are not positive or overlap is not between 0 and 1.
        """
        assert img_height > 0 and img_width > 0, 'Image dimensions should be positive.'
        assert 0 <= self.overlap < 1, 'Overlap should be between 0 and 1.'

        # Calculating the number of movements in x and y directions
        x_count = round(img_width / (self.crop_size * (1 - self.overlap)))
        y_count = round(img_height / (self.crop_size * (1 - self.overlap)))

        # Calculating the movement in x and y directions
        x_move = img_width / x_count
        y_move = img_height / y_count

        # Initializing the positions array
        positions = []

        # Iterating through x and y coordinates and adding to the positions list
        for i in range(x_count):
            for j in range(y_count):
                x = int(i * x_move)
                y = int(j * y_move)

                # Correcting the coordinates if necessary
                x = min(x, img_width - self.crop_size)
                y = min(y, img_height - self.crop_size)

                positions.append((x, y))

        return positions
    
        

    def process_image(self, img_name):
        """
        Process the image and generate crops.
        """

        brightfield_img = cv2.imread(os.path.join(self.img_path, 'brightfield', img_name))
        if brightfield_img is None:
            raise FileNotFoundError(f'Image file {img_name} not found in {self.img_path}/brightfield.')
        
        brightfield_gray = cv2.cvtColor(brightfield_img, cv2.COLOR_BGR2GRAY)
        img_height, img_width = brightfield_gray.shape
        positions = self.get_positions(img_height, img_width)

        for count, position in enumerate(positions):
            self.process_crops(position, img_name, count)


    
    def process_crops(self, position, img_name, count):
        """
        Create cutouts for each position and save them.

        Parameters:
        -----------
        position : tuple
            The (x, y) position for the crop.
        img_name : str
            The name of the image file.
        count : int
            The count for naming the cropped images.

        Raises:
        -------
        FileNotFoundError
            If the image file cannot be found.
        """
        x1, y1 = position
        x2 = self.crop_size + position[0]
        y2 = self.crop_size + position[1]
        final_crop_name = f'{img_name[:-4]}_{count}'

        for channel in self.channels:
            modified_img_name = f'{img_name[:-4]}.png' if channel != 'brightfield' else img_name
            img_channel_path = os.path.join(self.img_path, channel, modified_img_name)
            
            if os.path.exists(img_channel_path):
                img_channel = cv2.imread(img_channel_path)
                if img_channel is None:
                    raise FileNotFoundError(f'Image file {modified_img_name} not found in {img_channel_path}.')
                
                img_channel_gray = cv2.cvtColor(img_channel, cv2.COLOR_BGR2GRAY)

                if channel != 'brightfield':
                    _, img_channel_thresh = cv2.threshold(img_channel_gray, 10, 255, cv2.THRESH_BINARY)
                else:
                    img_channel_thresh = img_channel_gray

                img_cutout = img_channel_thresh[y1:y2 , x1:x2]
                img_crop_path = os.path.join(self.crop_path, channel)
                if not os.path.exists(img_crop_path):
                    os.makedirs(img_crop_path)
                cv2.imwrite(os.path.join(img_crop_path, f'{final_crop_name}.png'), img_cutout)



    def mod_pos_size(self, in_path, out_path, max_pos_circle_size=8, min_pos_circle_size=3):
        """
        Modify position size based on distances between positions.

        Parameters:
        -----------
        in_path : str
            Path to the input directory containing images.
        out_path : str
            Path to the output directory where modified images will be saved.
        max_pos_circle_size : int, optional
            Maximum allowed circle size. Default is 8.
        min_pos_circle_size : int, optional
            Minimum allowed circle size. Default is 3.

        Raises:
        -------
        FileNotFoundError
            If the input directory or an image file does not exist.
        """
        if not os.path.exists(in_path):
            raise FileNotFoundError(f'Input directory {in_path} does not exist.')

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for posimg in tqdm(os.listdir(in_path)):
            imreadpath = os.path.join(in_path, posimg)
            img = cv2.imread(imreadpath)
            if img is None:
                raise FileNotFoundError(f'Image file {posimg} not found in {in_path}.')
            
            positions = get_pos_contours(img)
            h, w, c = img.shape
            out_img = np.zeros((h, w, 3), np.uint8)

            for pos in positions:
                min_dist = get_min_dist(pos, positions)
                circle_size = int(min_dist // 4)
                circle_size = max(min(circle_size, max_pos_circle_size), min_pos_circle_size)
                cv2.circle(out_img, (pos[0], pos[1]), circle_size, (255, 255, 255), -1)

            out_img = cv2.blur(out_img, (2, 2))
            out_img[out_img < 254] = 0
            imwritepath = os.path.join(out_path, posimg)
            cv2.imwrite(imwritepath, out_img)



def main():
    base_img_path = paths.HELA_ORG_DIR # 'D:/EBAS/Hela/Original_Data/'
    base_crop_path = paths.HELA_BASE_DIR    # 'D:/EBAS/Hela/'
    data_sets = ['train_full', 'val', 'test']

    USE_MOD_POS_SIZE = config['HELA']['USE_MOD_POS_SIZE'].lower() == 'true'
    

    for data_set in data_sets:
        img_path = os.path.join(base_img_path, data_set)
        crop_path = os.path.join(base_crop_path, data_set)
        preprocessor = ImagePreprocessor(img_path, crop_path, crop_size=256, overlap=0.6, use_mod_pos_size=USE_MOD_POS_SIZE)

        
        bf_path = os.path.join(img_path, 'brightfield')
        if os.path.exists(bf_path):
            for img_name in tqdm(os.listdir(bf_path)):
                try:
                    preprocessor.process_image(img_name)
                except Exception as e:
                    print(e)



if __name__ == '__main__':
    main()

