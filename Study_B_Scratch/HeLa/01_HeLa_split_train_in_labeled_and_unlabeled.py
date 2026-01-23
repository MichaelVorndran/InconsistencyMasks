import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import random
from tqdm import tqdm
import paths
import configparser

# Configuration reading
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
config_dir = os.path.join(parent_dir, 'config.ini')

config = configparser.ConfigParser()
config.read(config_dir)

SEED = int(config['DEFAULT']['SEED'])
USE_MOD_POS_SIZE = config['HELA']['USE_MOD_POS_SIZE'].lower() == 'true'

# Directory Paths
root_dir = paths.HELA_TRAIN_FULL_DIR
labeled_dir = paths.HELA_TRAIN_LABELED_DIR
unlabeled_dir = paths.HELA_TRAIN_UNLABELED_DIR

# Folder names
folders = ["brightfield", "alive", "dead"]
folders.append("mod_position" if USE_MOD_POS_SIZE else "position")

# Shuffle and split
bf_img_names = os.listdir(os.path.join(root_dir, "brightfield"))
random.seed(SEED)
random.shuffle(bf_img_names)
split_idx = int(len(bf_img_names) * 0.10)
labeled_imgs, unlabeled_imgs = bf_img_names[:split_idx], bf_img_names[split_idx:]

# Directory Creation
for folder in folders:
    for dir in [labeled_dir, unlabeled_dir]:
        os.makedirs(os.path.join(dir, folder), exist_ok=True)

# File Copying
for folder in folders:
    for imagename in tqdm(labeled_imgs, desc=f"Copying labeled images from {folder}"):
        src = os.path.join(root_dir, folder, imagename)
        dst = os.path.join(labeled_dir, folder, imagename)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Source file {src} does not exist.")

    for imagename in tqdm(unlabeled_imgs, desc=f"Copying unlabeled images from {folder}"):
        src = os.path.join(root_dir, folder, imagename)
        dst = os.path.join(unlabeled_dir, folder, imagename)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Source file {src} does not exist.")
