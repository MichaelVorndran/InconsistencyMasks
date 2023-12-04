import os
import configparser

config = configparser.ConfigParser()
config.read(os.path.join('IM', 'config.ini'))

ISIC_2018_BASE_DIR = config['ISIC_2018']['BASE_DIR']

ISIC_2018_ORG_TRAIN_IMAGES_DIR = os.path.join(ISIC_2018_BASE_DIR, 'original_data', 'train', 'Images')
ISIC_2018_ORG_TRAIN_MASKS_DIR = os.path.join(ISIC_2018_BASE_DIR, 'original_data', 'train', 'GT')
ISIC_2018_ORG_VAL_IMAGES_DIR = os.path.join(ISIC_2018_BASE_DIR, 'original_data', 'val', 'Images')
ISIC_2018_ORG_VAL_MASKS_DIR = os.path.join(ISIC_2018_BASE_DIR, 'original_data', 'val', 'GT')
ISIC_2018_ORG_TEST_IMAGES_DIR = os.path.join(ISIC_2018_BASE_DIR, 'original_data', 'test', 'Images')
ISIC_2018_ORG_TEST_MASKS_DIR = os.path.join(ISIC_2018_BASE_DIR, 'original_data', 'test', 'GT')

ISIC_2018_TRAIN_FULL_IMAGES_DIR = os.path.join(ISIC_2018_BASE_DIR, 'train_full', 'images')        
ISIC_2018_TRAIN_FULL_MASKS_DIR = os.path.join(ISIC_2018_BASE_DIR, 'train_full', 'masks')       

ISIC_2018_TRAIN_LABELED_MAIN_DIR = os.path.join(ISIC_2018_BASE_DIR, 'train_labeled')
ISIC_2018_TRAIN_LABELED_IMAGES_DIR = os.path.join(ISIC_2018_TRAIN_LABELED_MAIN_DIR, 'images')
ISIC_2018_TRAIN_LABELED_MASKS_DIR = os.path.join(ISIC_2018_TRAIN_LABELED_MAIN_DIR, 'masks')    

ISIC_2018_TRAIN_LABELED_AUG_MAIN_DIR = os.path.join(ISIC_2018_BASE_DIR, 'train_labeled_aug')
ISIC_2018_TRAIN_LABELED_AUG_IMAGES_DIR = os.path.join(ISIC_2018_TRAIN_LABELED_AUG_MAIN_DIR, 'images')
ISIC_2018_TRAIN_LABELED_AUG_MASKS_DIR = os.path.join(ISIC_2018_TRAIN_LABELED_AUG_MAIN_DIR, 'masks')

ISIC_2018_VAL_IMAGES_DIR = os.path.join(ISIC_2018_BASE_DIR, 'val', 'images')
ISIC_2018_VAL_MASKS_DIR = os.path.join(ISIC_2018_BASE_DIR, 'val', 'masks')
ISIC_2018_TEST_IMAGES_DIR = os.path.join(ISIC_2018_BASE_DIR, 'test', 'images')
ISIC_2018_TEST_MASKS_DIR = os.path.join(ISIC_2018_BASE_DIR, 'test', 'masks')
ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR = os.path.join(ISIC_2018_BASE_DIR, 'train_unlabeled', 'images')
ISIC_2018_TRAIN_UNLABELED_MASKS_DIR = os.path.join(ISIC_2018_BASE_DIR, 'train_unlabeled', 'masks')

ISIC_2018_MODEL_DIR = os.path.join(ISIC_2018_BASE_DIR, 'models')
ISIC_2018_CSV_DIR = os.path.join(ISIC_2018_BASE_DIR, 'csv')



HELA_BASE_DIR = config['HELA']['BASE_DIR']

HELA_ORG_DIR = os.path.join(HELA_BASE_DIR, 'original_data')
HELA_ORG_TRAIN_DIR = os.path.join(HELA_BASE_DIR, 'original_data', 'train')
HELA_ORG_TRAIN_BRIGHTFIELD_DIR = os.path.join(HELA_ORG_TRAIN_DIR, 'brightfield')
HELA_ORG_TRAIN_ALIVE_DIR = os.path.join(HELA_ORG_TRAIN_DIR, 'alive')
HELA_ORG_TRAIN_DEAD_DIR = os.path.join(HELA_ORG_TRAIN_DIR, 'dead')
HELA_ORG_TRAIN_POS_DIR = os.path.join(HELA_ORG_TRAIN_DIR, 'pos')
HELA_ORG_TRAIN_MOD_POS_DIR = os.path.join(HELA_ORG_TRAIN_DIR, 'mod_position')

HELA_ORG_VAL_DIR = os.path.join(HELA_BASE_DIR, 'original_data', 'val')
HELA_ORG_VAL_BRIGHTFIELD_DIR = os.path.join(HELA_ORG_VAL_DIR, 'brightfield')
HELA_ORG_VAL_ALIVE_DIR = os.path.join(HELA_ORG_VAL_DIR, 'alive')
HELA_ORG_VAL_DEAD_DIR = os.path.join(HELA_ORG_VAL_DIR, 'dead')
HELA_ORG_VAL_POS_DIR = os.path.join(HELA_ORG_VAL_DIR, 'pos')
HELA_ORG_VAL_MOD_POS_DIR = os.path.join(HELA_ORG_VAL_DIR, 'mod_position')

HELA_ORG_TEST_DIR = os.path.join(HELA_BASE_DIR, 'original_data', 'test')
HELA_ORG_TEST_BRIGHTFIELD_DIR = os.path.join(HELA_ORG_TEST_DIR, 'brightfield')
HELA_ORG_TEST_ALIVE_DIR = os.path.join(HELA_ORG_TEST_DIR, 'alive')
HELA_ORG_TEST_DEAD_DIR = os.path.join(HELA_ORG_TEST_DIR, 'dead')
HELA_ORG_TEST_POS_DIR = os.path.join(HELA_ORG_TEST_DIR, 'pos')
HELA_ORG_TEST_MOD_POS_DIR = os.path.join(HELA_ORG_TEST_DIR, 'mod_position')

HELA_TRAIN_FULL_DIR = os.path.join(HELA_BASE_DIR, 'train_full')
HELA_TRAIN_FULL_BRIGHTFIELD_DIR = os.path.join(HELA_TRAIN_FULL_DIR, 'brightfield')
HELA_TRAIN_FULL_ALIVE_DIR = os.path.join(HELA_TRAIN_FULL_DIR, 'alive')
HELA_TRAIN_FULL_DEAD_DIR = os.path.join(HELA_TRAIN_FULL_DIR, 'dead')
HELA_TRAIN_FULL_POS_DIR = os.path.join(HELA_TRAIN_FULL_DIR, 'pos')
HELA_TRAIN_FULL_MOD_POS_DIR = os.path.join(HELA_TRAIN_FULL_DIR, 'mod_position')

HELA_TRAIN_LABELED_DIR = os.path.join(HELA_BASE_DIR, 'train_labeled')
HELA_TRAIN_LABELED_BRIGHTFIELD_DIR = os.path.join(HELA_TRAIN_LABELED_DIR, 'brightfield')
HELA_TRAIN_LABELED_ALIVE_DIR = os.path.join(HELA_TRAIN_LABELED_DIR, 'alive')
HELA_TRAIN_LABELED_DEAD_DIR = os.path.join(HELA_TRAIN_LABELED_DIR, 'dead')
HELA_TRAIN_LABELED_POS_DIR = os.path.join(HELA_TRAIN_LABELED_DIR, 'pos')
HELA_TRAIN_LABELED_MOD_POS_DIR = os.path.join(HELA_TRAIN_LABELED_DIR, 'mod_position')

HELA_TRAIN_LABELED_AUG_DIR = os.path.join(HELA_BASE_DIR, 'train_labeled_aug')
HELA_TRAIN_LABELED_AUG_BRIGHTFIELD_DIR = os.path.join(HELA_TRAIN_LABELED_AUG_DIR, 'brightfield')
HELA_TRAIN_LABELED_AUG_ALIVE_DIR = os.path.join(HELA_TRAIN_LABELED_AUG_DIR, 'alive')
HELA_TRAIN_LABELED_AUG_DEAD_DIR = os.path.join(HELA_TRAIN_LABELED_AUG_DIR, 'dead')
HELA_TRAIN_LABELED_AUG_POS_DIR = os.path.join(HELA_TRAIN_LABELED_AUG_DIR, 'pos')
HELA_TRAIN_LABELED_AUG_MOD_POS_DIR = os.path.join(HELA_TRAIN_LABELED_AUG_DIR, 'mod_position')

HELA_TRAIN_UNLABELED_DIR = os.path.join(HELA_BASE_DIR, 'train_unlabeled')
HELA_TRAIN_UNLABELED_BRIGHTFIELD_DIR = os.path.join(HELA_TRAIN_UNLABELED_DIR, 'brightfield')
HELA_TRAIN_UNLABELED_ALIVE_DIR = os.path.join(HELA_TRAIN_UNLABELED_DIR, 'alive')
HELA_TRAIN_UNLABELED_DEAD_DIR = os.path.join(HELA_TRAIN_UNLABELED_DIR, 'dead')
HELA_TRAIN_UNLABELED_POS_DIR = os.path.join(HELA_TRAIN_UNLABELED_DIR, 'pos')
HELA_TRAIN_UNLABELED_MOD_POS_DIR = os.path.join(HELA_TRAIN_UNLABELED_DIR, 'mod_position')

HELA_VAL_DIR = os.path.join(HELA_BASE_DIR, 'val')
HELA_VAL_BRIGHTFIELD_DIR = os.path.join(HELA_VAL_DIR, 'brightfield')
HELA_VAL_ALIVE_DIR = os.path.join(HELA_VAL_DIR, 'alive')
HELA_VAL_DEAD_DIR = os.path.join(HELA_VAL_DIR, 'dead')
HELA_VAL_POS_DIR = os.path.join(HELA_VAL_DIR, 'pos')
HELA_VAL_MOD_POS_DIR = os.path.join(HELA_VAL_DIR, 'mod_position')

HELA_TEST_DIR = os.path.join(HELA_BASE_DIR, 'test')
HELA_TEST_BRIGHTFIELD_DIR = os.path.join(HELA_TEST_DIR, 'brightfield')
HELA_TEST_ALIVE_DIR = os.path.join(HELA_TEST_DIR, 'alive')
HELA_TEST_DEAD_DIR = os.path.join(HELA_TEST_DIR, 'dead')
HELA_TEST_POS_DIR = os.path.join(HELA_TEST_DIR, 'pos')
HELA_TEST_MOD_POS_DIR = os.path.join(HELA_TEST_DIR, 'mod_position')

HELA_MODEL_DIR = os.path.join(HELA_BASE_DIR, 'models')
HELA_CSV_DIR = os.path.join(HELA_BASE_DIR, 'csv')




SUIM_BASE_DIR = config['SUIM']['BASE_DIR']

SUIM_ORG_DATA_DIR = os.path.join(SUIM_BASE_DIR, 'original_data')
SUIM_ORG_TRAIN_FULL_IMAGES_DIR = os.path.join(SUIM_ORG_DATA_DIR, 'train_full', 'images')
SUIM_ORG_TRAIN_FULL_MASKS_DIR = os.path.join(SUIM_ORG_DATA_DIR, 'train_full', 'masks')
SUIM_ORG_TRAIN_LABELED_IMAGES_DIR = os.path.join(SUIM_ORG_DATA_DIR, 'train_labeled', 'images')
SUIM_ORG_TRAIN_LABELED_MASKS_DIR = os.path.join(SUIM_ORG_DATA_DIR, 'train_labeled', 'masks')
SUIM_ORG_TRAIN_UNLABELED_IMAGES_DIR = os.path.join(SUIM_ORG_DATA_DIR, 'train_unlabeled', 'images')
SUIM_ORG_TRAIN_UNLABELED_MASKS_DIR = os.path.join(SUIM_ORG_DATA_DIR, 'train_unlabeled', 'masks')
SUIM_ORG_VAL_IMAGES_DIR = os.path.join(SUIM_ORG_DATA_DIR, 'val', 'images')
SUIM_ORG_VAL_MASKS_DIR = os.path.join(SUIM_ORG_DATA_DIR, 'val', 'masks')

SUIM_ORG_TRAIN_VAL_IMAGES_DIR = os.path.join(SUIM_ORG_DATA_DIR, 'train_val', 'images')
SUIM_ORG_TRAIN_VAL_MASKS_BMP_DIR = os.path.join(SUIM_ORG_DATA_DIR, 'train_val', 'masks')
SUIM_ORG_TRAIN_VAL_MASKS_PNG_DIR = os.path.join(SUIM_ORG_DATA_DIR, 'train_val', 'masks_png')

SUIM_ORG_TEST_IMAGES_DIR = os.path.join(SUIM_ORG_DATA_DIR, 'TEST', 'images')
SUIM_ORG_TEST_MASKS_BMP_PATH = os.path.join(SUIM_ORG_DATA_DIR, 'TEST', 'masks')
SUIM_ORG_TEST_MASKS_PNG_PATH = os.path.join(SUIM_ORG_DATA_DIR, 'TEST', 'masks_png')


SUIM_TRAIN_FULL_MAIN_DIR = os.path.join(SUIM_BASE_DIR, 'train_full')
SUIM_TRAIN_FULL_IMAGES_DIR = os.path.join(SUIM_TRAIN_FULL_MAIN_DIR, 'images')
SUIM_TRAIN_FULL_MASKS_DIR = os.path.join(SUIM_TRAIN_FULL_MAIN_DIR, 'masks')

SUIM_TRAIN_LABELED_MAIN_DIR = os.path.join(SUIM_BASE_DIR, 'train_labeled')
SUIM_TRAIN_LABELED_IMAGES_DIR = os.path.join(SUIM_TRAIN_LABELED_MAIN_DIR, 'images')
SUIM_TRAIN_LABELED_MASKS_DIR = os.path.join(SUIM_TRAIN_LABELED_MAIN_DIR, 'masks')    

SUIM_TRAIN_LABELED_AUG_MAIN_DIR = os.path.join(SUIM_BASE_DIR, 'train_labeled_aug')
SUIM_TRAIN_LABELED_AUG_IMAGES_DIR = os.path.join(SUIM_TRAIN_LABELED_AUG_MAIN_DIR, 'images')
SUIM_TRAIN_LABELED_AUG_MASKS_DIR = os.path.join(SUIM_TRAIN_LABELED_AUG_MAIN_DIR, 'masks')

SUIM_VAL_MAIN_DIR = os.path.join(SUIM_BASE_DIR, 'val')
SUIM_VAL_IMAGES_DIR = os.path.join(SUIM_VAL_MAIN_DIR, 'images')
SUIM_VAL_MASKS_DIR = os.path.join(SUIM_VAL_MAIN_DIR, 'masks')

SUIM_TEST_MAIN_DIR = os.path.join(SUIM_BASE_DIR, 'test')
SUIM_TEST_IMAGES_DIR = os.path.join(SUIM_TEST_MAIN_DIR, 'images')
SUIM_TEST_MASKS_DIR = os.path.join(SUIM_TEST_MAIN_DIR, 'masks')

SUIM_TRAIN_UNLABELED_MAIN_DIR = os.path.join(SUIM_BASE_DIR, 'train_unlabeled')
SUIM_TRAIN_UNLABELED_IMAGES_DIR = os.path.join(SUIM_TRAIN_UNLABELED_MAIN_DIR, 'images')
SUIM_TRAIN_UNLABELED_MASKS_DIR = os.path.join(SUIM_TRAIN_UNLABELED_MAIN_DIR, 'masks')

SUIM_MODEL_DIR = os.path.join(SUIM_BASE_DIR, 'models')
SUIM_CSV_DIR = os.path.join(SUIM_BASE_DIR, 'csv')




CITYSCAPES_BASE_DIR = config['CITYSCAPES']['BASE_DIR']

CITYSCAPES_ORG_DATA_DIR = os.path.join(CITYSCAPES_BASE_DIR, 'original_data')
CITYSCAPES_ORG_TRAIN_IMAGES_DIR = os.path.join(CITYSCAPES_ORG_DATA_DIR, 'leftImg8bit', 'train')
CITYSCAPES_ORG_TRAIN_MASKS_DIR = os.path.join(CITYSCAPES_ORG_DATA_DIR, 'gtFine', 'train')
CITYSCAPES_ORG_VAL_IMAGES_DIR = os.path.join(CITYSCAPES_ORG_DATA_DIR, 'leftImg8bit', 'val')
CITYSCAPES_ORG_VAL_MASKS_DIR = os.path.join(CITYSCAPES_ORG_DATA_DIR, 'gtFine', 'val')
CITYSCAPES_ORG_VAL_TEST_IMAGES_DIR = os.path.join(CITYSCAPES_ORG_DATA_DIR, 'val_test', 'images')
CITYSCAPES_ORG_VAL_TEST_MASKS_DIR = os.path.join(CITYSCAPES_ORG_DATA_DIR, 'val_test', 'masks')


CITYSCAPES_TRAIN_FULL_MAIN_DIR = os.path.join(CITYSCAPES_BASE_DIR, 'train_full')
CITYSCAPES_TRAIN_FULL_IMAGES_DIR = os.path.join(CITYSCAPES_TRAIN_FULL_MAIN_DIR, 'images')
CITYSCAPES_TRAIN_FULL_MASKS_DIR = os.path.join(CITYSCAPES_TRAIN_FULL_MAIN_DIR, 'masks')

CITYSCAPES_TRAIN_LABELED_MAIN_DIR = os.path.join(CITYSCAPES_BASE_DIR, 'train_labeled')
CITYSCAPES_TRAIN_LABELED_IMAGES_DIR = os.path.join(CITYSCAPES_TRAIN_LABELED_MAIN_DIR, 'images')
CITYSCAPES_TRAIN_LABELED_MASKS_DIR = os.path.join(CITYSCAPES_TRAIN_LABELED_MAIN_DIR, 'masks')

CITYSCAPES_TRAIN_LABELED_AUG_MAIN_DIR = os.path.join(CITYSCAPES_BASE_DIR, 'train_labeled_aug')
CITYSCAPES_TRAIN_LABELED_AUG_IMAGES_DIR = os.path.join(CITYSCAPES_TRAIN_LABELED_AUG_MAIN_DIR, 'images')
CITYSCAPES_TRAIN_LABELED_AUG_MASKS_DIR = os.path.join(CITYSCAPES_TRAIN_LABELED_AUG_MAIN_DIR, 'masks')

CITYSCAPES_TRAIN_UNLABELED_MAIN_DIR = os.path.join(CITYSCAPES_BASE_DIR, 'train_unlabeled')
CITYSCAPES_TRAIN_UNLABELED_IMAGES_DIR = os.path.join(CITYSCAPES_TRAIN_UNLABELED_MAIN_DIR, 'images')
CITYSCAPES_TRAIN_UNLABELED_MASKS_DIR = os.path.join(CITYSCAPES_TRAIN_UNLABELED_MAIN_DIR, 'masks')

CITYSCAPES_VAL_MAIN_DIR = os.path.join(CITYSCAPES_BASE_DIR, 'val')
CITYSCAPES_VAL_IMAGES_DIR = os.path.join(CITYSCAPES_VAL_MAIN_DIR, 'images')
CITYSCAPES_VAL_MASKS_DIR = os.path.join(CITYSCAPES_VAL_MAIN_DIR, 'masks')

CITYSCAPES_TEST_MAIN_DIR = os.path.join(CITYSCAPES_BASE_DIR, 'test')
CITYSCAPES_TEST_IMAGES_DIR = os.path.join(CITYSCAPES_TEST_MAIN_DIR, 'images')
CITYSCAPES_TEST_MASKS_DIR = os.path.join(CITYSCAPES_TEST_MAIN_DIR, 'masks')

CITYSCAPES_MODEL_DIR = os.path.join(CITYSCAPES_BASE_DIR, 'models')
CITYSCAPES_CSV_DIR = os.path.join(CITYSCAPES_BASE_DIR, 'csv')
