
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SUIM_class_mapping import COLOR_TO_CLASS_MAPPING_SUIM
from functions import train_multiclass
from unet import get_unet
import tensorflow as tf
from tensorflow.keras import mixed_precision
import gc
import csv
import paths
import configparser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
mixed_precision.set_global_policy('mixed_float16')

config = configparser.ConfigParser()
config.read(os.path.join('InconsistencyMasks', 'config.ini'))

IMAGE_WIDTH = int(config['SUIM']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['SUIM']['IMAGE_HEIGHT'])
IMAGE_CHANNELS = int(config['SUIM']['IMAGE_CHANNELS'])
NUM_CLASSES = int(config['SUIM']['NUM_CLASSES'])
ALPHA =  float(config['SUIM']['ALPHA'])
ACTIFU = str(config['SUIM']['ACTIFU'])
ACTIFU_OUTPUT = str(config['SUIM']['ACTIFU_OUTPUT'])

BATCH_SIZE = int(config['DEFAULT']['BATCH_SIZE'])
TOP_Ks = int(config['DEFAULT']['TOP_Ks'])

approach = 'full_dataset'

num_imgs = len(os.listdir(paths.SUIM_TRAIN_FULL_IMAGES_DIR))
STEPS_PER_EPOCH = num_imgs // BATCH_SIZE



with tf.device('/gpu:0'):

    for runid in range(1,4):

        modelname = f'SUIM_{approach}_{runid}'
        modelname_benchmarks = []

        for i in range(0,10):

            modelname_i = f'{modelname}_{i}'
            filepath_h5 = os.path.join(paths.SUIM_MODEL_DIR, modelname_i + '.h5')
            val_pred_dir = os.path.join(paths.SUIM_BASE_DIR, 'val_predictions', approach, modelname_i)
            test_pred_dir = os.path.join(paths.SUIM_BASE_DIR, 'test_predictions', approach, modelname_i)
            train_unlabeled_pred_dir = os.path.join(paths.SUIM_BASE_DIR, 'train_unlabeled_predictions', approach, modelname_i)

            model = get_unet(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, ALPHA, ACTIFU, ACTIFU_OUTPUT)  

            mPA_val, mPA_test, mPA_train_unlabeled, mIoU_val, mIoU_test, mIoU_train_unlabeled = train_multiclass(paths.SUIM_TRAIN_FULL_IMAGES_DIR, 
                                                                                                                                    paths.SUIM_VAL_IMAGES_DIR, 
                                                                                                                                    paths.SUIM_VAL_MASKS_DIR,
                                                                                                                                    paths.SUIM_TEST_IMAGES_DIR,
                                                                                                                                    paths.SUIM_TEST_MASKS_DIR,
                                                                                                                                    paths.SUIM_TRAIN_UNLABELED_IMAGES_DIR,
                                                                                                                                    paths.SUIM_TRAIN_UNLABELED_MASKS_DIR,
                                                                                                                                    modelname_i, 
                                                                                                                                    filepath_h5, 
                                                                                                                                    model,
                                                                                                                                    tf.keras.losses.CategoricalCrossentropy(),
                                                                                                                                    STEPS_PER_EPOCH,
                                                                                                                                    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,
                                                                                                                                    NUM_CLASSES,
                                                                                                                                    COLOR_TO_CLASS_MAPPING_SUIM,
                                                                                                                                    val_pred_dir, 
                                                                                                                                    test_pred_dir, 
                                                                                                                                    train_unlabeled_pred_dir)

            modelname_benchmarks.append((modelname_i, mPA_val, mPA_test, mPA_train_unlabeled, mIoU_val, mIoU_test, mIoU_train_unlabeled))

            del model
            tf.keras.backend.clear_session()
            gc.collect()


        sorted_mIoU_val = sorted(modelname_benchmarks, key=lambda x: x[4], reverse=True)
        
        top_K_mIoUs = sorted_mIoU_val[:TOP_Ks]
        
        print(top_K_mIoUs)
        
        
        for i, top_k in enumerate(top_K_mIoUs, start=1):
            old_filename = os.path.join(paths.SUIM_MODEL_DIR, f'{top_k[0]}.h5')
            new_filename = os.path.join(paths.SUIM_MODEL_DIR, f'{top_k[0][:-2]}_topK_{i}.h5')
            
            os.rename(old_filename, new_filename)
        
        
        Header = ['modelname', 'mPA_val', 'mPA_test', 'mPA_train_unlabeled', 'mIoU_val', 'mIoU_test', 'mIoU_train_unlabeled']
        
        os.makedirs(paths.SUIM_CSV_DIR, exist_ok=True)
        
        with open(os.path.join(paths.SUIM_CSV_DIR, f'results_{modelname}.csv'), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(Header)
            for row in modelname_benchmarks:
                writer.writerow(row)
    
    
