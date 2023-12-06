import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import train_ISIC_2018_consistency_loss, dice_loss
from unet import get_unet
import tensorflow as tf
from tensorflow.keras import mixed_precision
import gc
import csv
import paths
import configparser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
mixed_precision.set_global_policy('mixed_float16')

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
config_dir = os.path.join(parent_dir, 'config.ini')

config = configparser.ConfigParser()
config.read(config_dir)

IMAGE_WIDTH = int(config['ISIC_2018']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['ISIC_2018']['IMAGE_HEIGHT'])
IMAGE_CHANNELS = int(config['ISIC_2018']['IMAGE_CHANNELS'])
NUM_CLASSES = int(config['ISIC_2018']['NUM_CLASSES'])
ALPHA =  float(config['ISIC_2018']['ALPHA'])
ACTIFU = str(config['ISIC_2018']['ACTIFU'])
ACTIFU_OUTPUT = str(config['ISIC_2018']['ACTIFU_OUTPUT'])

BATCH_SIZE = int(config['DEFAULT']['BATCH_SIZE'])
TOP_Ks = int(config['DEFAULT']['TOP_Ks'])

approach = 'consistency_loss'

num_imgs = len(os.listdir(paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR))
STEPS_PER_EPOCH = num_imgs // BATCH_SIZE


aug_strenghts = ['low', 'mid', 'high'] 
max_blurs = [1,2,3]
max_noises = [10, 17, 25]
brightness_range_alphas = [(0.85, 1.15), (0.7, 1.3), (0.5, 1.5)]  
brightness_range_betas = [(-10, 10), (-17, 17), (-25, 25)]



with tf.device('/gpu:0'):

    for runid in range(1,4):

        for index, augs in enumerate(aug_strenghts):

            blur = max_blurs[index]
            noise = max_noises[index]
            brightness_range_alpha = brightness_range_alphas[index]
            brightness_range_beta = brightness_range_betas[index]

            modelname = f'ISIC_2018_{approach}_{runid}_aug_{augs}'
            modelname_benchmarks = []

            for i in range(0,5):

                modelname_i = f'{modelname}_{i}'
                filepath_h5 = os.path.join(paths.ISIC_2018_MODEL_DIR, modelname_i + '.h5')
                val_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'val_predictions', approach, modelname_i)
                test_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'test_predictions', approach, modelname_i)
                train_unlabeled_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'train_unlabeled_predictions', approach, modelname_i)

                #model = get_unet(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, ALPHA, ACTIFU, ACTIFU_OUTPUT)  

                best_pretrained_filepath_h5 = os.path.join(paths.ISIC_2018_MODEL_DIR, f'ISIC_2018_subset_{runid}_topK_1.h5')
                best_pretrained_model = tf.keras.models.load_model(best_pretrained_filepath_h5, custom_objects={'dice_loss': dice_loss})

                mIoU_val, mIoU_test, mIoU_train_unlabeled, dice_score_val, dice_score_test, dice_score_train_unlabeled = train_ISIC_2018_consistency_loss(paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR, 
                                                                                                                                        paths.ISIC_2018_TRAIN_LABELED_MASKS_DIR,
                                                                                                                                        paths.ISIC_2018_VAL_IMAGES_DIR, 
                                                                                                                                        paths.ISIC_2018_VAL_MASKS_DIR,
                                                                                                                                        paths.ISIC_2018_TEST_IMAGES_DIR,
                                                                                                                                        paths.ISIC_2018_TEST_MASKS_DIR,
                                                                                                                                        paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR,
                                                                                                                                        paths.ISIC_2018_TRAIN_UNLABELED_MASKS_DIR,
                                                                                                                                        modelname_i, 
                                                                                                                                        filepath_h5, 
                                                                                                                                        best_pretrained_model,
                                                                                                                                        'mse',  # loss function
                                                                                                                                        IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,
                                                                                                                                        val_pred_dir, 
                                                                                                                                        test_pred_dir, 
                                                                                                                                        train_unlabeled_pred_dir,
                                                                                                                                        blur, noise, brightness_range_alpha, brightness_range_beta)

                modelname_benchmarks.append((modelname_i, mIoU_val, mIoU_test, mIoU_train_unlabeled, dice_score_val, dice_score_test, dice_score_train_unlabeled))

                del best_pretrained_model
                tf.keras.backend.clear_session()
                gc.collect()


            sorted_mIoU_val = sorted(modelname_benchmarks, key=lambda x: x[1], reverse=True)
            
            top_K_mIoUs = sorted_mIoU_val[:TOP_Ks]
            
            print(top_K_mIoUs)
            
            
            for i, top_k in enumerate(top_K_mIoUs, start=1):
                old_filename = os.path.join(paths.ISIC_2018_MODEL_DIR, f'{top_k[0]}.h5')
                new_filename = os.path.join(paths.ISIC_2018_MODEL_DIR, f'{top_k[0][:-2]}_topK_{i}.h5')
                
                os.rename(old_filename, new_filename)
            
            
            Header = ['modelname', 'mIoU_val', 'mIoU_test', 'mIoU_train_unlabeled', 'dice_score_val', 'dice_score_test', 'dice_score_train_unlabeled']
            
            os.makedirs(paths.ISIC_2018_CSV_DIR, exist_ok=True)
            
            with open(os.path.join(paths.ISIC_2018_CSV_DIR, f'results_{modelname}.csv'), 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(Header)
                for row in modelname_benchmarks:
                    writer.writerow(row)
    
    
