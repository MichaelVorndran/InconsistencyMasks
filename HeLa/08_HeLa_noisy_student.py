import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import train_hela, create_pseudo_labels_noisy_student_hela
from unet import get_unet
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tqdm import tqdm
import shutil
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

IMAGE_WIDTH = int(config['HELA']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['HELA']['IMAGE_HEIGHT'])
IMAGE_CHANNELS = int(config['HELA']['IMAGE_CHANNELS'])
NUM_CLASSES = int(config['HELA']['NUM_CLASSES'])
ALPHA =  float(config['HELA']['ALPHA'])
ACTIFU = str(config['HELA']['ACTIFU'])
ACTIFU_OUTPUT = str(config['HELA']['ACTIFU_OUTPUT'])

BATCH_SIZE = int(config['DEFAULT']['BATCH_SIZE'])
TOP_Ks = int(config['DEFAULT']['TOP_Ks'])

FREE_ROTATION = config['HELA']['FREE_ROTATION'].lower() == 'true'

approach = 'noisy_student'


alphas = [1, 1.25, 1.5, 1.75, 2]
max_blurs = [0,1,1,2,3]
max_noises = [5, 10, 15, 20, 25]
brightness_range_alphas = [(0.9, 1.1), (0.9, 1.1), (0.8, 1.2), (0.8, 1.2), (0.7, 1.3)]  
brightness_range_betas = [(-3, 3), (-6, 6), (-9, 9), (-12, 12), (-15, 15)]



with tf.device('/gpu:0'):

    for runid in range(1,4):

        for gen in range(0,5):

            modelname_benchmarks = []
            
            modelname = f'HELA_{approach}_{runid}_gen{gen}'

            train_unlabeled_pseudo_label_dir = os.path.join(paths.HELA_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)
            train_unlabeled_pseudo_label_dir_bf_images = os.path.join(train_unlabeled_pseudo_label_dir, 'brightfield')
            train_unlabeled_pseudo_label_dir_alive_masks = os.path.join(train_unlabeled_pseudo_label_dir, 'alive')
            train_unlabeled_pseudo_label_dir_dead_masks = os.path.join(train_unlabeled_pseudo_label_dir, 'dead')
            train_unlabeled_pseudo_label_dir_pos_masks = os.path.join(train_unlabeled_pseudo_label_dir, 'mod_position')


            if gen == 0:
                best_model_filepath_h5 = os.path.join(paths.HELA_MODEL_DIR, f'HELA_subset_{runid}_topK_1.h5')
            else:
                best_model_filepath_h5 = os.path.join(paths.HELA_MODEL_DIR, f'HELA_{approach}_{runid}_gen{gen-1}_topK_1.h5')

            best_model = tf.keras.models.load_model(best_model_filepath_h5)

            blur = max_blurs[gen]
            noise = max_noises[gen]
            brightness_range_alpha = brightness_range_alphas[gen]
            brightness_range_beta = brightness_range_betas[gen]

            create_pseudo_labels_noisy_student_hela(best_model, 
                                                    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, 
                                                    paths.HELA_TRAIN_UNLABELED_BRIGHTFIELD_DIR, 
                                                    train_unlabeled_pseudo_label_dir, 
                                                    brightness_range_alpha, brightness_range_beta, blur, noise, FREE_ROTATION)

           
            for imagename in tqdm(os.listdir(paths.HELA_TRAIN_LABELED_BRIGHTFIELD_DIR)):
                shutil.copy(os.path.join(paths.HELA_TRAIN_LABELED_BRIGHTFIELD_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_bf_images, imagename))
                shutil.copy(os.path.join(paths.HELA_TRAIN_LABELED_ALIVE_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_alive_masks, imagename))
                shutil.copy(os.path.join(paths.HELA_TRAIN_LABELED_DEAD_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_dead_masks, imagename))
                shutil.copy(os.path.join(paths.HELA_TRAIN_LABELED_MOD_POS_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_pos_masks, imagename))
               

            num_imgs = len(os.listdir(train_unlabeled_pseudo_label_dir_bf_images))
            STEPS_PER_EPOCH = num_imgs // BATCH_SIZE
        
            for i in range(0,5):
        
                modelname_i = f'{modelname}_{i}'
                filepath_h5 = os.path.join(paths.HELA_MODEL_DIR, modelname_i + '.h5')
                val_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'val_predictions', approach, modelname_i)
                test_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'test_predictions', approach, modelname_i)
                train_unlabeled_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'train_unlabeled_predictions', approach, modelname_i)
        
                model = get_unet(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, alphas[gen], ACTIFU, ACTIFU_OUTPUT)  

                mIoU_val, mIoU_ad_val, mcce_val, mIoU_test, mIoU_ad_test, mcce_test, mIoU_unlabeled, mIoU_ad_unlabeled, mcce_unlabeled = train_hela(
                                                                                                            train_unlabeled_pseudo_label_dir_bf_images, 
                                                                                                            paths.HELA_VAL_BRIGHTFIELD_DIR,
                                                                                                            paths.HELA_VAL_DIR, 
                                                                                                            paths.HELA_TEST_DIR, 
                                                                                                            paths.HELA_TRAIN_UNLABELED_DIR, 
                                                                                                            modelname_i, 
                                                                                                            filepath_h5, 
                                                                                                            model,
                                                                                                            'mse',
                                                                                                            STEPS_PER_EPOCH,
                                                                                                            IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,
                                                                                                            val_pred_dir, 
                                                                                                            test_pred_dir, 
                                                                                                            train_unlabeled_pred_dir)

                modelname_benchmarks.append((modelname_i, mIoU_val, mIoU_ad_val, mcce_val, mIoU_test, mIoU_ad_test, mcce_test, mIoU_unlabeled, mIoU_ad_unlabeled, mcce_unlabeled))


        
                del model
                tf.keras.backend.clear_session()
                gc.collect()
        
        
            sorted_mIoU_val = sorted(modelname_benchmarks, key=lambda x: x[6], reverse=False)
            
            top_K_mIoUs = sorted_mIoU_val[:TOP_Ks]
            
            print(top_K_mIoUs)
            
            
            for i, top_k in enumerate(top_K_mIoUs, start=1):
                old_filename = os.path.join(paths.HELA_MODEL_DIR, f'{top_k[0]}.h5')
                new_filename = os.path.join(paths.HELA_MODEL_DIR, f'{top_k[0][:-2]}_topK_{i}.h5')
                
                os.rename(old_filename, new_filename)
            
            Header = ['modelname', 'mIoU_val', 'mIoU_ad_val', 'mcce_val', 'mIoU_test', 'mIoU_ad_test', 'mcce_test', 'mIoU_unlabeled', 'mIoU_ad_unlabeled', 'mcce_unlabeled']
            
            os.makedirs(paths.HELA_CSV_DIR, exist_ok=True)
            
            
            with open(os.path.join(paths.HELA_CSV_DIR, f'results_{modelname}.csv'), 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(Header)
                for row in modelname_benchmarks:
                    writer.writerow(row)

