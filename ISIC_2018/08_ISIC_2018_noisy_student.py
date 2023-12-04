import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import train_ISIC_2018, create_pseudo_labels_noisy_student_ISIC_2018, dice_loss
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

config = configparser.ConfigParser()
config.read(os.path.join('IM', 'config.ini'))

IMAGE_WIDTH = int(config['ISIC_2018']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['ISIC_2018']['IMAGE_HEIGHT'])
IMAGE_CHANNELS = int(config['ISIC_2018']['IMAGE_CHANNELS'])
NUM_CLASSES = int(config['ISIC_2018']['NUM_CLASSES'])
ALPHA =  float(config['ISIC_2018']['ALPHA'])
ACTIFU = str(config['ISIC_2018']['ACTIFU'])
ACTIFU_OUTPUT = str(config['ISIC_2018']['ACTIFU_OUTPUT'])

BATCH_SIZE = int(config['DEFAULT']['BATCH_SIZE'])
TOP_Ks = int(config['DEFAULT']['TOP_Ks'])

approach = 'noisy_student'


alphas = [0.5, 0.75, 1, 1.25, 1.5]
max_blurs = [0,1,1,2,3]
max_noises = [5, 10, 15, 20, 25]
brightness_range_alphas = [(0.9, 1.1), (0.8, 1.2), (0.7, 1.3), (0.6, 1.4), (0.5, 1.5)]  
brightness_range_betas = [(-5, 5), (-10, 10), (-15, 15), (-20, 20), (-25, 25)]
FREE_ROTATION = config['ISIC_2018']['FREE_ROTATION'].lower() == 'true'



with tf.device('/gpu:0'):

    for runid in range(1,4):

        for gen in range(0,5):

            modelname_benchmarks = []
            
            modelname = f'ISIC_2018_{approach}_{runid}_gen{gen}'

            train_unlabeled_pseudo_label_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)
            train_unlabeled_pseudo_label_dir_images = os.path.join(train_unlabeled_pseudo_label_dir, 'images')
            train_unlabeled_pseudo_label_dir_masks = os.path.join(train_unlabeled_pseudo_label_dir, 'masks')


            if gen == 0:
                best_model_filepath_h5 = os.path.join(paths.ISIC_2018_MODEL_DIR, f'ISIC_2018_subset_{runid}_topK_1.h5')
            else:
                best_model_filepath_h5 = os.path.join(paths.ISIC_2018_MODEL_DIR, f'ISIC_2018_{approach}_{runid}_gen{gen-1}_topK_1.h5')

            best_model = tf.keras.models.load_model(best_model_filepath_h5, custom_objects={'dice_loss': dice_loss})

            blur = max_blurs[gen]
            noise = max_noises[gen]
            brightness_range_alpha = brightness_range_alphas[gen]
            brightness_range_beta = brightness_range_betas[gen]

            create_pseudo_labels_noisy_student_ISIC_2018(best_model, 
                                                         IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, 
                                                         paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR, 
                                                         train_unlabeled_pseudo_label_dir, 
                                                         True,  # convert to rgb
                                                         brightness_range_alpha, brightness_range_beta, blur, noise, FREE_ROTATION)

           
            for imagename in tqdm(os.listdir(paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR)):
                shutil.copy(os.path.join(paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_images, imagename))
                shutil.copy(os.path.join(paths.ISIC_2018_TRAIN_LABELED_MASKS_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_masks, imagename))

            num_imgs = len(os.listdir(train_unlabeled_pseudo_label_dir_images))
            STEPS_PER_EPOCH = num_imgs // BATCH_SIZE
        
            for i in range(0,5):
        
                modelname_i = f'{modelname}_{i}'
                filepath_h5 = os.path.join(paths.ISIC_2018_MODEL_DIR, modelname_i + '.h5')
                val_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'val_predictions', approach, modelname_i)
                test_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'test_predictions', approach, modelname_i)
                train_unlabeled_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'train_unlabeled_predictions', approach, modelname_i)
        
                model = get_unet(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, alphas[gen], ACTIFU, ACTIFU_OUTPUT)  
        
                mIoU_val, mIoU_test, mIoU_train_unlabeled, dice_score_val, dice_score_test, dice_score_train_unlabeled = train_ISIC_2018(train_unlabeled_pseudo_label_dir_images, 
                                                                                                                                        paths.ISIC_2018_VAL_IMAGES_DIR, 
                                                                                                                                        paths.ISIC_2018_VAL_MASKS_DIR,
                                                                                                                                        paths.ISIC_2018_TEST_IMAGES_DIR,
                                                                                                                                        paths.ISIC_2018_TEST_MASKS_DIR,
                                                                                                                                        paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR,
                                                                                                                                        paths.ISIC_2018_TRAIN_UNLABELED_MASKS_DIR,
                                                                                                                                        modelname_i, 
                                                                                                                                        filepath_h5, 
                                                                                                                                        model,
                                                                                                                                        'mse',  # loss function
                                                                                                                                        STEPS_PER_EPOCH,
                                                                                                                                        IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,
                                                                                                                                        val_pred_dir, 
                                                                                                                                        test_pred_dir, 
                                                                                                                                        train_unlabeled_pred_dir)
        
                modelname_benchmarks.append((modelname_i, mIoU_val, mIoU_test, mIoU_train_unlabeled, dice_score_val, dice_score_test, dice_score_train_unlabeled))
        
                del model
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
