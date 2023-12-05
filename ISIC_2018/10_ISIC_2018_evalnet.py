import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import train_ISIC_2018, create_training_data_evalnet_ISIC_2018, train_evalnet_ISIC_2018, create_training_data_for_segnet_ISIC_2018, dice_loss
from unet import get_unet
from evalnet import get_evalnet
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
config.read(os.path.join('InconsistencyMasks', 'config.ini'))

IMAGE_WIDTH = int(config['ISIC_2018']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['ISIC_2018']['IMAGE_HEIGHT'])
IMAGE_CHANNELS = int(config['ISIC_2018']['IMAGE_CHANNELS'])
NUM_CLASSES = int(config['ISIC_2018']['NUM_CLASSES'])
ALPHA =  float(config['ISIC_2018']['ALPHA'])
ALPHA_EVALNET =  float(config['ISIC_2018']['ALPHA_EVALNET'])
ACTIFU = str(config['ISIC_2018']['ACTIFU'])
ACTIFU_OUTPUT = str(config['ISIC_2018']['ACTIFU_OUTPUT'])
THRESHOLD = float(config['ISIC_2018']['MAX_THRESHOLD'])

BATCH_SIZE = int(config['DEFAULT']['BATCH_SIZE'])
BATCH_SIZE_EVALNET = int(config['DEFAULT']['BATCH_SIZE_EVALNET'])
NUM_EPOCHS_EVALNET = int(config['DEFAULT']['NUM_EPOCHS_EVALNET'])
TOP_Ks = int(config['DEFAULT']['TOP_Ks'])



train_new_evalnet = True


with tf.device('/gpu:0'):

    for runid in range(1,4):

        if train_new_evalnet == True:

            main_output_evalnet_dir_train = os.path.join(paths.ISIC_2018_BASE_DIR, 'evalnet', f'run_{runid}', 'train')
            main_output_evalnet_dir_val = os.path.join(paths.ISIC_2018_BASE_DIR, 'evalnet', f'run_{runid}', 'val')

            model_i = 0
            for modelname in os.listdir(paths.ISIC_2018_MODEL_DIR):
            
                if f'ISIC_2018_subset_{runid}' in modelname:        
            
                    model_filepath_h5 = os.path.join(paths.ISIC_2018_MODEL_DIR, modelname)
            
                    model = tf.keras.models.load_model(model_filepath_h5, custom_objects={'dice_loss': dice_loss})
                    create_training_data_evalnet_ISIC_2018(model, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR, paths.ISIC_2018_TRAIN_LABELED_MASKS_DIR, main_output_evalnet_dir_train, model_i)
                    create_training_data_evalnet_ISIC_2018(model, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, paths.ISIC_2018_VAL_IMAGES_DIR, paths.ISIC_2018_VAL_MASKS_DIR, main_output_evalnet_dir_val, model_i)
                    model_i += 1
             
                    del model
                    tf.keras.backend.clear_session()
                    gc.collect()
            
            model_i = 10
            for modelname in os.listdir(paths.ISIC_2018_MODEL_DIR):
            
                if f'ISIC_2018_subset_aug_{runid}' in modelname:        
            
                    model_filepath_h5 = os.path.join(paths.ISIC_2018_MODEL_DIR, modelname)
            
                    model = tf.keras.models.load_model(model_filepath_h5, custom_objects={'dice_loss': dice_loss})
                    create_training_data_evalnet_ISIC_2018(model, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, paths.ISIC_2018_TRAIN_LABELED_AUG_IMAGES_DIR, paths.ISIC_2018_TRAIN_LABELED_AUG_MASKS_DIR, main_output_evalnet_dir_train, model_i)
                    create_training_data_evalnet_ISIC_2018(model, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, paths.ISIC_2018_VAL_IMAGES_DIR, paths.ISIC_2018_VAL_MASKS_DIR, main_output_evalnet_dir_val, model_i)
                    model_i += 1
             
                    del model
                    tf.keras.backend.clear_session()
                    gc.collect()
            
            modelname_evalnet = f'ISIC_2018_evalnet_{runid}'
            model_filepath_h5 = os.path.join(paths.ISIC_2018_MODEL_DIR , f'{modelname_evalnet}.h5')
            
            evalnet_model = get_evalnet(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, ALPHA_EVALNET, normalize_B=True)
            evalnet_model.summary()
            mse, mae = train_evalnet_ISIC_2018(evalnet_model, main_output_evalnet_dir_train, main_output_evalnet_dir_val, model_filepath_h5, BATCH_SIZE_EVALNET, NUM_EPOCHS_EVALNET)  


            Header = ['modelname', 'mse', 'mae']
            results = [modelname_evalnet, mse, mae]
            
            os.makedirs(paths.ISIC_2018_CSV_DIR, exist_ok=True)
            
            with open(os.path.join(paths.ISIC_2018_CSV_DIR, f'results_{modelname_evalnet}.csv'), 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(Header)
                writer.writerow(results)



        for gen in range(0,5):

            modelname_benchmarks = []
            best_model_filepaths_h5 = []
            best_models = []
            
            modelname = f'ISIC_2018_segnet_{runid}_gen{gen}'
            modelname_last_gen = f'ISIC_2018_segnet_{runid}_gen{gen-1}'

            val_n_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'val_predictions', 'segnet', modelname)
            test_n_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'test_predictions', 'segnet', modelname)
            train_unlabeled_pseudo_label_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'train_unlabeled_predictions', 'segnet', modelname)
            train_unlabeled_pseudo_label_dir_last_gen = os.path.join(paths.ISIC_2018_BASE_DIR, 'train_unlabeled_predictions', 'segnet', modelname_last_gen)
            train_unlabeled_pseudo_label_dir_images = os.path.join(train_unlabeled_pseudo_label_dir, 'images')
            train_unlabeled_pseudo_label_dir_masks = os.path.join(train_unlabeled_pseudo_label_dir, 'masks')


            evalnet_model_filepath_h5 = os.path.join(paths.ISIC_2018_MODEL_DIR, f'ISIC_2018_evalnet_{runid}.h5')
            evalnet_model = tf.keras.models.load_model(evalnet_model_filepath_h5)

            if gen == 0:
                mask_dirs = []

                for j in range(0,10):
                    mask_dirs.append(os.path.join(paths.ISIC_2018_BASE_DIR, 'train_unlabeled_predictions', 'subset', f'ISIC_2018_subset_{runid}_{j}'))

                create_training_data_for_segnet_ISIC_2018(evalnet_model, 
                                                          IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, 
                                                          paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR, 
                                                          mask_dirs, 
                                                          train_unlabeled_pseudo_label_dir, 
                                                          THRESHOLD)  
                
            else:
                mask_dirs = []

                for j in range(0,5):
                    mask_dirs.append(os.path.join(paths.ISIC_2018_BASE_DIR, 'train_unlabeled_predictions', 'segnet', f'ISIC_2018_segnet_{runid}_gen{gen-1}_{j}'))

                create_training_data_for_segnet_ISIC_2018(evalnet_model, 
                                                          IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, 
                                                          paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR, 
                                                          mask_dirs, 
                                                          train_unlabeled_pseudo_label_dir, 
                                                          THRESHOLD, 
                                                          train_unlabeled_pseudo_label_dir_last_gen)  
                

           
            for imagename in tqdm(os.listdir(paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR)):
                shutil.copy(os.path.join(paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_images, imagename))
                shutil.copy(os.path.join(paths.ISIC_2018_TRAIN_LABELED_MASKS_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_masks, imagename))

            num_imgs = len(os.listdir(train_unlabeled_pseudo_label_dir_images))
            STEPS_PER_EPOCH = num_imgs // BATCH_SIZE
        
            for i in range(0,5):
        
                modelname_i = f'{modelname}_{i}'
                filepath_h5 = os.path.join(paths.ISIC_2018_MODEL_DIR, modelname_i + '.h5')
                val_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'val_predictions', 'segnet', modelname_i)
                test_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'test_predictions', 'segnet', modelname_i)
                train_unlabeled_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'train_unlabeled_predictions', 'segnet', modelname_i)
        
                model = get_unet(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, ALPHA, ACTIFU, ACTIFU_OUTPUT)  
        
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
            
            with open(os.path.join(paths.ISIC_2018_CSV_DIR, f'results_{modelname}.csv'), 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(Header)
                for row in modelname_benchmarks:
                    writer.writerow(row)
