import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SUIM_class_mapping import COLOR_TO_CLASS_MAPPING_SUIM
from functions import train_multiclass, create_training_data_evalnet_miou_multiclass, train_evalnet_miou_model_multiclass, create_training_data_by_evalnet_miou_for_segnet_multiclass
from unet import get_unet
from evalnet import get_evalnet_miou
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

IMAGE_WIDTH = int(config['SUIM']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['SUIM']['IMAGE_HEIGHT'])
IMAGE_CHANNELS = int(config['SUIM']['IMAGE_CHANNELS'])
NUM_CLASSES = int(config['SUIM']['NUM_CLASSES'])
ALPHA =  float(config['SUIM']['ALPHA'])
ALPHA_EVALNET =  float(config['SUIM']['ALPHA_EVALNET'])
ACTIFU = str(config['SUIM']['ACTIFU'])
ACTIFU_OUTPUT = str(config['SUIM']['ACTIFU_OUTPUT'])

BATCH_SIZE = int(config['DEFAULT']['BATCH_SIZE'])
BATCH_SIZE_EVALNET = int(config['DEFAULT']['BATCH_SIZE_EVALNET'])
NUM_EPOCHS_EVALNET = int(config['DEFAULT']['NUM_EPOCHS_EVALNET'])
TOP_Ks = int(config['DEFAULT']['TOP_Ks'])
THRESHOLD = float(config['DEFAULT']['THRESHOLD'])


train_new_evalnet = True

os.makedirs(paths.SUIM_CSV_DIR, exist_ok=True)


with tf.device('/gpu:0'):

    for runid in range(1,4):

        if train_new_evalnet == True:

            main_output_evalnet_dir_train = os.path.join(paths.SUIM_BASE_DIR, 'evalnet', f'run_{runid}', 'train')
            main_output_evalnet_dir_val = os.path.join(paths.SUIM_BASE_DIR, 'evalnet', f'run_{runid}', 'val')

            model_i = 0
            for modelname in os.listdir(paths.SUIM_MODEL_DIR):
            
                if f'SUIM_subset_{runid}' in modelname:        
            
                    model_filepath_h5 = os.path.join(paths.SUIM_MODEL_DIR, modelname)
            
                    model = tf.keras.models.load_model(model_filepath_h5)
                    create_training_data_evalnet_miou_multiclass(model, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, paths.SUIM_TRAIN_LABELED_IMAGES_DIR, paths.SUIM_TRAIN_LABELED_MASKS_DIR, main_output_evalnet_dir_train, model_i)
                    create_training_data_evalnet_miou_multiclass(model, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, paths.SUIM_VAL_IMAGES_DIR, paths.SUIM_VAL_MASKS_DIR, main_output_evalnet_dir_val, model_i)
                    model_i += 1
             
                    del model
                    tf.keras.backend.clear_session()
                    gc.collect()
            
            
            model_i = 10
            for modelname in os.listdir(paths.SUIM_MODEL_DIR):
            
                if f'SUIM_subset_aug_{runid}' in modelname:        
            
                    model_filepath_h5 = os.path.join(paths.SUIM_MODEL_DIR, modelname)
            
                    model = tf.keras.models.load_model(model_filepath_h5)
                    create_training_data_evalnet_miou_multiclass(model, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, paths.SUIM_TRAIN_LABELED_AUG_IMAGES_DIR, paths.SUIM_TRAIN_LABELED_AUG_MASKS_DIR, main_output_evalnet_dir_train, model_i)
                    create_training_data_evalnet_miou_multiclass(model, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, paths.SUIM_VAL_IMAGES_DIR, paths.SUIM_VAL_MASKS_DIR, main_output_evalnet_dir_val, model_i)
                    model_i += 1
             
                    del model
                    tf.keras.backend.clear_session()
                    gc.collect()
            
            
            modelname_evalnet = f'SUIM_evalnet_miou_{runid}'
            model_filepath_h5 = os.path.join(paths.SUIM_MODEL_DIR , f'{modelname_evalnet}.h5')
            
            evalnet_model = get_evalnet_miou(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, ALPHA_EVALNET)
            evalnet_model.summary()
            total_loss, iou_loss, conf_loss, iou_mae, conf_mae = train_evalnet_miou_model_multiclass(evalnet_model, IMAGE_HEIGHT, IMAGE_WIDTH, main_output_evalnet_dir_train, main_output_evalnet_dir_val, model_filepath_h5, BATCH_SIZE_EVALNET, NUM_CLASSES, NUM_EPOCHS_EVALNET)  

            Header = ['modelname', 'mse', 'mae']
            results = [total_loss, iou_loss, conf_loss, iou_mae, conf_mae]
            
            os.makedirs(paths.SUIM_CSV_DIR, exist_ok=True)
            
            with open(os.path.join(paths.SUIM_CSV_DIR, f'results_{modelname_evalnet}.csv'), 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(Header)
                writer.writerow(results)






        for gen in range(0,5):

            modelname_benchmarks = []
            best_model_filepaths_h5 = []
            best_models = []
            
            modelname = f'SUIM_segnet_miou_{runid}_gen{gen}'
            modelname_last_gen = f'SUIM_segnet_miou_{runid}_gen{gen-1}'

            val_n_pred_dir = os.path.join(paths.SUIM_BASE_DIR, 'val_predictions', 'segnet', modelname)
            test_n_pred_dir = os.path.join(paths.SUIM_BASE_DIR, 'test_predictions', 'segnet', modelname)
            train_unlabeled_pseudo_label_dir = os.path.join(paths.SUIM_BASE_DIR, 'train_unlabeled_predictions', 'segnet', modelname)
            train_unlabeled_pseudo_label_dir_last_gen = os.path.join(paths.SUIM_BASE_DIR, 'train_unlabeled_predictions', 'segnet', modelname_last_gen)
            train_unlabeled_pseudo_label_dir_images = os.path.join(train_unlabeled_pseudo_label_dir, 'images')
            train_unlabeled_pseudo_label_dir_masks = os.path.join(train_unlabeled_pseudo_label_dir, 'masks')


            evalnet_model_filepath_h5 = os.path.join(paths.SUIM_MODEL_DIR, f'SUIM_evalnet_miou_{runid}.h5')
            evalnet_model = tf.keras.models.load_model(evalnet_model_filepath_h5)

            if gen == 0:
                mask_dirs = []

                for j in range(0,10):
                    mask_dirs.append(os.path.join(paths.SUIM_BASE_DIR, 'train_unlabeled_predictions', 'subset', f'SUIM_subset_{runid}_{j}'))

                create_training_data_by_evalnet_miou_for_segnet_multiclass(evalnet_model, 
                                                                           IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, 
                                                                           NUM_CLASSES,
                                                                           paths.SUIM_TRAIN_UNLABELED_IMAGES_DIR, 
                                                                           mask_dirs,
                                                                           train_unlabeled_pseudo_label_dir, 
                                                                           THRESHOLD)  
                
            else:
                mask_dirs = []

                for j in range(0,5):
                    mask_dirs.append(os.path.join(paths.SUIM_BASE_DIR, 'train_unlabeled_predictions', 'segnet', f'SUIM_segnet_miou_{runid}_gen{gen-1}_{j}'))

                create_training_data_by_evalnet_miou_for_segnet_multiclass(evalnet_model, 
                                                                           IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, 
                                                                           NUM_CLASSES,
                                                                           paths.SUIM_TRAIN_UNLABELED_IMAGES_DIR, 
                                                                           mask_dirs, 
                                                                           train_unlabeled_pseudo_label_dir, 
                                                                           THRESHOLD, 
                                                                           train_unlabeled_pseudo_label_dir_last_gen)  
                

           
            for imagename in tqdm(os.listdir(paths.SUIM_TRAIN_LABELED_IMAGES_DIR)):
                shutil.copy(os.path.join(paths.SUIM_TRAIN_LABELED_IMAGES_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_images, imagename))
                shutil.copy(os.path.join(paths.SUIM_TRAIN_LABELED_MASKS_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_masks, imagename))

            num_imgs = len(os.listdir(train_unlabeled_pseudo_label_dir_images))
            STEPS_PER_EPOCH = num_imgs // BATCH_SIZE
        
            for i in range(0,5):
        
                modelname_i = f'{modelname}_{i}'
                filepath_h5 = os.path.join(paths.SUIM_MODEL_DIR, modelname_i + '.h5')
                val_pred_dir = os.path.join(paths.SUIM_BASE_DIR, 'val_predictions', 'segnet', modelname_i)
                test_pred_dir = os.path.join(paths.SUIM_BASE_DIR, 'test_predictions', 'segnet', modelname_i)
                train_unlabeled_pred_dir = os.path.join(paths.SUIM_BASE_DIR, 'train_unlabeled_predictions', 'segnet', modelname_i)
        
                model = get_unet(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, ALPHA, ACTIFU, ACTIFU_OUTPUT)  

                mPA_val, mPA_test, mPA_train_unlabeled, mIoU_val, mIoU_test, mIoU_train_unlabeled = train_multiclass(train_unlabeled_pseudo_label_dir_images, 
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


       


