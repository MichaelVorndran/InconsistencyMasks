import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Cityscapes_class_mapping import COLOR_TO_CLASS_MAPPING_CITYSCAPES
from functions import train_multiclass, create_pseudo_labels_im_multiclass, create_augment_images_and_masks_with_evalnet_ensemble_multiclass, train_evalnet_miou_model_multiclass, create_training_data_evalnet_miou_im_multiclass
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

IMAGE_WIDTH = int(config['CITYSCAPES']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['CITYSCAPES']['IMAGE_HEIGHT'])
IMAGE_CHANNELS = int(config['CITYSCAPES']['IMAGE_CHANNELS'])
NUM_CLASSES = int(config['CITYSCAPES']['NUM_CLASSES'])
ALPHA_EVALNET =  float(config['CITYSCAPES']['ALPHA_EVALNET'])
ALPHA =  float(config['CITYSCAPES']['ALPHA'])
ACTIFU = str(config['CITYSCAPES']['ACTIFU'])
ACTIFU_OUTPUT = str(config['CITYSCAPES']['ACTIFU_OUTPUT'])

BATCH_SIZE_EVALNET = int(config['DEFAULT']['BATCH_SIZE_EVALNET'])
NUM_EPOCHS_EVALNET = int(config['DEFAULT']['NUM_EPOCHS_EVALNET'])
NUM_LOOPS_TRAIN = int(config['DEFAULT']['NUM_LOOPS_TRAIN'])
NUM_LOOPS_VAL = int(config['DEFAULT']['NUM_LOOPS_VAL'])
BATCH_SIZE = int(config['DEFAULT']['BATCH_SIZE'])
TOP_Ks = int(config['DEFAULT']['TOP_Ks'])

EK = int(config['CITYSCAPES']['ERODE_KERNEL'])
DK = int(config['CITYSCAPES']['DILATE_KERNEL'])
BI = config['CITYSCAPES']['BLOCK_INPUT'].lower() == 'true'
BO = config['CITYSCAPES']['BLOCK_OUTPUT'].lower() == 'true'
MIN_THRESHOLD = float(config['CITYSCAPES']['MIN_THRESHOLD'])
MAX_THRESHOLD = float(config['CITYSCAPES']['MAX_THRESHOLD'])
FREE_ROTATION = config['CITYSCAPES']['FREE_ROTATION'].lower() == 'true'


approach = 'aug_IM_plus_plus'

alphas = [1, 1.25, 1.5, 1.75, 2]
max_blurs = [0,0,0,0,1]
max_noises = [3, 6, 9, 12, 15]
brightness_range_alphas = [(0.95, 1.05), (0.9, 1.1), (0.8, 1.2), (0.7, 1.3), (0.6, 1.4)]  
brightness_range_betas = [(-3, 3), (-6, 6), (-9, 9), (-12, 12), (-15, 15)]

train_new_evalnet = True


with tf.device('/gpu:0'):

    for runid in range(1,4):

        if train_new_evalnet == True:
        
            main_output_evalnet_im_dir_train = os.path.join(paths.CITYSCAPES_BASE_DIR, 'evalnet_aug_im', f'run_{runid}', 'train')
            main_output_evalnet_im_dir_val = os.path.join(paths.CITYSCAPES_BASE_DIR, 'evalnet_aug_im', f'run_{runid}', 'val')
        
            subset_models = []

            for modelname in os.listdir(paths.CITYSCAPES_MODEL_DIR):
            
                if f'CITYSCAPES_subset_aug_{runid}' in modelname:        
            
                    model_filepath_h5 = os.path.join(paths.CITYSCAPES_MODEL_DIR, modelname)
            
                    model = tf.keras.models.load_model(model_filepath_h5)
                    subset_models.append(model)
            
            
            create_training_data_evalnet_miou_im_multiclass(subset_models,
                                                         IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, 
                                                         NUM_CLASSES,
                                                         paths.CITYSCAPES_TRAIN_LABELED_IMAGES_DIR, 
                                                         paths.CITYSCAPES_TRAIN_LABELED_MASKS_DIR, 
                                                         main_output_evalnet_im_dir_train,
                                                         NUM_LOOPS_TRAIN)
            
            create_training_data_evalnet_miou_im_multiclass(subset_models,
                                                         IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, 
                                                         NUM_CLASSES,
                                                         paths.CITYSCAPES_VAL_IMAGES_DIR, 
                                                         paths.CITYSCAPES_VAL_MASKS_DIR, 
                                                         main_output_evalnet_im_dir_val,
                                                         NUM_LOOPS_VAL)
            
            del model
            tf.keras.backend.clear_session()
            gc.collect()

            modelname_evalnet_benchmarks = []

            for i in range(0,5):

                modelname_evalnet_im = f'CITYSCAPES_evalnet_miou_aug_im_{runid}_{i}'
                model_filepath_h5 = os.path.join(paths.CITYSCAPES_MODEL_DIR , f'{modelname_evalnet_im}.h5')
                
                evalnet_model = get_evalnet_miou(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, ALPHA_EVALNET)
                #evalnet_model.summary()
                total_loss, iou_loss, conf_loss, iou_mae, conf_mae = train_evalnet_miou_model_multiclass(evalnet_model, 
                                                                                                   IMAGE_HEIGHT, IMAGE_WIDTH, 
                                                                                                   main_output_evalnet_im_dir_train, 
                                                                                                   main_output_evalnet_im_dir_val, 
                                                                                                   model_filepath_h5, 
                                                                                                   BATCH_SIZE_EVALNET, 
                                                                                                   NUM_CLASSES, 
                                                                                                   NUM_EPOCHS_EVALNET)  

                modelname_evalnet_benchmarks.append((modelname_evalnet_im, total_loss, iou_loss, conf_loss, iou_mae, conf_mae))

                del evalnet_model
                tf.keras.backend.clear_session()
                gc.collect()


            sorted_mae = sorted(modelname_evalnet_benchmarks, key=lambda x: x[1], reverse=False)
            
            top_K_maes = sorted_mae[:TOP_Ks]
            
            print(top_K_maes)
            
            
            for i, top_k in enumerate(top_K_maes, start=1):
                old_filename = os.path.join(paths.CITYSCAPES_MODEL_DIR, f'{top_k[0]}.h5')
                new_filename = os.path.join(paths.CITYSCAPES_MODEL_DIR, f'{top_k[0][:-2]}_topK_{i}.h5')
                
                os.rename(old_filename, new_filename)


            Header = ['modelname', 'total_loss', 'iou_loss', 'conf_loss', 'iou_mae', 'conf_mae']
            
            os.makedirs(paths.CITYSCAPES_CSV_DIR, exist_ok=True)
            
            with open(os.path.join(paths.CITYSCAPES_CSV_DIR, f'results_{modelname_evalnet_im}.csv'), 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(Header)
                for row in modelname_evalnet_benchmarks:
                    writer.writerow(row)



        for n in range(2,3):

            for gen in range(0,5):

                modelname_benchmarks = []
                best_model_filepaths_h5 = []
                best_evalnet_filepaths_h5 = []
                best_models = []
                best_evalnets = []
                
                modelname = f'CITYSCAPES_{approach}_{runid}_n{n}_gen{gen}_e{EK}_d{DK}_bi_{BI}_bo_{BO}'

                val_pseudo_label_dir_im = os.path.join(paths.CITYSCAPES_BASE_DIR, 'val_predictions', approach, 'temp', modelname)
                test_pseudo_label_dir_im = os.path.join(paths.CITYSCAPES_BASE_DIR, 'test_predictions', approach, 'temp', modelname)
                train_unlabeled_pseudo_label_dir_im = os.path.join(paths.CITYSCAPES_BASE_DIR, 'train_unlabeled_predictions', approach, 'temp', modelname)
                train_unlabeled_pseudo_label_dir_im_images = os.path.join(train_unlabeled_pseudo_label_dir_im, 'images')
                train_unlabeled_pseudo_label_dir_im_masks = os.path.join(train_unlabeled_pseudo_label_dir_im, 'masks')

                train_unlabeled_pseudo_label_dir_im_plus_plus = os.path.join(paths.CITYSCAPES_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)
                train_unlabeled_pseudo_label_dir_im_plus_plus_images = os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus, 'images')
                train_unlabeled_pseudo_label_dir_im_plus_plus_masks = os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus, 'masks')


                if gen == 0:
                    for j in range(1,n+1):
                        best_model_filepaths_h5.append(os.path.join(paths.CITYSCAPES_MODEL_DIR, f'CITYSCAPES_subset_aug_{runid}_topK_{j}.h5'))
                else:
                    for j in range(1,n+1):
                        best_model_filepaths_h5.append(os.path.join(paths.CITYSCAPES_MODEL_DIR, f'CITYSCAPES_{approach}_{runid}_n{n}_gen{gen-1}_e{EK}_d{DK}_bi_{BI}_bo_{BO}_topK_{j}.h5'))
                
                for best_model_filepath_h5 in best_model_filepaths_h5:
                    best_model = tf.keras.models.load_model(best_model_filepath_h5)
                    best_models.append(best_model)
                
                val_mean_im_size = create_pseudo_labels_im_multiclass(best_models, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, paths.CITYSCAPES_VAL_IMAGES_DIR, val_pseudo_label_dir_im, True, EK, DK, BI, BO)
                test_mean_im_size = create_pseudo_labels_im_multiclass(best_models, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, paths.CITYSCAPES_TEST_IMAGES_DIR, test_pseudo_label_dir_im, True, EK, DK, BI, BO)
                unlabeled_mean_im_size = create_pseudo_labels_im_multiclass(best_models, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, paths.CITYSCAPES_TRAIN_UNLABELED_IMAGES_DIR, train_unlabeled_pseudo_label_dir_im, True, EK, DK, BI, BO)

                blur = max_blurs[gen]
                noise = max_noises[gen]
                brightness_range_alpha = brightness_range_alphas[gen]
                brightness_range_beta = brightness_range_betas[gen]


                for j in range(1,n+1):
                    best_evalnet_filepaths_h5.append(os.path.join(paths.CITYSCAPES_MODEL_DIR, f'CITYSCAPES_evalnet_miou_aug_im_{runid}_topK_{j}.h5'))
                
                for best_evalnet_filepath_h5 in best_evalnet_filepaths_h5:
                    best_model = tf.keras.models.load_model(best_evalnet_filepath_h5)
                    best_evalnets.append(best_model)

                create_augment_images_and_masks_with_evalnet_ensemble_multiclass(best_evalnets,
                                                                                 IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,
                                                                                 NUM_CLASSES,
                                                                                 MIN_THRESHOLD,
                                                                                 MAX_THRESHOLD,
                                                                                 train_unlabeled_pseudo_label_dir_im,
                                                                                 train_unlabeled_pseudo_label_dir_im_plus_plus,
                                                                                 brightness_range_alpha,
                                                                                 brightness_range_beta,
                                                                                 blur,
                                                                                 noise, 
                                                                                 FREE_ROTATION,
                                                                                 True)  # convert to rgb


                for imagename in tqdm(os.listdir(train_unlabeled_pseudo_label_dir_im_images)):
                    shutil.copy(os.path.join(train_unlabeled_pseudo_label_dir_im_images, imagename), os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus_images, imagename))
                    shutil.copy(os.path.join(train_unlabeled_pseudo_label_dir_im_masks, imagename), os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus_masks, imagename))

                for imagename in tqdm(os.listdir(paths.CITYSCAPES_TRAIN_LABELED_AUG_IMAGES_DIR)):
                    shutil.copy(os.path.join(paths.CITYSCAPES_TRAIN_LABELED_AUG_IMAGES_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus_images, imagename))
                    shutil.copy(os.path.join(paths.CITYSCAPES_TRAIN_LABELED_AUG_MASKS_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus_masks, imagename))
               
                num_imgs = len(os.listdir(train_unlabeled_pseudo_label_dir_im_plus_plus_images))
                STEPS_PER_EPOCH = num_imgs // BATCH_SIZE
        
                for i in range(0,5):
        
                    modelname_i = f'{modelname}_{i}'
                    filepath_h5 = os.path.join(paths.CITYSCAPES_MODEL_DIR, modelname_i + '.h5')
                    val_pred_dir = os.path.join(paths.CITYSCAPES_BASE_DIR, 'val_predictions',approach, modelname_i)
                    test_pred_dir = os.path.join(paths.CITYSCAPES_BASE_DIR, 'test_predictions', approach, modelname_i)
                    train_unlabeled_pred_dir = os.path.join(paths.CITYSCAPES_BASE_DIR, 'train_unlabeled_predictions', approach, modelname_i)
        
                    model = get_unet(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, alphas[gen], ACTIFU, ACTIFU_OUTPUT) 
        
                    mPA_val, mPA_test, mPA_train_unlabeled, mIoU_val, mIoU_test, mIoU_train_unlabeled = train_multiclass(train_unlabeled_pseudo_label_dir_im_plus_plus_images, 
                                                                                                                         paths.CITYSCAPES_VAL_IMAGES_DIR, 
                                                                                                                         paths.CITYSCAPES_VAL_MASKS_DIR,
                                                                                                                         paths.CITYSCAPES_TEST_IMAGES_DIR,
                                                                                                                         paths.CITYSCAPES_TEST_MASKS_DIR,
                                                                                                                         paths.CITYSCAPES_TRAIN_UNLABELED_IMAGES_DIR,
                                                                                                                         paths.CITYSCAPES_TRAIN_UNLABELED_MASKS_DIR,
                                                                                                                         modelname_i, 
                                                                                                                         filepath_h5, 
                                                                                                                         model,
                                                                                                                         tf.keras.losses.CategoricalCrossentropy(),
                                                                                                                         STEPS_PER_EPOCH,
                                                                                                                         IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,
                                                                                                                         NUM_CLASSES,
                                                                                                                         COLOR_TO_CLASS_MAPPING_CITYSCAPES,
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
                    old_filename = os.path.join(paths.CITYSCAPES_MODEL_DIR, f'{top_k[0]}.h5')
                    new_filename = os.path.join(paths.CITYSCAPES_MODEL_DIR, f'{top_k[0][:-2]}_topK_{i}.h5')
                    
                    os.rename(old_filename, new_filename)
                
                Header = ['modelname', 'mPA_val', 'mPA_test', 'mPA_train_unlabeled', 'mIoU_val', 'mIoU_test', 'mIoU_train_unlabeled']
                
                os.makedirs(paths.CITYSCAPES_CSV_DIR, exist_ok=True)

                
                with open(os.path.join(paths.CITYSCAPES_CSV_DIR, f'results_{modelname}.csv'), 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow(Header)
                    for row in modelname_benchmarks:
                        writer.writerow(row)


                with open(os.path.join(paths.CITYSCAPES_CSV_DIR, f'mean_im_size_{modelname}.csv'), 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow(['val_mean_im_size', 'test_mean_im_size', 'unlabeled_mean_im_size'])
                    writer.writerow([val_mean_im_size, test_mean_im_size, unlabeled_mean_im_size])    
    


