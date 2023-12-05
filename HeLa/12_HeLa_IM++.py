import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import train_hela, create_pseudo_labels_im_hela, create_augment_images_and_masks_with_evalnet_ensemble_hela, train_evalnet_miou_model_hela, create_training_data_evalnet_miou_im_hela, ignore_im_categorical_crossentropy, ignore_im_dice_loss_multiclass
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

config = configparser.ConfigParser()
config.read(os.path.join('IM', 'config.ini'))

IMAGE_WIDTH = int(config['HELA']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['HELA']['IMAGE_HEIGHT'])
IMAGE_CHANNELS = int(config['HELA']['IMAGE_CHANNELS'])
NUM_CLASSES = int(config['HELA']['NUM_CLASSES'])
ALPHA_EVALNET =  float(config['HELA']['ALPHA_EVALNET'])
ALPHA =  float(config['HELA']['ALPHA'])
ACTIFU = str(config['HELA']['ACTIFU'])
ACTIFU_OUTPUT = str(config['HELA']['ACTIFU_OUTPUT'])

BATCH_SIZE_EVALNET = int(config['DEFAULT']['BATCH_SIZE_EVALNET'])
NUM_EPOCHS_EVALNET = int(config['DEFAULT']['NUM_EPOCHS_EVALNET'])
NUM_LOOPS_TRAIN = int(config['DEFAULT']['NUM_LOOPS_TRAIN'])
NUM_LOOPS_VAL = int(config['DEFAULT']['NUM_LOOPS_VAL'])
BATCH_SIZE = int(config['DEFAULT']['BATCH_SIZE'])
TOP_Ks = int(config['DEFAULT']['TOP_Ks'])

EK = int(config['HELA']['ERODE_KERNEL'])
DK = int(config['HELA']['DILATE_KERNEL'])
BI = config['HELA']['BLOCK_INPUT'].lower() == 'true'
BO = config['HELA']['BLOCK_OUTPUT'].lower() == 'true'
MIN_THRESHOLD = float(config['HELA']['MIN_THRESHOLD'])
MAX_THRESHOLD = float(config['HELA']['MAX_THRESHOLD'])
FREE_ROTATION = config['HELA']['FREE_ROTATION'].lower() == 'true'


approach = 'IM_plus_plus'

alphas = [1, 1.25, 1.5, 1.75, 2]
max_blurs = [0,1,1,2,3]
max_noises = [5, 10, 15, 20, 25]
brightness_range_alphas = [(0.9, 1.1), (0.9, 1.1), (0.8, 1.2), (0.8, 1.2), (0.7, 1.3)]  
brightness_range_betas = [(-3, 3), (-6, 6), (-9, 9), (-12, 12), (-15, 15)]

train_new_evalnet = True


with tf.device('/gpu:0'):

    for runid in range(1,4):

        if train_new_evalnet == True:
        
            main_output_evalnet_im_dir_train = os.path.join(paths.HELA_BASE_DIR, 'evalnet_im', f'run_{runid}', 'train')
            main_output_evalnet_im_dir_val = os.path.join(paths.HELA_BASE_DIR, 'evalnet_im', f'run_{runid}', 'val')
        
            subset_models = []

            for modelname in os.listdir(paths.HELA_MODEL_DIR):
            
                if f'HELA_subset_{runid}' in modelname:        
            
                    model_filepath_h5 = os.path.join(paths.HELA_MODEL_DIR, modelname)
            
                    model = tf.keras.models.load_model(model_filepath_h5)
                    subset_models.append(model)
            
            
            create_training_data_evalnet_miou_im_hela(subset_models,
                                                         IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, 
                                                         paths.HELA_TRAIN_LABELED_DIR, 
                                                         main_output_evalnet_im_dir_train,
                                                         NUM_LOOPS_TRAIN)
            
            create_training_data_evalnet_miou_im_hela(subset_models,
                                                         IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, 
                                                         paths.HELA_VAL_DIR, 
                                                         main_output_evalnet_im_dir_val,
                                                         NUM_LOOPS_VAL)
            
            del model
            tf.keras.backend.clear_session()
            gc.collect()

            modelname_evalnet_benchmarks = []

            for i in range(0,5):

                modelname_evalnet_im = f'HELA_evalnet_miou_im_{runid}_{i}'
                model_filepath_h5 = os.path.join(paths.HELA_MODEL_DIR , f'{modelname_evalnet_im}.h5')
                
                evalnet_model = get_evalnet_miou(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, ALPHA_EVALNET)
                #evalnet_model.summary()

                total_loss, iou_loss, detection_loss, iou_mae, detection_mae = train_evalnet_miou_model_hela(evalnet_model, 
                                                                                                             main_output_evalnet_im_dir_train, 
                                                                                                             main_output_evalnet_im_dir_val, 
                                                                                                             model_filepath_h5, 
                                                                                                             BATCH_SIZE_EVALNET, 
                                                                                                             NUM_EPOCHS_EVALNET)  


                modelname_evalnet_benchmarks.append((modelname_evalnet_im, total_loss, iou_loss, detection_loss, iou_mae, detection_mae))

                del evalnet_model
                tf.keras.backend.clear_session()
                gc.collect()


            sorted_mae = sorted(modelname_evalnet_benchmarks, key=lambda x: x[4], reverse=False)
            
            top_K_maes = sorted_mae[:TOP_Ks]
            
            print(top_K_maes)
            
            
            for i, top_k in enumerate(top_K_maes, start=1):
                old_filename = os.path.join(paths.HELA_MODEL_DIR, f'{top_k[0]}.h5')
                new_filename = os.path.join(paths.HELA_MODEL_DIR, f'{top_k[0][:-2]}_topK_{i}.h5')
                
                os.rename(old_filename, new_filename)


            Header = ['modelname', 'total_loss', 'iou_loss', 'detection_loss', 'iou_mae', 'detection_mae']
            
            os.makedirs(paths.HELA_CSV_DIR, exist_ok=True)
            
            with open(os.path.join(paths.HELA_CSV_DIR, f'results_{modelname_evalnet_im}.csv'), 'w', encoding='utf-8', newline='') as f:
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
                
                modelname = f'HELA_{approach}_{runid}_n{n}_gen{gen}_e{EK}_d{DK}_bi_{BI}_bo_{BO}'

                val_pseudo_label_dir_im = os.path.join(paths.HELA_BASE_DIR, 'val_predictions', approach, 'temp', modelname)
                test_pseudo_label_dir_im = os.path.join(paths.HELA_BASE_DIR, 'test_predictions', approach, 'temp', modelname)
                train_unlabeled_pseudo_label_dir_im = os.path.join(paths.HELA_BASE_DIR, 'train_unlabeled_predictions', approach, 'temp', modelname)
                train_unlabeled_pseudo_label_dir_im_bf_images = os.path.join(train_unlabeled_pseudo_label_dir_im, 'brightfield')
                train_unlabeled_pseudo_label_dir_im_alive_masks = os.path.join(train_unlabeled_pseudo_label_dir_im, 'alive')
                train_unlabeled_pseudo_label_dir_im_dead_masks = os.path.join(train_unlabeled_pseudo_label_dir_im, 'dead')
                train_unlabeled_pseudo_label_dir_im_pos_masks = os.path.join(train_unlabeled_pseudo_label_dir_im, 'mod_position')


                train_unlabeled_pseudo_label_dir_im_plus_plus = os.path.join(paths.HELA_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)
                train_unlabeled_pseudo_label_dir_im_plus_plus_bf_images = os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus, 'brightfield')
                train_unlabeled_pseudo_label_dir_im_plus_plus_alive_masks = os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus, 'alive')
                train_unlabeled_pseudo_label_dir_im_plus_plus_dead_masks = os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus, 'dead')
                train_unlabeled_pseudo_label_dir_im_plus_plus_pos_masks = os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus, 'mod_position')


                if gen == 0:
                    for j in range(1,n+1):
                        best_model_filepaths_h5.append(os.path.join(paths.HELA_MODEL_DIR, f'HELA_subset_{runid}_topK_{j}.h5'))
                else:
                    for j in range(1,n+1):
                        best_model_filepaths_h5.append(os.path.join(paths.HELA_MODEL_DIR, f'HELA_{approach}_{runid}_n{n}_gen{gen-1}_e{EK}_d{DK}_bi_{BI}_bo_{BO}_topK_{j}.h5'))
                
                for best_model_filepath_h5 in best_model_filepaths_h5:
                    best_model = tf.keras.models.load_model(best_model_filepath_h5)
                    best_models.append(best_model)

                val_mean_im_size = create_pseudo_labels_im_hela(best_models, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, paths.HELA_VAL_BRIGHTFIELD_DIR, val_pseudo_label_dir_im, EK, DK, BI, BO)
                test_mean_im_size = create_pseudo_labels_im_hela(best_models, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, paths.HELA_TEST_BRIGHTFIELD_DIR, test_pseudo_label_dir_im, EK, DK, BI, BO)
                unlabeled_mean_im_size = create_pseudo_labels_im_hela(best_models, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, paths.HELA_TRAIN_UNLABELED_BRIGHTFIELD_DIR, train_unlabeled_pseudo_label_dir_im, EK, DK, BI, BO)
                
                blur = max_blurs[gen]
                noise = max_noises[gen]
                brightness_range_alpha = brightness_range_alphas[gen]
                brightness_range_beta = brightness_range_betas[gen]


                for j in range(1,n+1):
                    best_evalnet_filepaths_h5.append(os.path.join(paths.HELA_MODEL_DIR, f'HELA_evalnet_miou_im_{runid}_topK_{j}.h5'))
                
                for best_evalnet_filepath_h5 in best_evalnet_filepaths_h5:
                    best_model = tf.keras.models.load_model(best_evalnet_filepath_h5)
                    best_evalnets.append(best_model)

                create_augment_images_and_masks_with_evalnet_ensemble_hela(best_evalnets,
                                                                                 IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,
                                                                                 MIN_THRESHOLD,
                                                                                 MAX_THRESHOLD,
                                                                                 train_unlabeled_pseudo_label_dir_im,
                                                                                 train_unlabeled_pseudo_label_dir_im_plus_plus,
                                                                                 brightness_range_alpha,
                                                                                 brightness_range_beta,
                                                                                 blur,
                                                                                 noise, 
                                                                                 FREE_ROTATION)


                for imagename in tqdm(os.listdir(paths.HELA_TRAIN_LABELED_BRIGHTFIELD_DIR)):
                    shutil.copy(os.path.join(paths.HELA_TRAIN_LABELED_BRIGHTFIELD_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus_bf_images, imagename))
                    shutil.copy(os.path.join(paths.HELA_TRAIN_LABELED_ALIVE_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus_alive_masks, imagename))
                    shutil.copy(os.path.join(paths.HELA_TRAIN_LABELED_DEAD_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus_dead_masks, imagename))
                    shutil.copy(os.path.join(paths.HELA_TRAIN_LABELED_MOD_POS_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_im_plus_plus_pos_masks, imagename))

 
                num_imgs = len(os.listdir(train_unlabeled_pseudo_label_dir_im_plus_plus_bf_images))
                STEPS_PER_EPOCH = num_imgs // BATCH_SIZE
        
                for i in range(0,5):
        
                    modelname_i = f'{modelname}_{i}'
                    filepath_h5 = os.path.join(paths.HELA_MODEL_DIR, modelname_i + '.h5')
                    val_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'val_predictions',approach, modelname_i)
                    test_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'test_predictions', approach, modelname_i)
                    train_unlabeled_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'train_unlabeled_predictions', approach, modelname_i)
        
                    model = get_unet(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, alphas[gen], ACTIFU, ACTIFU_OUTPUT) 
        
                    mIoU_val, mIoU_ad_val, mcce_val, mIoU_test, mIoU_ad_test, mcce_test, mIoU_unlabeled, mIoU_ad_unlabeled, mcce_unlabeled = train_hela(
                                                                                                            train_unlabeled_pseudo_label_dir_im_plus_plus_bf_images, 
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
        
        
                sorted_mIoU_val = sorted(modelname_benchmarks, key=lambda x: x[4], reverse=True)
                
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


                with open(os.path.join(paths.HELA_CSV_DIR, f'mean_im_size_{modelname}.csv'), 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow(['val_mean_im_size', 'test_mean_im_size', 'unlabeled_mean_im_size'])
                    writer.writerow([val_mean_im_size, test_mean_im_size, unlabeled_mean_im_size])    
    


