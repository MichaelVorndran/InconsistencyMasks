import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SUIM_class_mapping import COLOR_TO_CLASS_MAPPING_SUIM
from functions import train_multiclass, create_pseudo_labels_ibas_multiclass, create_augment_images_and_masks_with_gt
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

IMAGE_WIDTH = int(config['SUIM']['IMAGE_WIDTH'])
IMAGE_HEIGHT = int(config['SUIM']['IMAGE_HEIGHT'])
IMAGE_CHANNELS = int(config['SUIM']['IMAGE_CHANNELS'])
NUM_CLASSES = int(config['SUIM']['NUM_CLASSES'])
ALPHA_EVALNET =  float(config['SUIM']['ALPHA_EVALNET'])
ALPHA =  float(config['SUIM']['ALPHA'])
ACTIFU = str(config['SUIM']['ACTIFU'])
ACTIFU_OUTPUT = str(config['SUIM']['ACTIFU_OUTPUT'])

BATCH_SIZE_EVALNET = int(config['DEFAULT']['BATCH_SIZE_EVALNET'])
NUM_EPOCHS_EVALNET = int(config['DEFAULT']['NUM_EPOCHS_EVALNET'])
NUM_LOOPS_TRAIN = int(config['DEFAULT']['NUM_LOOPS_TRAIN'])
NUM_LOOPS_VAL = int(config['DEFAULT']['NUM_LOOPS_VAL'])
BATCH_SIZE = int(config['DEFAULT']['BATCH_SIZE'])
TOP_Ks = int(config['DEFAULT']['TOP_Ks'])

EK = int(config['SUIM']['ERODE_KERNEL'])
DK = int(config['SUIM']['DILATE_KERNEL'])
BI = config['SUIM']['BLOCK_INPUT'].lower() == 'true'
BO = config['SUIM']['BLOCK_OUTPUT'].lower() == 'true'
MIN_THRESHOLD = float(config['SUIM']['MIN_THRESHOLD'])
MAX_THRESHOLD = float(config['SUIM']['MAX_THRESHOLD'])
FREE_ROTATION = config['SUIM']['FREE_ROTATION'].lower() == 'true'


approach = 'GT_IM_plus_plus'

alphas = [1, 1.25, 1.5, 1.75, 2]
max_blurs = [0,1,1,2,3]
max_noises = [5, 10, 15, 20, 25]
brightness_range_alphas = [(0.9, 1.1), (0.8, 1.2), (0.7, 1.3), (0.6, 1.4), (0.5, 1.5)]  
brightness_range_betas = [(-5, 5), (-10, 10), (-15, 15), (-20, 20), (-25, 25)]


with tf.device('/gpu:0'):

    for runid in range(1,4):


        for n in range(2,3):

            for gen in range(0,5):

                modelname_benchmarks = []
                best_model_filepaths_h5 = []
                best_evalnet_filepaths_h5 = []
                best_models = []
                best_evalnets = []
                
                modelname = f'SUIM_{approach}_{runid}_n{n}_gen{gen}_e{EK}_d{DK}_bi_{BI}_bo_{BO}'
                modelname_evalnet_base = 'SUIM_evalnet_miou_ibas'

                val_pseudo_label_dir_ibas = os.path.join(paths.SUIM_BASE_DIR, 'val_predictions', approach, 'temp', modelname)
                test_pseudo_label_dir_ibas = os.path.join(paths.SUIM_BASE_DIR, 'test_predictions', approach, 'temp', modelname)
                train_unlabeled_pseudo_label_dir_ibas = os.path.join(paths.SUIM_BASE_DIR, 'train_unlabeled_predictions', approach, 'temp', modelname)
                train_unlabeled_pseudo_label_dir_ibas_images = os.path.join(train_unlabeled_pseudo_label_dir_ibas, 'images')
                train_unlabeled_pseudo_label_dir_ibas_masks = os.path.join(train_unlabeled_pseudo_label_dir_ibas, 'masks')

                train_unlabeled_pseudo_label_dir_ibas_plus_plus = os.path.join(paths.SUIM_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)
                train_unlabeled_pseudo_label_dir_ibas_plus_plus_images = os.path.join(train_unlabeled_pseudo_label_dir_ibas_plus_plus, 'images')
                train_unlabeled_pseudo_label_dir_ibas_plus_plus_masks = os.path.join(train_unlabeled_pseudo_label_dir_ibas_plus_plus, 'masks')

                if gen == 0:
                    for j in range(1,n+1):
                        best_model_filepaths_h5.append(os.path.join(paths.SUIM_MODEL_DIR, f'SUIM_subset_{runid}_topK_{j}.h5'))
                else:
                    for j in range(1,n+1):
                        best_model_filepaths_h5.append(os.path.join(paths.SUIM_MODEL_DIR, f'SUIM_{approach}_{runid}_n{n}_gen{gen-1}_e{EK}_d{DK}_bi_{BI}_bo_{BO}_topK_{j}.h5'))

                
                for best_model_filepath_h5 in best_model_filepaths_h5:
                    best_model = tf.keras.models.load_model(best_model_filepath_h5)
                    best_models.append(best_model)
                
                val_mean_im_size = create_pseudo_labels_ibas_multiclass(best_models, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, paths.SUIM_VAL_IMAGES_DIR, val_pseudo_label_dir_ibas, True, EK, DK, BI, BO)
                test_mean_im_size = create_pseudo_labels_ibas_multiclass(best_models, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, paths.SUIM_TEST_IMAGES_DIR, test_pseudo_label_dir_ibas, True, EK, DK, BI, BO)
                unlabeled_mean_im_size = create_pseudo_labels_ibas_multiclass(best_models, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, paths.SUIM_TRAIN_UNLABELED_IMAGES_DIR, train_unlabeled_pseudo_label_dir_ibas, True, EK, DK, BI, BO)

                blur = max_blurs[gen]
                noise = max_noises[gen]
                brightness_range_alpha = brightness_range_alphas[gen]
                brightness_range_beta = brightness_range_betas[gen]

                create_augment_images_and_masks_with_gt(paths.SUIM_TRAIN_UNLABELED_MASKS_DIR,    # main_gt_input_path
                                                        MIN_THRESHOLD,
                                                        MAX_THRESHOLD,
                                                        train_unlabeled_pseudo_label_dir_ibas,
                                                        train_unlabeled_pseudo_label_dir_ibas_plus_plus,
                                                        brightness_range_alpha,
                                                        brightness_range_beta,
                                                        blur,
                                                        noise,
                                                        FREE_ROTATION,
                                                        True)   # convert to rgb


                for imagename in tqdm(os.listdir(paths.SUIM_TRAIN_LABELED_IMAGES_DIR)):
                    shutil.copy(os.path.join(paths.SUIM_TRAIN_LABELED_IMAGES_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_ibas_plus_plus_images, imagename))
                    shutil.copy(os.path.join(paths.SUIM_TRAIN_LABELED_MASKS_DIR, imagename), os.path.join(train_unlabeled_pseudo_label_dir_ibas_plus_plus_masks, imagename))
               
                num_imgs = len(os.listdir(train_unlabeled_pseudo_label_dir_ibas_plus_plus_images))
                PSEUDO_LABEL_STEPS_PER_EPOCH = num_imgs // BATCH_SIZE

                num_imgs = len(os.listdir(paths.SUIM_TRAIN_FULL_IMAGES_DIR))
                MIN_STEPS_PER_EPOCH = num_imgs // BATCH_SIZE 
                
                STEPS_PER_EPOCH = max(MIN_STEPS_PER_EPOCH, PSEUDO_LABEL_STEPS_PER_EPOCH)
        
                for i in range(0,5):
        
                    modelname_i = f'{modelname}_{i}'
                    filepath_h5 = os.path.join(paths.SUIM_MODEL_DIR, modelname_i + '.h5')
                    val_pred_dir = os.path.join(paths.SUIM_BASE_DIR, 'val_predictions',approach, modelname_i)
                    test_pred_dir = os.path.join(paths.SUIM_BASE_DIR, 'test_predictions', approach, modelname_i)
                    train_unlabeled_pred_dir = os.path.join(paths.SUIM_BASE_DIR, 'train_unlabeled_predictions', approach, modelname_i)
        
                    model = get_unet(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, NUM_CLASSES, alphas[gen], ACTIFU, ACTIFU_OUTPUT) 
        
                    mPA_val, mPA_test, mPA_train_unlabeled, mIoU_val, mIoU_test, mIoU_train_unlabeled = train_multiclass(train_unlabeled_pseudo_label_dir_ibas_plus_plus_images, 
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


                with open(os.path.join(paths.SUIM_CSV_DIR, f'mean_im_size_{modelname}.csv'), 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow(['val_mean_im_size', 'test_mean_im_size', 'unlabeled_mean_im_size'])
                    writer.writerow([val_mean_im_size, test_mean_im_size, unlabeled_mean_im_size])    
    


