import sys
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision, Model
from tensorflow.keras.mixed_precision import LossScaleOptimizer
import configparser
import gc
import csv
import math
import numpy as np
import pandas as pd
import tensorflow_addons as tfa

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import train_isic_2018_fixmatch, benchmark_ISIC2018, dice_loss 
from unet import get_unet
import paths

# --- CONFIGS ---
dataset_name = 'ISIC_2018'
approach = 'FixMatch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mixed_precision.set_global_policy('mixed_float16')

config = configparser.ConfigParser()
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.ini')
config.read(config_dir)

dataset_cfg  = config[dataset_name.upper()]
approach_cfg = config[approach.upper()]
default_cfg  = config['DEFAULT']

IMAGE_HEIGHT   = int(dataset_cfg['IMAGE_HEIGHT'])
IMAGE_WIDTH    = int(dataset_cfg['IMAGE_WIDTH'])
IMAGE_CHANNELS = int(dataset_cfg['IMAGE_CHANNELS'])

NUM_CLASSES = int(dataset_cfg['NUM_CLASSES_SSL']) 
ALPHA  = float(dataset_cfg['ALPHA'])
ACTIFU = str(dataset_cfg['ACTIFU'])

LR         = float(default_cfg['LR'])
WD         = float(default_cfg['WD'])
BATCH_SIZE = int(default_cfg['BATCH_SIZE'])
TOTAL_EPOCHS = 50  

CONFIDENCE_THRESHOLD  = float(approach_cfg['CONFIDENCE_THRESHOLD'])
UNLABELED_LOSS_WEIGHT = float(approach_cfg['UNLABELED_LOSS_WEIGHT'])


with tf.device('/gpu:0'):
    for runid in range(1, 4):
        modelname = f'{dataset_name}_{approach}_{runid}'
        all_run_benchmarks = []
        base_model_path = os.path.join(paths.ISIC_2018_MODEL_DIR, f'ISIC_2018_subset_{runid}_topK_1.h5')
        
        for i in range(5):
            print(f"\n--- Running Repetition {i+1}/5 for {modelname} ---")
            tf.keras.backend.clear_session()
            modelname_i = f'{modelname}_{i}'
            
            model = get_unet(
                i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS,
                num_outputmasks=NUM_CLASSES, alpha=ALPHA, actifu=ACTIFU,
                actifuout='linear' 
            )

            try:
                base_model = tf.keras.models.load_model(base_model_path, custom_objects={'dice_loss': dice_loss})
                print("Transferring weights from pre-trained model...")
                for layer in model.layers:
                    try:
                        base_layer = base_model.get_layer(name=layer.name)
                        if base_layer and layer.get_weights() and all(s1 == s2 for s1, s2 in zip(layer.get_weights()[0].shape, base_layer.get_weights()[0].shape)):
                            layer.set_weights(base_layer.get_weights())
                    except (ValueError, TypeError): pass
            except IOError:
                print(f"Base model not found at {base_model_path}. Starting from scratch.")
                base_model = None
            
            optimizer = LossScaleOptimizer(
                tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
            )
            model_save_path = os.path.join(paths.ISIC_2018_MODEL_DIR, f'{modelname_i}_best.h5')
            
            best_model, history = train_isic_2018_fixmatch(
                model=model,
                filepath_h5=model_save_path,
                h=IMAGE_HEIGHT, w=IMAGE_WIDTH, c=IMAGE_CHANNELS,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                unlabeled_loss_weight=UNLABELED_LOSS_WEIGHT,
                epochs=TOTAL_EPOCHS,
                batch_size=BATCH_SIZE,
                optimizer=optimizer
            )

            pseudo_util_series = history.get('pseudo_label_utilization', [])
            pseudo_label_utilization = float(pseudo_util_series[-1]) if len(pseudo_util_series) > 0 else float('nan')

            print(f"--- Benchmarking Final Model for Repetition {i+1} ---")
            inp = best_model.input
            logits = best_model.output

            probs = tf.keras.layers.Softmax(dtype='float32')(logits)
            lesion_prob = probs[..., 1:]
            inference_model = Model(inputs=inp, outputs=lesion_prob)

            val_pred_dir       = os.path.join(paths.ISIC_2018_BASE_DIR, 'val_predictions', approach, modelname_i)
            test_pred_dir      = os.path.join(paths.ISIC_2018_BASE_DIR, 'test_predictions', approach, modelname_i)
            unlabeled_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'train_unlabeled_predictions', approach, modelname_i)
            
            mIoU_val, d_val = benchmark_ISIC2018(
                inference_model,
                paths.ISIC_2018_VAL_IMAGES_DIR, paths.ISIC_2018_VAL_MASKS_DIR,
                val_pred_dir, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
            )
            mIoU_test, d_test = benchmark_ISIC2018(
                inference_model,
                paths.ISIC_2018_TEST_IMAGES_DIR, paths.ISIC_2018_TEST_MASKS_DIR,
                test_pred_dir, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
            )
            mIoU_unlabeled, d_unlabeled = benchmark_ISIC2018(
                inference_model,
                paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR, paths.ISIC_2018_TRAIN_UNLABELED_MASKS_DIR,
                unlabeled_pred_dir, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
            )
            
            all_run_benchmarks.append((
                modelname_i,
                mIoU_val, mIoU_test, mIoU_unlabeled,
                d_val, d_test, d_unlabeled,
                pseudo_label_utilization
            ))

            if history:
                csv_save_path = os.path.join(paths.ISIC_2018_CSV_DIR, f'history_{modelname_i}.csv')
                pd.DataFrame(history).to_csv(csv_save_path, index=False)

            del best_model, inference_model, model
            gc.collect()
            
        header = [
            'modelname',
            'mIoU_val', 'mIoU_test', 'mIoU_train_unlabeled',
            'dice_score_val', 'dice_score_test', 'dice_score_train_unlabeled',
            'pseudo_label_utilization'
        ]
        csv_path = os.path.join(paths.ISIC_2018_CSV_DIR, f'results_{modelname}.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(header)
            writer.writerows(all_run_benchmarks)
        print(f"Saved final benchmark results for run {runid} to {csv_path}")
