# 15_HELA_FixMatch.py

import sys
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.mixed_precision import LossScaleOptimizer
import configparser
import gc
import csv
import numpy as np
import pandas as pd
import tensorflow_addons as tfa

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import train_hela_fixmatch
from unet import get_unet
import paths

# --- CONFIGS ---
dataset_name = 'HeLa'
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

NUM_OUTPUTS = 3
ALPHA  = float(dataset_cfg['ALPHA'])
ACTIFU = str(dataset_cfg['ACTIFU'])

LR         = float(default_cfg['LR'])
WD         = float(default_cfg['WD'])
BATCH_SIZE = int(default_cfg['BATCH_SIZE'])
TOTAL_EPOCHS = 50  

CONFIDENCE_THRESHOLD  = float(approach_cfg['CONFIDENCE_THRESHOLD'])
UNLABELED_LOSS_WEIGHT = float(approach_cfg['UNLABELED_LOSS_WEIGHT'])

# --- Main Execution ---
with tf.device('/gpu:0'):
    for runid in range(1, 4):
        modelname = f'{dataset_name}_{approach}_{runid}'
        all_run_benchmarks = []
        base_model_path = os.path.join(paths.HELA_MODEL_DIR, f'HeLa_subset_{runid}_topK_1.h5')
        
        for i in range(5):
            print(f"\n--- Running Repetition {i+1}/5 for {modelname} ---")
            tf.keras.backend.clear_session()
            modelname_i = f'{modelname}_{i}'

            # HeLa: output 3 regression/probability maps -> use sigmoid head
            model = get_unet(
                i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS,
                num_outputmasks=NUM_OUTPUTS, alpha=ALPHA, actifu=ACTIFU,
                actifuout='sigmoid'
            )

            try:
                base_model = tf.keras.models.load_model(base_model_path)
                print("Transferring weights from pre-trained model...")
                model.set_weights(base_model.get_weights())
            except IOError:
                print(f"Base model not found at {base_model_path}. Starting from scratch.")
                base_model = None

            optimizer = LossScaleOptimizer(
                tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
            )
            model_save_path = os.path.join(paths.HELA_MODEL_DIR, f'{modelname_i}_best.h5')

            (mIoU_val, mIoU_test, mIoU_unlabeled,
             mIoU_ad_val, mIoU_ad_test, mIoU_ad_unlabeled,
             mcce_val, mcce_test, mcce_unlabeled,
             history) = train_hela_fixmatch(
                model=model,
                filepath_h5=model_save_path,
                modelname=modelname_i,
                h=IMAGE_HEIGHT, w=IMAGE_WIDTH, c=IMAGE_CHANNELS,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                unlabeled_loss_weight=UNLABELED_LOSS_WEIGHT,
                epochs=TOTAL_EPOCHS,
                batch_size=BATCH_SIZE,
                optimizer=optimizer,
                approach=approach.lower()
            )

            # Representative scalar for pseudo_label_utilization in the summary CSV:
            pseudo_series = history.get('pseudo_label_utilization', [])
            pseudo_label_utilization = float(pseudo_series[-1]) if len(pseudo_series) > 0 else float('nan')

            # Collect metrics for final table (add pseudo_label_utilization at the end)
            all_run_benchmarks.append((
                modelname_i,
                mIoU_val, mIoU_test, mIoU_unlabeled,
                mIoU_ad_val, mIoU_ad_test, mIoU_ad_unlabeled,
                mcce_val, mcce_test, mcce_unlabeled,
                pseudo_label_utilization
            ))

            # Save per-epoch history (includes 'pseudo_label_utilization')
            if history:
                csv_hist_path = os.path.join(paths.HELA_CSV_DIR, f'history_{modelname_i}.csv')
                pd.DataFrame(history).to_csv(csv_hist_path, index=False)

            del model
            gc.collect()

        # Final results CSV (includes 'pseudo_label_utilization')
        header = [
            'modelname',
            'mIoU_val','mIoU_test','mIoU_train_unlabeled',
            'mIoU_AD_val','mIoU_AD_test','mIoU_AD_train_unlabeled',
            'MCCE_val','MCCE_test','MCCE_train_unlabeled',
            'pseudo_label_utilization'
        ]
        csv_path = os.path.join(paths.HELA_CSV_DIR, f'results_{modelname}.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(header)
            writer.writerows(all_run_benchmarks)
        print(f"Saved final benchmark results for run {runid} to {csv_path}")
