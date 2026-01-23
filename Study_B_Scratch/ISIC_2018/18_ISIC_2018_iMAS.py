import sys
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision, Model
from tensorflow.keras.mixed_precision import LossScaleOptimizer
import configparser
import gc
import csv
import numpy as np
import pandas as pd
import tensorflow_addons as tfa

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import (
    train_isic_imas,  
    dice_loss         
)
from unet import get_unet
import paths

# --- CONFIGS ---
dataset_name = 'ISIC_2018'
approach = 'iMAS'
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

NUM_CLASSES = 2
ALPHA  = float(dataset_cfg['ALPHA'])
ACTIFU = str(dataset_cfg['ACTIFU'])

LR         = 0.0001 #float(default_cfg['LR'])
WD         = float(default_cfg['WD'])
BATCH_SIZE = int(default_cfg['BATCH_SIZE'])
TOTAL_EPOCHS = 50  


TAU                    = float(approach_cfg['TAU'])
UNLABELED_LOSS_WEIGHT  = float(approach_cfg['UNLABELED_LOSS_WEIGHT'])


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
                actifuout='linear'  # logits
            )

            try:
                base_model = tf.keras.models.load_model(base_model_path, custom_objects={'dice_loss': dice_loss})
                print("Transferring weights from pre-trained model...")
                for layer in model.layers:
                    try:
                        base_layer = base_model.get_layer(name=layer.name)
                        if base_layer and layer.get_weights():

                            w_src = base_layer.get_weights()
                            w_dst = layer.get_weights()
                            if len(w_src) == len(w_dst):
                                ok = True
                                for a, b in zip(w_src, w_dst):
                                    if a.shape != b.shape:
                                        ok = False; break
                                if ok:
                                    layer.set_weights(w_src)
                    except (ValueError, TypeError):
                        pass
            except IOError:
                print(f"Base model not found at {base_model_path}. Starting from scratch.")
                base_model = None

            optimizer = LossScaleOptimizer(
                tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
            )
            model_save_path = os.path.join(paths.ISIC_2018_MODEL_DIR, f'{modelname_i}_best.h5')

            (mIoU_val, mIoU_test, mIoU_unlabeled,
             dice_val, dice_test, dice_unlabeled,
             history) = train_isic_imas(
                model=model,
                filepath_h5=model_save_path,
                modelname=modelname_i,
                h=IMAGE_HEIGHT, w=IMAGE_WIDTH, c=IMAGE_CHANNELS,
                tau=TAU,
                lambda_u=UNLABELED_LOSS_WEIGHT,
                epochs=TOTAL_EPOCHS,
                batch_size=BATCH_SIZE,
                optimizer=optimizer,
                approach=approach.lower()
            )

            imas_util_series = history.get('imas_utilization', [])
            imas_utilization = float(imas_util_series[-1]) if len(imas_util_series) > 0 else float('nan')

            all_run_benchmarks.append((
                modelname_i,
                mIoU_val, mIoU_test, mIoU_unlabeled,
                dice_val, dice_test, dice_unlabeled,
                imas_utilization
            ))

            if history:
                csv_hist_path = os.path.join(paths.ISIC_2018_CSV_DIR, f'history_{modelname_i}.csv')
                pd.DataFrame(history).to_csv(csv_hist_path, index=False)

            del model
            gc.collect()


        header = [
            'modelname',
            'mIoU_val', 'mIoU_test', 'mIoU_train_unlabeled',
            'dice_score_val', 'dice_score_test', 'dice_score_train_unlabeled',
            'imas_utilization'
        ]
        csv_path = os.path.join(paths.ISIC_2018_CSV_DIR, f'results_{modelname}.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(header)
            writer.writerows(all_run_benchmarks)
        print(f"Saved final benchmark results for run {runid} to {csv_path}")
