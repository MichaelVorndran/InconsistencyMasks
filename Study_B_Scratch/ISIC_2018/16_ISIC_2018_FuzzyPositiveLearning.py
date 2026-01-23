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
    train_isic_2018_fpl,
    benchmark_ISIC2018,
    plot_fpl_history,
    save_fpl_history_to_csv,
    dice_loss, 
)
from unet import get_unet
import paths

# --- CONFIGS ---
dataset_name = 'ISIC_2018'
approach = 'FPL'
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
NUM_CLASSES    = int(dataset_cfg['NUM_CLASSES_SSL'])  # binary (bg, lesion)
ALPHA  = float(dataset_cfg['ALPHA'])
ACTIFU = str(dataset_cfg['ACTIFU'])

LR         = float(default_cfg['LR'])
WD         = float(default_cfg['WD'])
BATCH_SIZE = int(default_cfg['BATCH_SIZE'])
TOTAL_EPOCHS = 50


T_THRESHOLD           = float(approach_cfg['T_THRESHOLD'])
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

            
            fpl_model = get_unet(
                i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS,
                num_outputmasks=NUM_CLASSES, alpha=ALPHA, actifu=ACTIFU,
                actifuout='linear'
            )

            base_model = get_unet(
                i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS,
                num_outputmasks=NUM_CLASSES, alpha=ALPHA, actifu=ACTIFU,
                actifuout='linear'
            )


            try:
                pretrained = tf.keras.models.load_model(base_model_path, custom_objects={'dice_loss': dice_loss})
                print("Transferring weights from pre-trained model...")
                for layer in fpl_model.layers:
                    try:
                        pl = pretrained.get_layer(name=layer.name)
                        if pl is not None and layer.get_weights() and pl.get_weights() and \
                           all(a.shape == b.shape for a, b in zip(layer.get_weights(), pl.get_weights())):
                            layer.set_weights(pl.get_weights())
                    except Exception:
                        pass

                base_model.set_weights(fpl_model.get_weights())
            except Exception:
                print(f"Base model not found at {base_model_path}. Starting from scratch.")

            optimizer = LossScaleOptimizer(tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD))
            model_save_path = os.path.join(paths.ISIC_2018_MODEL_DIR, f'{modelname_i}_best.h5')

            (mIoU_val, mIoU_test, mIoU_unlabeled,
            dice_val, dice_test, dice_unlabeled,
            history) = train_isic_2018_fpl(
                model=fpl_model,
                base_model_to_save=base_model,
                filepath_h5=model_save_path,
                modelname=modelname_i,
                h=IMAGE_HEIGHT, w=IMAGE_WIDTH, c=IMAGE_CHANNELS,
                t_threshold=T_THRESHOLD,
                unlabeled_loss_weight=UNLABELED_LOSS_WEIGHT,
                epochs=TOTAL_EPOCHS,
                batch_size=BATCH_SIZE,
                optimizer=optimizer,
                approach=approach.lower()
            )


            fuzzy_series = history.get('fuzzy_positive_ratio', [])
            fuzzy_positive_ratio = float(fuzzy_series[-1]) if len(fuzzy_series) > 0 else float('nan')

            all_run_benchmarks.append((
                modelname_i,
                mIoU_val, mIoU_test, mIoU_unlabeled,
                dice_val, dice_test, dice_unlabeled,
                fuzzy_positive_ratio
            ))


            if history:
                csv_save_path = os.path.join(paths.ISIC_2018_CSV_DIR, f'history_{modelname_i}.csv')
                pd.DataFrame(history).to_csv(csv_save_path, index=False)

            del fpl_model
            gc.collect()

        header = [
            'modelname',
            'mIoU_val', 'mIoU_test', 'mIoU_train_unlabeled',
            'dice_score_val', 'dice_score_test', 'dice_score_train_unlabeled',
            'fuzzy_positive_ratio'
        ]
        csv_path = os.path.join(paths.ISIC_2018_CSV_DIR, f'results_{modelname}.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(header)
            writer.writerows(all_run_benchmarks)
        print(f"Saved final benchmark results for run {runid} to {csv_path}")
