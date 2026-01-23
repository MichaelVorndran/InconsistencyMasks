# HeLa_U2PL.py

import sys, os, gc, csv, configparser
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
from tensorflow.keras.mixed_precision import LossScaleOptimizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unet import get_unet
from functions import train_hela_u2pl
import paths

# --- CONFIG ---
dataset_name = 'HeLa'
approach = 'U2PL'
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
ALPHA  = float(dataset_cfg.get('ALPHA', 1.0))
ACTIFU = str(dataset_cfg.get('ACTIFU', 'relu'))

LR           = 0.0001 #float(default_cfg['LR'])
WD           = float(default_cfg['WD'])
BATCH_SIZE   = 4 #int(default_cfg['BATCH_SIZE'])
TOTAL_EPOCHS = 50 #int(default_cfg.get('TOTAL_EPOCHS', 50))

CONFIDENCE_THRESHOLD  = float(approach_cfg.get('CONFIDENCE_THRESHOLD', 0.8))
UNLABELED_LOSS_WEIGHT = float(approach_cfg.get('UNLABELED_LOSS_WEIGHT', 1.0))
OHEM_FRACTION         = float(approach_cfg.get('OHEM_FRACTION', 0.25))  # top-q unreliable fraction
CHANNEL_WEIGHTS       = (1.0, 1.0, 3.0)

# --- Main ---
with tf.device('/gpu:0'):
    for runid in range(1, 4):
        modelname = f'{dataset_name}_{approach}_{runid}'
        all_rows = []
        base_model_path = os.path.join(paths.HELA_MODEL_DIR, f'HeLa_subset_{runid}_topK_1.h5')

        for i in range(5):
            print(f"\n--- Running Repetition {i+1}/5 for {modelname} ---")
            tf.keras.backend.clear_session()
            modelname_i = f'{modelname}_{i}'

            # HeLa: sigmoid head (probability per channel)
            model = get_unet(
                i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS,
                num_outputmasks=NUM_OUTPUTS, alpha=ALPHA, actifu=ACTIFU, actifuout='sigmoid'
            )

            # optional weight transfer
            try:
                base_model = tf.keras.models.load_model(base_model_path)
                print("Transferring weights from pre-trained model...")
                for layer in model.layers:
                    try:
                        bl = base_model.get_layer(name=layer.name)
                        if bl is not None and layer.get_weights() and bl.get_weights():
                            lw = layer.get_weights(); bw = bl.get_weights()
                            if all(a.shape == b.shape for a, b in zip(lw, bw)):
                                layer.set_weights(bw)
                    except Exception:
                        pass
            except Exception:
                print(f"Base model not found at {base_model_path}. Starting from scratch.")

            #optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9)
            optimizer = LossScaleOptimizer(
                tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
            )
            optimizer = LossScaleOptimizer(optimizer)

            save_path = os.path.join(paths.HELA_MODEL_DIR, f'{modelname_i}_best.h5')

            (mIoU_val, mIoU_test, mIoU_unlabeled,
             mIoU_ad_val, mIoU_ad_test, mIoU_ad_unlabeled,
             mcce_val, mcce_test, mcce_unlabeled,
             history) = train_hela_u2pl(
                model=model,
                filepath_h5=save_path,
                modelname=modelname_i,
                h=IMAGE_HEIGHT, w=IMAGE_WIDTH, c=IMAGE_CHANNELS,
                #confidence_threshold=CONFIDENCE_THRESHOLD,
                #unlabeled_loss_weight=UNLABELED_LOSS_WEIGHT,
                #ohem_fraction=OHEM_FRACTION,
                epochs=TOTAL_EPOCHS,
                batch_size=BATCH_SIZE,
                optimizer=optimizer,
                approach=approach.lower(),
                channel_weights=CHANNEL_WEIGHTS,
            )

            # Save per-epoch history
            if history:
                hist_csv = os.path.join(paths.HELA_CSV_DIR, f'history_{modelname_i}.csv')
                pd.DataFrame(history).to_csv(hist_csv, index=False)

            # Collect row (same columns you use elsewhere)
            all_rows.append((
                modelname_i,
                mIoU_val, mIoU_test, mIoU_unlabeled,
                mIoU_ad_val, mIoU_ad_test, mIoU_ad_unlabeled,
                mcce_val, mcce_test, mcce_unlabeled
            ))

            del model
            gc.collect()

        # Final CSV
        header = [
            'modelname',
            'mIoU_val','mIoU_test','mIoU_train_unlabeled',
            'mIoU_AD_val','mIoU_AD_test','mIoU_AD_train_unlabeled',
            'MCCE_val','MCCE_test','MCCE_train_unlabeled'
        ]
        out_csv = os.path.join(paths.HELA_CSV_DIR, f'results_{modelname}.csv')
        with open(out_csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(header)
            w.writerows(all_rows)
        print(f"Saved final benchmark results for run {runid} to {out_csv}")
