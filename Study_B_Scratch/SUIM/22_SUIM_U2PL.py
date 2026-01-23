import sys, os, gc, csv, configparser
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
from tensorflow.keras.mixed_precision import LossScaleOptimizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unet import get_unet
from functions import train_multiclass_u2pl
from SUIM_class_mapping import COLOR_TO_CLASS_MAPPING_SUIM
import paths


dataset_name = 'SUIM'
approach = 'U2PL'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mixed_precision.set_global_policy('mixed_float16')

config = configparser.ConfigParser()
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.ini')
config.read(config_dir)

dataset_cfg  = config[dataset_name.upper()]
approach_cfg = config[approach.upper()] if approach.upper() in config else config['DEFAULT']
default_cfg  = config['DEFAULT']

IMAGE_HEIGHT   = int(dataset_cfg['IMAGE_HEIGHT'])
IMAGE_WIDTH    = int(dataset_cfg['IMAGE_WIDTH'])
IMAGE_CHANNELS = int(dataset_cfg['IMAGE_CHANNELS'])
NUM_CLASSES    = int(dataset_cfg['NUM_CLASSES'])
ALPHA          = float(dataset_cfg.get('ALPHA', 1.0))
ACTIFU         = str(dataset_cfg.get('ACTIFU', 'relu'))

LR            = float(default_cfg['LR'])
WD            = float(default_cfg['WD'])
BATCH_SIZE    = 4 #int(default_cfg.get('BATCH_SIZE', 8)) 
TOTAL_EPOCHS  = 50 #int(default_cfg.get('TOTAL_EPOCHS', 120))

# --- U2PL: Load hyperparameters from config.ini ---
ALPHA0                = float(approach_cfg.get('ALPHA0', 0.2))
LAMBDA_U_ETA          = float(approach_cfg.get('LAMBDA_U_ETA', 1.0))
TEMPERATURE           = 0.5 #float(approach_cfg.get('TEMPERATURE', 0.5))
PRT_RL                = int(approach_cfg.get('PRT_RL', 3))
PRT_RH                = int(approach_cfg.get('PRT_RH', 20))
DELTA_P               = 0.3 #float(approach_cfg.get('DELTA_P', 0.3))
BANK_CAPACITY         = int(approach_cfg.get('BANK_CAPACITY_PER_CLASS', 4096))
M_ANCHORS             = int(approach_cfg.get('M_ANCHORS_PER_CLASS', 16))
N_NEGS                = int(approach_cfg.get('N_NEGS_PER_CLASS', 256))
EMA_MOMENTUM          = float(approach_cfg.get('EMA_MOMENTUM', 0.996))
LAMBDA_C              = 0.1 #float(approach_cfg.get('LAMBDA_C', 1.0))


with tf.device('/gpu:0'):
    for runid in range(1, 4):
        modelname = f'{dataset_name}_{approach}_{runid}'
        rows = []
        base_model_path = os.path.join(paths.SUIM_MODEL_DIR, f'SUIM_subset_{runid}_topK_1.h5')

        for i in range(5):
            print(f"\n--- Running Repetition {i+1}/5 for {modelname} ---")
            tf.keras.backend.clear_session()
            modelname_i = f'{modelname}_{i}'

            model = get_unet(
                i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS,
                num_outputmasks=NUM_CLASSES, alpha=ALPHA, actifu=ACTIFU, actifuout='linear'
            )

            if os.path.exists(base_model_path):
                try:
                    base_model = tf.keras.models.load_model(base_model_path)
                    print("Transferring weights from pre-trained model...")
                    for layer in model.layers:
                        try:
                            bl = base_model.get_layer(name=layer.name)
                            if bl and layer.get_weights():
                                lw = layer.get_weights(); bw = bl.get_weights()
                                if lw and bw and all(a.shape == b.shape for a, b in zip(lw, bw)):
                                    layer.set_weights(bw)
                        except (ValueError, TypeError):
                            pass
                    del base_model
                except Exception as e:
                    print(f"Could not load base model: {e}. Starting from scratch.")
            else:
                print(f"Base model not found at {base_model_path}. Starting from scratch.")

            #optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.9)
            optimizer = LossScaleOptimizer(
                tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
            )
            optimizer = LossScaleOptimizer(optimizer)

            save_path = os.path.join(paths.SUIM_MODEL_DIR, f'{modelname_i}_best.h5')

            (mPA_val, mPA_test, mPA_unlabeled,
             mIoU_val, mIoU_test, mIoU_unlabeled,
             history) = train_multiclass_u2pl(
                model=model,
                filepath_h5=save_path,
                modelname=modelname_i,
                h=IMAGE_HEIGHT, w=IMAGE_WIDTH, c=IMAGE_CHANNELS,
                num_classes=NUM_CLASSES,
                class_to_color_mapping=COLOR_TO_CLASS_MAPPING_SUIM,
                alpha0=ALPHA0,
                lambda_u_eta=LAMBDA_U_ETA,
                temperature=TEMPERATURE,
                prt_rl=PRT_RL,
                prt_rh=PRT_RH,
                delta_p=DELTA_P,
                bank_capacity_per_class=BANK_CAPACITY,
                M_anchors_per_class=M_ANCHORS,
                N_negs_per_class=N_NEGS,
                ema_momentum=EMA_MOMENTUM,
                lambda_c=LAMBDA_C,
                epochs=TOTAL_EPOCHS,
                batch_size=BATCH_SIZE,
                optimizer=optimizer,
                train_labeled_images_dir=paths.SUIM_TRAIN_LABELED_IMAGES_DIR,
                train_labeled_masks_dir=paths.SUIM_TRAIN_LABELED_MASKS_DIR,
                train_unlabeled_images_dir=paths.SUIM_TRAIN_UNLABELED_IMAGES_DIR,
                val_images_dir=paths.SUIM_VAL_IMAGES_DIR,
                val_masks_dir=paths.SUIM_VAL_MASKS_DIR,
                test_images_dir=paths.SUIM_TEST_IMAGES_DIR,
                test_masks_dir=paths.SUIM_TEST_MASKS_DIR,
                train_unlabeled_masks_dir=paths.SUIM_TRAIN_UNLABELED_MASKS_DIR,
                dataset_base_dir=paths.SUIM_BASE_DIR,
                dataset_name_for_aug='SUIM',
                approach=approach.lower(),
            )

            if history:
                hist_csv = os.path.join(paths.SUIM_CSV_DIR, f'history_{modelname_i}.csv')
                pd.DataFrame(history).to_csv(hist_csv, index=False)

            avg_entropy_series = history.get('avg_entropy', [])
            last_avg_entropy = float(avg_entropy_series[-1]) if len(avg_entropy_series) > 0 else float('nan')

            rows.append((
                modelname_i,
                mPA_val, mPA_test, mPA_unlabeled,
                mIoU_val, mIoU_test, mIoU_unlabeled,
                last_avg_entropy
            ))

            del model
            gc.collect()

        header = [
            'modelname',
            'mPA_val','mPA_test','mPA_train_unlabeled',
            'mIoU_val','mIoU_test','mIoU_train_unlabeled',
            'last_avg_entropy'
        ]
        out_csv = os.path.join(paths.SUIM_CSV_DIR, f'results_{modelname}.csv')
        with open(out_csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(header)
            w.writerows(rows)
        print(f"Saved final benchmark results for run {runid} to {out_csv}")