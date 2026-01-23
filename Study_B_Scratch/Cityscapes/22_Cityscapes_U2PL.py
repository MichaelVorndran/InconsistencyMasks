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
from Cityscapes_class_mapping import COLOR_TO_CLASS_MAPPING_CITYSCAPES
import paths


dataset_name = 'Cityscapes'
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
BATCH_SIZE    = 2 #int(default_cfg.get('BATCH_SIZE', 8)) 
TOTAL_EPOCHS  = 50 #int(default_cfg.get('TOTAL_EPOCHS', 50)) 


ALPHA0                = float(approach_cfg.get('ALPHA0', 0.2))
LAMBDA_U_ETA          = float(approach_cfg.get('LAMBDA_U_ETA', 1.0))
TEMPERATURE           = float(approach_cfg.get('TEMPERATURE', 0.07))
PRT_RL                = int(approach_cfg.get('PRT_RL', 3))
PRT_RH                = int(approach_cfg.get('PRT_RH', 20))
DELTA_P               = float(approach_cfg.get('DELTA_P', 0.6))
BANK_CAPACITY         = int(approach_cfg.get('BANK_CAPACITY_PER_CLASS', 4096))
M_ANCHORS             = int(approach_cfg.get('M_ANCHORS_PER_CLASS', 16))
N_NEGS                = int(approach_cfg.get('N_NEGS_PER_CLASS', 256))
EMA_MOMENTUM          = float(approach_cfg.get('EMA_MOMENTUM', 0.996))
LAMBDA_C              = float(approach_cfg.get('LAMBDA_C', 1.0))


with tf.device('/gpu:0'):
    for runid in range(1, 4):
        modelname = f'{dataset_name}_{approach}_{runid}'
        rows = []  # for the final CSV

        base_model_path = os.path.join(paths.CITYSCAPES_MODEL_DIR, f'CITYSCAPES_subset_{runid}_topK_1.h5')

        for i in range(5): # 5 repetitions per run
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

            save_path = os.path.join(paths.CITYSCAPES_MODEL_DIR, f'{modelname_i}_best.h5')

            (mPA_val, mPA_test, mPA_unlabeled,
             mIoU_val, mIoU_test, mIoU_unlabeled,
             history) = train_multiclass_u2pl(
                model=model,
                filepath_h5=save_path,
                modelname=modelname_i,
                h=IMAGE_HEIGHT, w=IMAGE_WIDTH, c=IMAGE_CHANNELS,
                num_classes=NUM_CLASSES,
                class_to_color_mapping=COLOR_TO_CLASS_MAPPING_CITYSCAPES,
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
                train_labeled_images_dir=paths.CITYSCAPES_TRAIN_LABELED_IMAGES_DIR,
                train_labeled_masks_dir=paths.CITYSCAPES_TRAIN_LABELED_MASKS_DIR,
                train_unlabeled_images_dir=paths.CITYSCAPES_TRAIN_UNLABELED_IMAGES_DIR,
                val_images_dir=paths.CITYSCAPES_VAL_IMAGES_DIR,
                val_masks_dir=paths.CITYSCAPES_VAL_MASKS_DIR,
                test_images_dir=paths.CITYSCAPES_TEST_IMAGES_DIR,
                test_masks_dir=paths.CITYSCAPES_TEST_MASKS_DIR,
                train_unlabeled_masks_dir=paths.CITYSCAPES_TRAIN_UNLABELED_MASKS_DIR,
                dataset_base_dir=paths.CITYSCAPES_BASE_DIR,
                dataset_name_for_aug='CITYSCAPES',
                approach=approach.lower(),
            )

            if history:
                hist_csv = os.path.join(paths.CITYSCAPES_CSV_DIR, f'history_{modelname_i}.csv')
                pd.DataFrame(history).to_csv(hist_csv, index=False)

            alpha_series = history.get('alpha', [])
            last_alpha = float(alpha_series[-1]) if len(alpha_series) > 0 else float('nan')

            rows.append((
                modelname_i,
                mPA_val, mPA_test, mPA_unlabeled,
                mIoU_val, mIoU_test, mIoU_unlabeled,
                last_alpha
            ))

            del model
            gc.collect()

        header = [
            'modelname',
            'mPA_val','mPA_test','mPA_train_unlabeled',
            'mIoU_val','mIoU_test','mIoU_train_unlabeled',
            'final_alpha' # Represents the final unreliable proportion
        ]
        out_csv = os.path.join(paths.CITYSCAPES_CSV_DIR, f'results_{modelname}.csv')
        with open(out_csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(header)
            w.writerows(rows)
        print(f"Saved final benchmark results for run {runid} to {out_csv}")