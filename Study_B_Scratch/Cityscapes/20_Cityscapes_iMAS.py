import sys, os, gc, csv, configparser
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
from tensorflow.keras.mixed_precision import LossScaleOptimizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unet import get_unet
from functions import train_multiclass_imas
from Cityscapes_class_mapping import COLOR_TO_CLASS_MAPPING_CITYSCAPES
import paths

# --- CONFIG ---
dataset_name = 'Cityscapes'
approach = 'iMAS'
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
ALPHA          = 1.0 #float(dataset_cfg.get('ALPHA', 1.0))
ACTIFU         = str(dataset_cfg.get('ACTIFU', 'relu'))

LR            = 0.0001 #float(default_cfg['LR'])
WD            = float(default_cfg['WD'])
BATCH_SIZE    = 8 #int(default_cfg.get('BATCH_SIZE', 16))
TOTAL_EPOCHS  = 50 #int(default_cfg.get('TOTAL_EPOCHS', 5))


TAU                    = float(approach_cfg.get('TAU', 0.7))
UNLABELED_LOSS_WEIGHT  = float(approach_cfg.get('UNLABELED_LOSS_WEIGHT', 3.0))
EMA_MOMENTUM           = float(approach_cfg.get('EMA_MOMENTUM', 0.996))


with tf.device('/gpu:0'):
    for runid in range(5, 6):
        modelname = f'{dataset_name}_{approach}_{runid}'
        rows = []  
        base_model_path = os.path.join(paths.CITYSCAPES_MODEL_DIR, f'CITYSCAPES_subset_{runid}_topK_1.h5')

        for i in range(5):
            print(f"\n--- Running Repetition {i+1}/5 for {modelname} ---")
            tf.keras.backend.clear_session()
            modelname_i = f'{modelname}_{i}'

            model = get_unet(
                i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS,
                num_outputmasks=NUM_CLASSES, alpha=ALPHA, actifu=ACTIFU, actifuout='linear'
            )

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
            except Exception:
                print(f"Base model not found at {base_model_path}. Starting from scratch.")

            optimizer = LossScaleOptimizer(
                tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
            )
            save_path = os.path.join(paths.CITYSCAPES_MODEL_DIR, f'{modelname_i}_best.h5')

            (mPA_val, mPA_test, mPA_unlabeled,
             mIoU_val, mIoU_test, mIoU_unlabeled,
             history) = train_multiclass_imas(
                model=model,
                filepath_h5=save_path,
                modelname=modelname_i,
                h=IMAGE_HEIGHT, w=IMAGE_WIDTH, c=IMAGE_CHANNELS,
                num_classes=NUM_CLASSES,
                class_to_color_mapping=COLOR_TO_CLASS_MAPPING_CITYSCAPES,
                tau=TAU,
                lambda_u=UNLABELED_LOSS_WEIGHT,
                ema_momentum=EMA_MOMENTUM,
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

            hardness_series = history.get('hardness', [])
            last_hardness = float(hardness_series[-1]) if len(hardness_series) > 0 else float('nan')

            rows.append((
                modelname_i,
                mPA_val, mPA_test, mPA_unlabeled,
                mIoU_val, mIoU_test, mIoU_unlabeled,
                last_hardness
            ))

            del model
            gc.collect()

        header = [
            'modelname',
            'mPA_val','mPA_test','mPA_train_unlabeled',
            'mIoU_val','mIoU_test','mIoU_train_unlabeled',
            'hardness'
        ]
        out_csv = os.path.join(paths.CITYSCAPES_CSV_DIR, f'results_{modelname}.csv')
        with open(out_csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(header)
            w.writerows(rows)
        print(f"Saved final benchmark results for run {runid} to {out_csv}")
