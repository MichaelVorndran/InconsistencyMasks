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

from functions import load_labeled_data_single_mask, load_unlabeled_images, train_isic_ups, get_uncertainty_and_predictions_mc_dropout, dice_loss
from unet import get_unet
import paths

# --- CONFIGS ---
dataset_name = 'ISIC_2018'
approach = 'UPS'
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
NUM_CLASSES    = 2
ALPHA  = float(dataset_cfg['ALPHA'])
ACTIFU = str(dataset_cfg['ACTIFU'])

LR         = float(default_cfg['LR'])
WD         = float(default_cfg['WD'])
BATCH_SIZE = int(default_cfg['BATCH_SIZE'])
NUM_EPOCHS_FINETUNE = int(default_cfg.get('NUM_EPOCHS_FINETUNE', 5))
EPOCHS_PER_STUDENT = 50 #int(default_cfg.get('EPOCHS_UPS', 5))
MAX_GENERATIONS = int(approach_cfg.get('MAX_GENERATIONS', 5))

TAU_POS   = 0.7 #0.9
KAPPA_POS = 0.05 #0.02
TAU_NEG   = 0.05 #0.02
KAPPA_NEG = 0.005 #0.003
MC_PASSES = int(approach_cfg.get('MC_DROPOUT_PASSES', 10))
DROPOUT_RATE = float(approach_cfg.get('DROPOUT_RATE', 0.10))
UNLABELED_LOSS_WEIGHT = 0.1 #float(approach_cfg.get('UNLABELED_LOSS_WEIGHT', 1.0))


CONVERGENCE_THRESHOLD = float(approach_cfg.get('CONVERGENCE_THRESHOLD', 0.01))
CONVERGENCE_PATIENCE = int(approach_cfg.get('CONVERGENCE_PATIENCE', 2))


with tf.device('/gpu:0'):
    for runid in range(1, 4):
        modelname = f'{dataset_name}_{approach}_{runid}'
        base_model_path = os.path.join(paths.ISIC_2018_MODEL_DIR, f'ISIC_2018_subset_{runid}_topK_1.h5')

        lab_imgs_np, lab_masks_np = load_labeled_data_single_mask(paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR, paths.ISIC_2018_TRAIN_LABELED_MASKS_DIR)
        unl_imgs_np = load_unlabeled_images(paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR)
        val_imgs_np, val_masks_np = load_labeled_data_single_mask(paths.ISIC_2018_VAL_IMAGES_DIR, paths.ISIC_2018_VAL_MASKS_DIR)
        
        lab_ds = tf.data.Dataset.from_tensor_slices((lab_imgs_np, lab_masks_np)).shuffle(len(lab_imgs_np)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((val_imgs_np, val_masks_np)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        teacher_model = get_unet(i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS, num_outputmasks=NUM_CLASSES, alpha=ALPHA, actifu=ACTIFU, actifuout='linear', dropout_rate_encoder=DROPOUT_RATE, dropout_rate_bottleneck=DROPOUT_RATE, dropout_rate_decoder=DROPOUT_RATE)
        
        initial_student_weights = None
        if os.path.exists(base_model_path):
            try:
                base_model = tf.keras.models.load_model(base_model_path, custom_objects={'dice_loss': dice_loss})
                initial_student_weights = base_model.get_weights()
                print("Initial student weights loaded from pre-trained model.")
                plain_model_for_weights = get_unet(i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS, num_outputmasks=1, alpha=ALPHA, actifu=ACTIFU, actifuout='sigmoid')
                plain_model_for_weights.set_weights(initial_student_weights)
                for layer in teacher_model.layers:
                    try:
                        base_layer = plain_model_for_weights.get_layer(name=layer.name)
                        if base_layer and all(a.shape == b.shape for a,b in zip(layer.get_weights(), base_layer.get_weights())):
                            layer.set_weights(base_layer.get_weights())
                    except: pass
                del plain_model_for_weights
            except Exception as e:
                print(f"Error loading base model: {e}. Starting from scratch.")
        else:
            print(f"Base model not found at {base_model_path}. Starting from scratch.")

        if NUM_EPOCHS_FINETUNE > 0:
            print(f"Fine-tuning teacher for {NUM_EPOCHS_FINETUNE} epochs to adapt dropout...")
            opt_ft = LossScaleOptimizer(tfa.optimizers.AdamW(learning_rate=LR/10.0, weight_decay=WD))
            teacher_model.compile(optimizer=opt_ft, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            teacher_model.fit(lab_ds, validation_data=val_ds, epochs=NUM_EPOCHS_FINETUNE, verbose=2)
        
        last_num_selected_pixels = -1
        patience_counter = 0

        for gen_num in range(MAX_GENERATIONS):
            print(f"\n--- {modelname}: GENERATION {gen_num}/{MAX_GENERATIONS-1} ---")
            
            print("Generating pseudo-labels with current teacher...")
            mean_probs, uncertainty_per_class = get_uncertainty_and_predictions_mc_dropout(teacher_model, unl_imgs_np, num_passes=MC_PASSES, batch_size=8)

            print("Selecting reliable pseudo-labels using NumPy...")
            mean_probs_np = mean_probs.numpy()
            uncertainty_np = uncertainty_per_class.numpy()
            del mean_probs, uncertainty_per_class
            gc.collect()

            pos_select_mask_bool = (mean_probs_np >= TAU_POS) & (uncertainty_np <= KAPPA_POS)
            neg_select_mask_bool = (mean_probs_np <= TAU_NEG) & (uncertainty_np <= KAPPA_NEG)
            
            pos_select_mask = pos_select_mask_bool.astype(np.float32)
            neg_select_mask = neg_select_mask_bool.astype(np.float32)
            
            pseudo_labels_1h = tf.one_hot(np.argmax(mean_probs_np, axis=-1), NUM_CLASSES, dtype=tf.float32)
            del mean_probs_np, uncertainty_np # Free memory
            gc.collect()

            num_selected_pixels = np.sum(pos_select_mask_bool | neg_select_mask_bool)
            total_possible_pixels = np.prod(unl_imgs_np.shape[:-1]) * NUM_CLASSES
            selection_ratio = num_selected_pixels / total_possible_pixels if total_possible_pixels > 0 else 0
            print(f"Total selected pseudo-entries: {num_selected_pixels} ({selection_ratio:.2%})")

            
            gen_benchmarks = []; student_weights_dict = {}

            for i in range(5):
                print(f"\n--- Training Student {i+1}/5 for Generation {gen_num} ---")
                tf.keras.backend.clear_session()
                modelname_i = f'{modelname}_gen{gen_num}_{i}'

                student_model = get_unet(i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS, num_outputmasks=NUM_CLASSES, alpha=ALPHA, actifu=ACTIFU, actifuout='linear', dropout_rate_encoder=DROPOUT_RATE, dropout_rate_bottleneck=DROPOUT_RATE, dropout_rate_decoder=DROPOUT_RATE)
                if initial_student_weights:
                    plain_model_for_weights = get_unet(i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS, num_outputmasks=1, alpha=ALPHA, actifu=ACTIFU, actifuout='sigmoid')
                    plain_model_for_weights.set_weights(initial_student_weights)
                    for layer in student_model.layers:
                        try:
                            base_layer = plain_model_for_weights.get_layer(name=layer.name)
                            if base_layer and all(a.shape == b.shape for a,b in zip(layer.get_weights(), base_layer.get_weights())):
                                layer.set_weights(base_layer.get_weights())
                        except: pass
                    del plain_model_for_weights
                
                base_model_to_save = get_unet(i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS, num_outputmasks=NUM_CLASSES, alpha=ALPHA, actifu=ACTIFU, actifuout='linear')
                optimizer = LossScaleOptimizer(tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD))
                model_save_path = os.path.join(paths.ISIC_2018_MODEL_DIR, f'{modelname_i}_best.h5')

                (mIoU_val, mIoU_test, mIoU_unl, d_val, d_test, d_unl, history) = train_isic_ups(
                    model=student_model, base_model_to_save=base_model_to_save, filepath_h5=model_save_path,
                    modelname=modelname_i, labeled_ds=lab_ds, unlabeled_ds=unl_imgs_np, val_ds=val_ds,
                    pseudo_labels=pseudo_labels_1h, selection_mask_pos=pos_select_mask, selection_mask_neg=neg_select_mask,
                    lambda_u=UNLABELED_LOSS_WEIGHT, epochs=EPOCHS_PER_STUDENT, optimizer=optimizer,
                    h=IMAGE_HEIGHT, w=IMAGE_WIDTH, c=IMAGE_CHANNELS
                )
                
                if history: pd.DataFrame(history).to_csv(os.path.join(paths.ISIC_2018_CSV_DIR, f'history_{modelname_i}.csv'), index=False)
                student_weights_dict[modelname_i] = student_model.get_weights()
                gen_benchmarks.append((modelname_i, mIoU_val, mIoU_test, mIoU_unl, d_val, d_test, d_unl, selection_ratio))
                del student_model, base_model_to_save; gc.collect()

            if not gen_benchmarks: break
            gen_benchmarks.sort(key=lambda x: x[1], reverse=True)
            best_student_name = gen_benchmarks[0][0]
            print(f"Best student in Gen {gen_num}: {best_student_name} (val mIoU={gen_benchmarks[0][1]:.4f})")
            
            best_weights = student_weights_dict.get(best_student_name)
            if best_weights is not None:
                teacher_model.set_weights(best_weights)
            else:
                print(f"ERROR: Could not find weights for {best_student_name}. Reusing old teacher.")

            header = ['modelname', 'mIoU_val', 'mIoU_test', 'mIoU_train_unlabeled', 'dice_score_val', 'dice_score_test', 'dice_score_train_unlabeled', 'selection_ratio']
            with open(os.path.join(paths.ISIC_2018_CSV_DIR, f'results_{modelname}_gen{gen_num}.csv'), 'w', encoding='utf-8', newline='') as f:
                writer=csv.writer(f, delimiter=';'); writer.writerow(header); writer.writerows(gen_benchmarks)
            print(f"Saved benchmark results for Generation {gen_num}")