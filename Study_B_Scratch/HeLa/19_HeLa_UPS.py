# 25_HeLa_UPS.py

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

from functions import load_images_and_masks_hela, load_unlabeled_images, train_hela_ups, get_uncertainty_and_predictions_mc_dropout_sigmoid
from unet import get_unet
import paths

# --- CONFIGS ---
dataset_name = 'HeLa'
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
NUM_OUTPUTS    = 3
ALPHA  = float(dataset_cfg['ALPHA'])
ACTIFU = str(dataset_cfg['ACTIFU'])

LR         = 0.0001 #float(default_cfg['LR'])
WD         = float(default_cfg['WD'])
BATCH_SIZE = int(default_cfg['BATCH_SIZE'])
NUM_EPOCHS_FINETUNE = 50 #int(default_cfg.get('NUM_EPOCHS_FINETUNE', 5))
EPOCHS_PER_STUDENT = 25 #int(default_cfg.get('EPOCHS_UPS', 2))
MAX_GENERATIONS = 5 #int(approach_cfg.get('MAX_GENERATIONS', 5))


TAU_POS   = 0.99 # float(approach_cfg.get('TAU_P_MULTILABEL', 0.5))
KAPPA_POS = float(approach_cfg.get('KAPPA_P_MULTILABEL', 0.05))
TAU_NEG   = float(approach_cfg.get('TAU_N_MULTILABEL', 0.5))
KAPPA_NEG = float(approach_cfg.get('KAPPA_N_MULTILABEL', 0.05))

MC_PASSES = int(approach_cfg.get('MC_DROPOUT_PASSES', 10))
DROPOUT_RATE = float(approach_cfg.get('DROPOUT_RATE', 0.30))
UNLABELED_LOSS_WEIGHT = 0.1 #float(approach_cfg.get('UNLABELED_LOSS_WEIGHT', 1.0))
CHANNEL_WEIGHTS = (1.0, 1.0, 3.0)

CONVERGENCE_THRESHOLD = float(approach_cfg.get('CONVERGENCE_THRESHOLD', 0.01))
CONVERGENCE_PATIENCE = int(approach_cfg.get('CONVERGENCE_PATIENCE', 2))
SUPERVISED_WARMUP_STEPS = 50

# --- Main Execution ---
with tf.device('/gpu:0'):
    for runid in range(1, 4):
        modelname = f'{dataset_name}_{approach}_{runid}'
        base_model_path = os.path.join(paths.HELA_MODEL_DIR, f'HeLa_subset_{runid}_topK_1.h5')

        l_imgs_np, lA, lD, lP = load_images_and_masks_hela(paths.HELA_TRAIN_LABELED_DIR)
        lab_masks_np = np.stack([lA, lD, lP], axis=-1).astype(np.float32)
        lab_imgs_np = np.expand_dims(l_imgs_np, axis=-1)
        
        unl_imgs_np = load_unlabeled_images(paths.HELA_TRAIN_UNLABELED_BRIGHTFIELD_DIR, read_as_grayscale=True)
        unl_imgs_np = np.expand_dims(unl_imgs_np, axis=-1)
        
        v_imgs_np, vA, vD, vP = load_images_and_masks_hela(paths.HELA_VAL_DIR)
        val_masks_np = np.stack([vA, vD, vP], axis=-1).astype(np.float32)
        v_imgs_np = np.expand_dims(v_imgs_np, axis=-1)
        
        val_ds = tf.data.Dataset.from_tensor_slices((v_imgs_np, val_masks_np)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        teacher_model = get_unet(i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS, num_outputmasks=NUM_OUTPUTS, alpha=ALPHA, actifu=ACTIFU, actifuout='sigmoid', dropout_rate_encoder=DROPOUT_RATE, dropout_rate_bottleneck=DROPOUT_RATE, dropout_rate_decoder=DROPOUT_RATE)
        
        initial_student_weights = None
        if os.path.exists(base_model_path):
            try:
                base_model = tf.keras.models.load_model(base_model_path)
                initial_student_weights = base_model.get_weights()
                teacher_model.set_weights(initial_student_weights)
                print("Teacher and student initial states set from pre-trained model.")
            except Exception as e:
                print(f"Error loading base model: {e}. Starting from scratch.")
        else:
            print(f"Base model not found at {base_model_path}. Starting from scratch.")

        if NUM_EPOCHS_FINETUNE > 0:
            print(f"Fine-tuning teacher for {NUM_EPOCHS_FINETUNE} epochs to adapt dropout...")
            lab_ds_for_ft = tf.data.Dataset.from_tensor_slices((lab_imgs_np, lab_masks_np)).shuffle(len(lab_imgs_np)).batch(BATCH_SIZE)
            opt_ft = LossScaleOptimizer(tfa.optimizers.AdamW(learning_rate=LR/10.0, weight_decay=WD))
            teacher_model.compile(optimizer=opt_ft, loss='binary_crossentropy')
            teacher_model.fit(lab_ds_for_ft, validation_data=val_ds, epochs=NUM_EPOCHS_FINETUNE, verbose=2)
        
        last_num_selected_pixels = -1
        patience_counter = 0

        for gen_num in range(MAX_GENERATIONS):
            print(f"\n--- {modelname}: GENERATION {gen_num}/{MAX_GENERATIONS-1} ---")
            
            print("Generating pseudo-labels with current teacher...")
            mean_probs, uncertainty_per_channel = get_uncertainty_and_predictions_mc_dropout_sigmoid(teacher_model, unl_imgs_np, num_passes=MC_PASSES, batch_size=8)

            mean_probs_np = mean_probs.numpy(); uncertainty_np = uncertainty_per_channel.numpy()
            del mean_probs, uncertainty_per_channel; gc.collect()

            pos_select_mask_bool = (mean_probs_np >= TAU_POS) & (uncertainty_np <= KAPPA_POS)
            neg_select_mask_bool = (mean_probs_np < TAU_NEG) & (uncertainty_np <= KAPPA_NEG)
            selection_mask = (pos_select_mask_bool | neg_select_mask_bool).astype(np.float32)
            
            num_selected_pixels = np.sum(selection_mask)
            total_possible_pixels = np.prod(unl_imgs_np.shape[:-1]) * NUM_OUTPUTS
            selection_ratio = num_selected_pixels / total_possible_pixels if total_possible_pixels > 0 else 0
            print(f"Total selected pseudo-entries: {int(num_selected_pixels)} ({selection_ratio:.2%})")

            #if last_num_selected_pixels > 0:
            #    change_ratio = abs(num_selected_pixels - last_num_selected_pixels) / last_num_selected_pixels
            #    print(f"Change in selected pixels: {change_ratio:.4f}")
            #    if change_ratio < CONVERGENCE_THRESHOLD: patience_counter += 1
            #    else: patience_counter = 0
            #    if patience_counter >= CONVERGENCE_PATIENCE:
            #        print(f"Convergence reached after {gen_num} generations. Stopping.")
            #        break
            #last_num_selected_pixels = num_selected_pixels
            
            gen_benchmarks = []; student_weights_dict = {}

            for i in range(5):
                print(f"\n--- Training Student {i+1}/5 for Generation {gen_num} ---")
                tf.keras.backend.clear_session()
                modelname_i = f'{modelname}_gen{gen_num}_{i}'

                student_model = get_unet(i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS, num_outputmasks=NUM_OUTPUTS, alpha=ALPHA, actifu=ACTIFU, actifuout='sigmoid', dropout_rate_encoder=DROPOUT_RATE, dropout_rate_bottleneck=DROPOUT_RATE, dropout_rate_decoder=DROPOUT_RATE)
                if initial_student_weights:
                    student_model.set_weights(initial_student_weights)
                
                base_model_to_save = get_unet(i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS, num_outputmasks=NUM_OUTPUTS, alpha=ALPHA, actifu=ACTIFU, actifuout='sigmoid')
                optimizer = LossScaleOptimizer(tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD))
                model_save_path = os.path.join(paths.HELA_MODEL_DIR, f'{modelname_i}_best.h5')

                (mIoU_val, mIoU_test, mIoU_unl,
                 mIoU_ad_val, mIoU_ad_test, mIoU_ad_unl,
                 mcce_val, mcce_test, mcce_unl, history) = train_hela_ups(
                    model=student_model, base_model_to_save=base_model_to_save, filepath_h5=model_save_path,
                    modelname=modelname_i, 
                    labeled_data_np=(lab_imgs_np, lab_masks_np), 
                    unlabeled_data_np=unl_imgs_np,
                    val_ds=val_ds,
                    pseudo_labels=mean_probs_np, 
                    selection_mask=selection_mask,
                    lambda_u=UNLABELED_LOSS_WEIGHT, 
                    epochs=EPOCHS_PER_STUDENT, 
                    optimizer=optimizer,
                    h=IMAGE_HEIGHT, w=IMAGE_WIDTH, c=IMAGE_CHANNELS,
                    batch_size=BATCH_SIZE,
                    channel_weights=CHANNEL_WEIGHTS,
                    supervised_warmup_steps=SUPERVISED_WARMUP_STEPS,
                    approach=approach.lower()
                )
                
                if history: pd.DataFrame(history).to_csv(os.path.join(paths.HELA_CSV_DIR, f'history_{modelname_i}.csv'), index=False)
                student_weights_dict[modelname_i] = student_model.get_weights()
                gen_benchmarks.append((modelname_i, mIoU_val, mIoU_test, mIoU_unl, mIoU_ad_val, mIoU_ad_test, mIoU_ad_unl, mcce_val, mcce_test, mcce_unl, selection_ratio))
                del student_model, base_model_to_save; gc.collect()

            if not gen_benchmarks: break

            gen_benchmarks.sort(key=lambda x: x[7], reverse=False) 
            best_student_name = gen_benchmarks[0][0]
            print(f"Best student in Gen {gen_num}: {best_student_name} (val MCCE={gen_benchmarks[0][7]:.4f})")

            #gen_benchmarks.sort(key=lambda x: x[1], reverse=True) # Sort by val mIoU
            #best_student_name = gen_benchmarks[0][0]
            #print(f"Best student in Gen {gen_num}: {best_student_name} (val mIoU={gen_benchmarks[0][1]:.4f})")
            
            best_weights = student_weights_dict.get(best_student_name)
            if best_weights is not None:
                teacher_model.set_weights(best_weights)
            else:
                print(f"ERROR: Could not find weights for {best_student_name}. Reusing old teacher.")

            header = ['modelname', 'mIoU_val','mIoU_test','mIoU_train_unlabeled', 'mIoU_AD_val','mIoU_AD_test','mIoU_AD_train_unlabeled', 'MCCE_val','MCCE_test','MCCE_train_unlabeled', 'selection_ratio']
            with open(os.path.join(paths.HELA_CSV_DIR, f'results_{modelname}_gen{gen_num}.csv'), 'w', encoding='utf-8', newline='') as f:
                writer=csv.writer(f, delimiter=';'); writer.writerow(header); writer.writerows(gen_benchmarks)
            print(f"Saved benchmark results for Generation {gen_num}")