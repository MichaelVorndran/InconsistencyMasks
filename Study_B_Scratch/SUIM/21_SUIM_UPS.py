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

from functions import load_labeled_data_multiclass_mask, load_unlabeled_images, train_multiclass_ups, get_uncertainty_and_predictions_mc_dropout, _ups_unlabeled_loss_singlelabel, ups_data_generator
from unet import get_unet
import paths
from SUIM_class_mapping import COLOR_TO_CLASS_MAPPING_SUIM

dataset_name = 'SUIM'
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
NUM_CLASSES    = int(dataset_cfg['NUM_CLASSES'])
ALPHA  = float(dataset_cfg['ALPHA'])
ACTIFU = str(dataset_cfg['ACTIFU'])

LR         = 0.0001 #float(default_cfg['LR'])
WD         = float(default_cfg['WD'])
BATCH_SIZE = 4# int(default_cfg['BATCH_SIZE'])
NUM_EPOCHS_FINETUNE = 50 #int(default_cfg.get('NUM_EPOCHS_FINETUNE', 5))
EPOCHS_PER_STUDENT = 50 #int(default_cfg.get('EPOCHS_UPS', 5))
MAX_GENERATIONS = 5 #int(approach_cfg.get('MAX_GENERATIONS', 10))

TAU_POS   = float(approach_cfg.get('TAU_P', 0.70))
KAPPA_POS = float(approach_cfg.get('KAPPA_P', 0.05))
TAU_NEG   = float(approach_cfg.get('TAU_N', 0.05))
KAPPA_NEG = float(approach_cfg.get('KAPPA_N', 0.005))
MC_PASSES = int(approach_cfg.get('MC_DROPOUT_PASSES', 10))
DROPOUT_RATE = float(approach_cfg.get('DROPOUT_RATE', 0.30))
UNLABELED_LOSS_WEIGHT = float(approach_cfg.get('UNLABELED_LOSS_WEIGHT', 1.0))

CONVERGENCE_THRESHOLD = float(approach_cfg.get('CONVERGENCE_THRESHOLD', 0.01))
CONVERGENCE_PATIENCE = int(approach_cfg.get('CONVERGENCE_PATIENCE', 2))
SUPERVISED_WARMUP_STEPS = 50

with tf.device('/gpu:0'):
    for runid in range(1, 4):
        modelname = f'{dataset_name}_{approach}_{runid}'
        base_model_path = os.path.join(paths.SUIM_MODEL_DIR, f'SUIM_subset_{runid}_topK_1.h5')

        lab_imgs_np, lab_masks_1h_np = load_labeled_data_multiclass_mask(paths.SUIM_TRAIN_LABELED_IMAGES_DIR, paths.SUIM_TRAIN_LABELED_MASKS_DIR, NUM_CLASSES)
        unl_imgs_np = load_unlabeled_images(paths.SUIM_TRAIN_UNLABELED_IMAGES_DIR)
        val_imgs_np, val_masks_1h_np = load_labeled_data_multiclass_mask(paths.SUIM_VAL_IMAGES_DIR, paths.SUIM_VAL_MASKS_DIR, NUM_CLASSES)
        
        lab_ids_np = np.argmax(lab_masks_1h_np, axis=-1)
        val_ids_np = np.argmax(val_masks_1h_np, axis=-1)
        
        val_ds = tf.data.Dataset.from_tensor_slices((val_imgs_np, val_ids_np)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        teacher_model = get_unet(i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS, num_outputmasks=NUM_CLASSES, alpha=ALPHA, actifu=ACTIFU, actifuout='linear', dropout_rate_encoder=DROPOUT_RATE, dropout_rate_bottleneck=DROPOUT_RATE, dropout_rate_decoder=DROPOUT_RATE)
        
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
            lab_ds_for_ft = tf.data.Dataset.from_tensor_slices((lab_imgs_np, lab_ids_np)).shuffle(len(lab_imgs_np)).batch(BATCH_SIZE)
            opt_ft = LossScaleOptimizer(tfa.optimizers.AdamW(learning_rate=LR/10.0, weight_decay=WD))
            teacher_model.compile(optimizer=opt_ft, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            teacher_model.fit(lab_ds_for_ft, validation_data=val_ds, epochs=NUM_EPOCHS_FINETUNE, verbose=2)
        
        last_num_selected_pixels = -1
        patience_counter = 0

        for gen_num in range(MAX_GENERATIONS):
            print(f"\n--- {modelname}: GENERATION {gen_num}/{MAX_GENERATIONS-1} ---")
            
            print("Generating pseudo-labels with current teacher...")
            mean_probs, uncertainty_per_class = get_uncertainty_and_predictions_mc_dropout(teacher_model, unl_imgs_np, num_passes=MC_PASSES, batch_size=8)
            
            mean_probs_np = mean_probs.numpy(); uncertainty_np = uncertainty_per_class.numpy()
            del mean_probs, uncertainty_per_class; gc.collect()

            pos_select_mask_bool = (mean_probs_np >= TAU_POS) & (uncertainty_np <= KAPPA_POS)
            neg_select_mask_bool = (mean_probs_np <= TAU_NEG) & (uncertainty_np <= KAPPA_NEG)
            
            pos_select_mask = pos_select_mask_bool.astype(np.float32)
            neg_select_mask = neg_select_mask_bool.astype(np.float32)
            pseudo_labels_1h = tf.one_hot(np.argmax(mean_probs_np, axis=-1), NUM_CLASSES, dtype=tf.float32).numpy()
            del mean_probs_np, uncertainty_np; gc.collect()

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
                    student_model.set_weights(initial_student_weights)
                
                base_model_to_save = get_unet(i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS, num_outputmasks=NUM_CLASSES, alpha=ALPHA, actifu=ACTIFU, actifuout='linear')
                optimizer = LossScaleOptimizer(tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD))
                model_save_path = os.path.join(paths.SUIM_MODEL_DIR, f'{modelname_i}_best.h5')

                (mPA_val, mPA_test, mPA_unl, mIoU_val, mIoU_test, mIoU_unl, history) = train_multiclass_ups(
                    model=student_model, base_model_to_save=base_model_to_save, filepath_h5=model_save_path,
                    modelname=modelname_i, 
                    labeled_data_np=(lab_imgs_np, lab_ids_np), 
                    unlabeled_data_np=unl_imgs_np, 
                    val_ds=val_ds,
                    pseudo_labels=pseudo_labels_1h, 
                    selection_mask_pos=pos_select_mask, 
                    selection_mask_neg=neg_select_mask,
                    lambda_u=UNLABELED_LOSS_WEIGHT, 
                    epochs=EPOCHS_PER_STUDENT, 
                    optimizer=optimizer,
                    h=IMAGE_HEIGHT, w=IMAGE_WIDTH, c=IMAGE_CHANNELS, num_classes=NUM_CLASSES,
                    class_to_color_mapping=COLOR_TO_CLASS_MAPPING_SUIM,
                    batch_size=BATCH_SIZE,
                    supervised_warmup_steps=SUPERVISED_WARMUP_STEPS,
                    val_images_dir=paths.SUIM_VAL_IMAGES_DIR, val_masks_dir=paths.SUIM_VAL_MASKS_DIR,
                    test_images_dir=paths.SUIM_TEST_IMAGES_DIR, test_masks_dir=paths.SUIM_TEST_MASKS_DIR,
                    train_unlabeled_images_dir=paths.SUIM_TRAIN_UNLABELED_IMAGES_DIR, train_unlabeled_masks_dir=paths.SUIM_TRAIN_UNLABELED_MASKS_DIR,
                    dataset_base_dir=paths.SUIM_BASE_DIR,
                    approach=approach.lower()
                )
                
                if history: pd.DataFrame(history).to_csv(os.path.join(paths.SUIM_CSV_DIR, f'history_{modelname_i}.csv'), index=False)
                student_weights_dict[modelname_i] = student_model.get_weights()
                gen_benchmarks.append((modelname_i, mPA_val, mPA_test, mPA_unl, mIoU_val, mIoU_test, mIoU_unl, selection_ratio))
                del student_model, base_model_to_save; gc.collect()

            if not gen_benchmarks: break
            gen_benchmarks.sort(key=lambda x: x[4], reverse=True)
            best_student_name = gen_benchmarks[0][0]
            print(f"Best student in Gen {gen_num}: {best_student_name} (val mIoU={gen_benchmarks[0][4]:.4f})")
            
            best_weights = student_weights_dict.get(best_student_name)
            if best_weights is not None:
                teacher_model.set_weights(best_weights)
            else:
                print(f"ERROR: Could not find weights for {best_student_name}. Reusing old teacher.")

            header = ['modelname', 'mPA_val', 'mPA_test', 'mPA_train_unlabeled', 'mIoU_val', 'mIoU_test', 'mIoU_train_unlabeled', 'selection_ratio']
            with open(os.path.join(paths.SUIM_CSV_DIR, f'results_{modelname}_gen{gen_num}.csv'), 'w', encoding='utf-8', newline='') as f:
                writer=csv.writer(f, delimiter=';'); writer.writerow(header); writer.writerows(gen_benchmarks)
            print(f"Saved benchmark results for Generation {gen_num}")