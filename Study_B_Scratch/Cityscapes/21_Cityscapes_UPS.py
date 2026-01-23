import sys
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.mixed_precision import LossScaleOptimizer
import configparser
import gc
import csv
import glob
import shutil
import numpy as np
import pandas as pd
import tensorflow_addons as tfa

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import (
    ImagePathLoader,
    get_uncertainty_for_chunk,
    load_labeled_data_multiclass_mask,
    load_unlabeled_images,
    get_uncertainty_and_predictions_mc_dropout_mem_efficient, # <-- MEMORY-EFFICIENT
    train_multiclass_ups_mem_eff,                          # <-- MEMORY-EFFICIENT
    _ups_unlabeled_loss_singlelabel,
    ups_data_generator
)
from unet import get_unet
import paths
from Cityscapes_class_mapping import COLOR_TO_CLASS_MAPPING_CITYSCAPES # <-- Assumes you have this file


dataset_name = 'Cityscapes'
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
    for runid in range(1, 2):
        modelname = f'{dataset_name}_{approach}_{runid}'
        base_model_path = os.path.join(paths.CITYSCAPES_MODEL_DIR, f'Cityscapes_subset_{runid}_topK_1.h5')

        print("Loading LABELED and VALIDATION data into RAM (these are small sets)...")
        lab_imgs_np, lab_masks_1h_np = load_labeled_data_multiclass_mask(paths.CITYSCAPES_TRAIN_LABELED_IMAGES_DIR, paths.CITYSCAPES_TRAIN_LABELED_MASKS_DIR, NUM_CLASSES)
        val_imgs_np, val_masks_1h_np = load_labeled_data_multiclass_mask(paths.CITYSCAPES_VAL_IMAGES_DIR, paths.CITYSCAPES_VAL_MASKS_DIR, NUM_CLASSES)
        
        lab_ids_np = np.argmax(lab_masks_1h_np, axis=-1)
        val_ids_np = np.argmax(val_masks_1h_np, axis=-1)
        del lab_masks_1h_np, val_masks_1h_np; gc.collect()

        print("Getting file paths for unlabeled data (NOT loading images into RAM)...")
        unlabeled_image_paths = sorted(glob.glob(os.path.join(paths.CITYSCAPES_TRAIN_UNLABELED_IMAGES_DIR, '*.png')))
        num_unlabeled = len(unlabeled_image_paths)

        labeled_data_np_for_gen = (lab_imgs_np, lab_ids_np)
        val_ds = tf.data.Dataset.from_tensor_slices((val_imgs_np, val_ids_np)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        teacher_model = get_unet(i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS, num_outputmasks=NUM_CLASSES, alpha=ALPHA, actifu=ACTIFU, actifuout='linear', dropout_rate_encoder=DROPOUT_RATE, dropout_rate_bottleneck=DROPOUT_RATE, dropout_rate_decoder=DROPOUT_RATE)
        
        initial_student_weights = None
        if os.path.exists(base_model_path):
            try:
                teacher_model.load_weights(base_model_path, by_name=True)
                initial_student_weights = teacher_model.get_weights()
                print("Teacher and student initial states set from pre-trained model.")
            except Exception as e:
                print(f"Error loading base model: {e}. Starting from scratch.")
                continue # Skip run if base model fails to load
        else:
            print(f"Base model not found at {base_model_path}. Starting from scratch is not recommended. Skipping run.")
            continue

        if NUM_EPOCHS_FINETUNE > 0:
            print(f"Fine-tuning teacher for {NUM_EPOCHS_FINETUNE} epochs...")
            lab_ds_for_ft = tf.data.Dataset.from_tensor_slices((lab_imgs_np, lab_ids_np)).shuffle(len(lab_imgs_np)).batch(BATCH_SIZE)
            opt_ft = LossScaleOptimizer(tfa.optimizers.AdamW(learning_rate=LR/10.0, weight_decay=WD))
            teacher_model.compile(optimizer=opt_ft, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            teacher_model.fit(lab_ds_for_ft, validation_data=val_ds, epochs=NUM_EPOCHS_FINETUNE, verbose=2)
        
        for gen_num in range(MAX_GENERATIONS):
            print(f"\n--- {modelname}: GENERATION {gen_num}/{MAX_GENERATIONS-1} ---")
            
            # === STAGE 1: CHUNKED PSEUDO-LABEL GENERATION ===
            gen_temp_dir = os.path.join(paths.CITYSCAPES_BASE_DIR, 'temp_chunks', f'{modelname}_gen{gen_num}')
            os.makedirs(gen_temp_dir, exist_ok=True)
            
            CHUNK_SIZE = 250  # Process 1000 images at a time. Adjust based on your RAM.
            num_chunks = int(np.ceil(num_unlabeled / CHUNK_SIZE))
            print(f"Generating pseudo-labels in {num_chunks} chunks of size {CHUNK_SIZE}...")
            
            for i in range(num_chunks):
                print(f"  Processing Chunk {i+1}/{num_chunks}...")
                start_idx = i * CHUNK_SIZE
                end_idx = min((i + 1) * CHUNK_SIZE, num_unlabeled)
                chunk_paths = unlabeled_image_paths[start_idx:end_idx]
                
                chunk_loader = ImagePathLoader(chunk_paths, batch_size=BATCH_SIZE, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
                mean_probs_chunk, uncertainty_chunk = get_uncertainty_for_chunk(teacher_model, chunk_loader, num_passes=MC_PASSES)
                
                np.save(os.path.join(gen_temp_dir, f'mean_probs_chunk_{i}.npy'), mean_probs_chunk)
                np.save(os.path.join(gen_temp_dir, f'uncertainty_chunk_{i}.npy'), uncertainty_chunk)
                del mean_probs_chunk, uncertainty_chunk, chunk_loader; gc.collect()

            # === STAGE 2: ASSEMBLY USING MEMORY-MAPPED FILES (NO RAM USED) ===
            print("\nAssembling full pseudo-labels using memory-mapped arrays on disk...")
            mmap_shape = (num_unlabeled, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES)
            
            pseudo_labels_mmap = np.memmap(os.path.join(gen_temp_dir, 'pseudo_labels.mmap'), dtype=np.float32, mode='w+', shape=mmap_shape)
            pos_mask_mmap = np.memmap(os.path.join(gen_temp_dir, 'pos_mask.mmap'), dtype=np.float32, mode='w+', shape=mmap_shape)
            neg_mask_mmap = np.memmap(os.path.join(gen_temp_dir, 'neg_mask.mmap'), dtype=np.float32, mode='w+', shape=mmap_shape)

            for i in range(num_chunks):
                mean_probs_chunk = np.load(os.path.join(gen_temp_dir, f'mean_probs_chunk_{i}.npy'))
                uncertainty_chunk = np.load(os.path.join(gen_temp_dir, f'uncertainty_chunk_{i}.npy'))
                start_idx = i * CHUNK_SIZE
                end_idx = start_idx + len(mean_probs_chunk)

                pos_mask_mmap[start_idx:end_idx] = ((mean_probs_chunk >= TAU_POS) & (uncertainty_chunk <= KAPPA_POS))
                neg_mask_mmap[start_idx:end_idx] = ((mean_probs_chunk <= TAU_NEG) & (uncertainty_chunk <= KAPPA_NEG))
                
                pseudo_labels_ids = np.argmax(mean_probs_chunk, axis=-1)
                pseudo_labels_mmap[start_idx:end_idx] = tf.one_hot(pseudo_labels_ids, NUM_CLASSES, dtype=tf.float32).numpy()
                del mean_probs_chunk, uncertainty_chunk; gc.collect()

            print("Calculating selection ratio...")
            num_selected_pixels = np.sum(pos_mask_mmap) + np.sum(neg_mask_mmap)
            total_possible_pixels = np.prod(mmap_shape, dtype=np.int64)
            selection_ratio = num_selected_pixels / total_possible_pixels if total_possible_pixels > 0 else 0
            print(f"Total selected pseudo-entries: {int(num_selected_pixels)} ({selection_ratio:.2%})")

            # === STAGE 3: STUDENT TRAINING USING PATH-BASED GENERATOR ===
            # Note: We now load the unlabeled images for training. If these ALSO don't fit,
            # you will need a more advanced path-based generator for training.
            # But let's assume they fit for now, as it's simpler.
            print("Loading unlabeled images into memory for the training generator...")
            unl_imgs_np = load_unlabeled_images(paths.CITYSCAPES_TRAIN_UNLABELED_IMAGES_DIR)

            gen_benchmarks, student_weights_dict = [], {}
            for i in range(5):
                print(f"\n--- Training Student {i+1}/5 for Generation {gen_num} ---")
                tf.keras.backend.clear_session()
                modelname_i = f'{modelname}_gen{gen_num}_student{i}'

                student_model = get_unet(i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS, num_outputmasks=NUM_CLASSES, alpha=ALPHA, actifu=ACTIFU, actifuout='linear', dropout_rate_encoder=DROPOUT_RATE, dropout_rate_bottleneck=DROPOUT_RATE, dropout_rate_decoder=DROPOUT_RATE)
                if initial_student_weights: student_model.set_weights(initial_student_weights)
                
                base_model_to_save = get_unet(i_height=IMAGE_HEIGHT, i_width=IMAGE_WIDTH, i_channels=IMAGE_CHANNELS, num_outputmasks=NUM_CLASSES, alpha=ALPHA, actifu=ACTIFU, actifuout='linear')
                optimizer = LossScaleOptimizer(tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD))
                model_save_path = os.path.join(paths.CITYSCAPES_MODEL_DIR, f'{modelname_i}_best.h5')

                # Here we call your ORIGINAL generator-based training function.
                # It receives the memmap arrays, which behave like numpy arrays.
                (mPA_val, mPA_test, mPA_unl, mIoU_val, mIoU_test, mIoU_unl, history) = train_multiclass_ups_mem_eff(
                    model=student_model, base_model_to_save=base_model_to_save, filepath_h5=model_save_path, modelname=modelname_i, 
                    labeled_data_np=labeled_data_np_for_gen, 
                    unlabeled_imgs_np=unl_imgs_np,
                    val_ds=val_ds,
                    pseudo_labels=pseudo_labels_mmap, # Pass the memmap array
                    selection_mask_pos=pos_mask_mmap, # Pass the memmap array
                    selection_mask_neg=neg_mask_mmap, # Pass the memmap array
                    lambda_u=UNLABELED_LOSS_WEIGHT, epochs=EPOCHS_PER_STUDENT, optimizer=optimizer,
                    h=IMAGE_HEIGHT, w=IMAGE_WIDTH, c=IMAGE_CHANNELS, num_classes=NUM_CLASSES,
                    class_to_color_mapping=COLOR_TO_CLASS_MAPPING_CITYSCAPES, batch_size=BATCH_SIZE,
                    val_images_dir=paths.CITYSCAPES_VAL_IMAGES_DIR, val_masks_dir=paths.CITYSCAPES_VAL_MASKS_DIR,
                    test_images_dir=paths.CITYSCAPES_TEST_IMAGES_DIR, test_masks_dir=paths.CITYSCAPES_TEST_MASKS_DIR,
                    train_unlabeled_images_dir=paths.CITYSCAPES_TRAIN_UNLABELED_IMAGES_DIR, train_unlabeled_masks_dir=paths.CITYSCAPES_TRAIN_UNLABELED_MASKS_DIR,
                    dataset_base_dir=paths.CITYSCAPES_BASE_DIR, approach=approach.lower()
                )
                
                if history: pd.DataFrame(history).to_csv(os.path.join(paths.CITYSCAPES_CSV_DIR, f'history_{modelname_i}.csv'), index=False)
                student_weights_dict[modelname_i] = student_model.get_weights()
                gen_benchmarks.append((modelname_i, mPA_val, mPA_test, mPA_unl, mIoU_val, mIoU_test, mIoU_unl, selection_ratio))
                del student_model, base_model_to_save; gc.collect()
            
            # === STAGE 4: CLEANUP FOR THE GENERATION ===
            print("Cleaning up generation temporary files...")
            del unl_imgs_np, pseudo_labels_mmap, pos_mask_mmap, neg_mask_mmap
            gc.collect() # Make sure file handles are released
            if os.path.exists(gen_temp_dir):
                shutil.rmtree(gen_temp_dir)

            if not gen_benchmarks:
                print("No students were trained, stopping.")
                break

            gen_benchmarks.sort(key=lambda x: x[4], reverse=True) # Sort by val mIoU
            best_student_name = gen_benchmarks[0][0]
            print(f"Best student in Gen {gen_num}: {best_student_name} (val mIoU={gen_benchmarks[0][4]:.4f})")
            
            best_weights = student_weights_dict.get(best_student_name)
            if best_weights is not None:
                teacher_model.set_weights(best_weights)
                initial_student_weights = best_weights
            else:
                print(f"ERROR: Could not find weights for {best_student_name}. Teacher remains unchanged.")

            header = ['modelname', 'mPA_val', 'mPA_test', 'mPA_train_unlabeled', 'mIoU_val', 'mIoU_test', 'mIoU_train_unlabeled', 'selection_ratio']
            with open(os.path.join(paths.CITYSCAPES_CSV_DIR, f'results_{modelname}_gen{gen_num}.csv'), 'w', newline='') as f:
                writer=csv.writer(f); writer.writerow(header); writer.writerows(gen_benchmarks)
            print(f"Saved benchmark results for Generation {gen_num}")