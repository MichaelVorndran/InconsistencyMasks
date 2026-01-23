import sys, os, gc, csv, configparser
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.mixed_precision import LossScaleOptimizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unet import get_unet_unimatch_parts
from functions import train_hela_unimatch_parts, benchmark_hela, save_history_to_csv, initialize_unimatch_with_baseline
import paths

tf.keras.mixed_precision.set_global_policy('mixed_float16')
config = configparser.ConfigParser()
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.ini')
config.read(config_dir)

dataset_name = 'HELA'
H = int(config[dataset_name]['IMAGE_HEIGHT'])
W = int(config[dataset_name]['IMAGE_WIDTH'])
C = int(config[dataset_name]['IMAGE_CHANNELS'])
NUM_CLASSES = 3 
ALPHA = float(config[dataset_name].get('ALPHA', 1.0))
ACTIFU = str(config[dataset_name].get('ACTIFU', 'relu'))

with tf.device('/gpu:0'):
    for runid in range(1, 4):
        rows = []
        baseline_path = os.path.join(paths.HELA_MODEL_DIR, f'HELA_subset_{runid}_topK_1.h5')

        for i in range(5):
            print(f"\n--- Run {runid}, Repetition {i} ---")
            tf.keras.backend.clear_session()
            modelname = f'HeLa_UniMatch_{runid}_{i}'
            save_path = os.path.join(paths.HELA_MODEL_DIR, f'{modelname}.h5')
            
            # HeLa uses Sigmoid
            enc, dec = get_unet_unimatch_parts(H, W, C, NUM_CLASSES, ALPHA, ACTIFU, actifuout='sigmoid')
            
            # Warm Start
            initialize_unimatch_with_baseline(enc, dec, baseline_path, H, W, C)

            opt = LossScaleOptimizer(tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4))
            
            history = train_hela_unimatch_parts(enc, dec, save_path, modelname, H, W, C, int(config['DEFAULT']['NUM_EPOCHS']), int(config['DEFAULT']['BATCH_SIZE']), opt)
            save_history_to_csv(history, os.path.join(paths.HELA_CSV_DIR, f'history_{modelname}.csv'))
            
            best_model = tf.keras.models.load_model(save_path, compile=False)
            val_dir = os.path.join(paths.HELA_BASE_DIR, 'val_predictions', 'unimatch', modelname)
            test_dir = os.path.join(paths.HELA_BASE_DIR, 'test_predictions', 'unimatch', modelname)
            unl_dir = os.path.join(paths.HELA_BASE_DIR, 'train_unlabeled_predictions', 'unimatch', modelname)
            
            rv = benchmark_hela(best_model, paths.HELA_VAL_DIR, val_dir, H, W, C)
            rt = benchmark_hela(best_model, paths.HELA_TEST_DIR, test_dir, H, W, C)
            ru = benchmark_hela(best_model, paths.HELA_TRAIN_UNLABELED_DIR, unl_dir, H, W, C)
            
            rows.append([modelname, rv[0], rt[0], ru[0], rv[1], rt[1], ru[1], rv[2], rt[2], ru[2], history['mask_ratio'][-1]])
            del enc, dec, best_model; gc.collect()

        with open(os.path.join(paths.HELA_CSV_DIR, f'results_HeLa_UniMatch_{runid}.csv'), 'w', newline='') as f:
            w = csv.writer(f, delimiter=';'); w.writerow(['modelname', 'mIoU_val', 'mIoU_test', 'mIoU_unlabeled', 'mIoU_ad_val', 'mIoU_ad_test', 'mIoU_ad_unlabeled', 'mcce_val', 'mcce_test', 'mcce_unlabeled', 'final_mask_ratio']); w.writerows(rows)