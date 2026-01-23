import sys, os, gc, csv, configparser
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.mixed_precision import LossScaleOptimizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unet import get_unet_unimatch_parts
from functions import train_isic_unimatch_parts, benchmark_ISIC2018, save_history_to_csv, initialize_unimatch_with_baseline
import paths

tf.keras.mixed_precision.set_global_policy('mixed_float16')
config = configparser.ConfigParser()
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.ini')
config.read(config_dir)

dataset_name = 'ISIC_2018'
H = int(config[dataset_name]['IMAGE_HEIGHT'])
W = int(config[dataset_name]['IMAGE_WIDTH'])
C = int(config[dataset_name]['IMAGE_CHANNELS'])
NUM_CLASSES = 2 # FORCE 2 CLASSES for SCCE
ALPHA = float(config[dataset_name].get('ALPHA', 1.0))
ACTIFU = str(config[dataset_name].get('ACTIFU', 'relu'))

with tf.device('/gpu:0'):
    for runid in range(1, 4):
        rows = []
        # Define baseline path (Best supervised model from this run)
        baseline_path = os.path.join(paths.ISIC_2018_MODEL_DIR, f'ISIC_2018_subset_{runid}_topK_1.h5')
        
        for i in range(5):
            print(f"\n--- Run {runid}, Repetition {i} ---")
            tf.keras.backend.clear_session()
            modelname = f'ISIC_2018_UniMatch_{runid}_{i}'
            save_path = os.path.join(paths.ISIC_2018_MODEL_DIR, f'{modelname}.h5')
            
            # 1. Build Parts
            enc, dec = get_unet_unimatch_parts(H, W, C, NUM_CLASSES, ALPHA, ACTIFU, actifuout='linear')
            
            # 2. WARM START
            initialize_unimatch_with_baseline(enc, dec, baseline_path, H, W, C)
            
            opt = LossScaleOptimizer(tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4))
            
            history = train_isic_unimatch_parts(enc, dec, save_path, modelname, H, W, C, int(config['DEFAULT']['NUM_EPOCHS']), int(config['DEFAULT']['BATCH_SIZE']), opt)
            save_history_to_csv(history, os.path.join(paths.ISIC_2018_CSV_DIR, f'history_{modelname}.csv'))
            
            # Benchmark
            best_model = tf.keras.models.load_model(save_path, compile=False)
            inp = best_model.input; out = tf.keras.layers.Softmax()(best_model.output)[..., 1:]
            infer = tf.keras.Model(inp, out)
            
            val_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'val_predictions', 'unimatch', modelname)
            test_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'test_predictions', 'unimatch', modelname)
            
            mv, dv = benchmark_ISIC2018(infer, paths.ISIC_2018_VAL_IMAGES_DIR, paths.ISIC_2018_VAL_MASKS_DIR, val_dir, H, W, C, print_results=False)
            mt, dt = benchmark_ISIC2018(infer, paths.ISIC_2018_TEST_IMAGES_DIR, paths.ISIC_2018_TEST_MASKS_DIR, test_dir, H, W, C, print_results=False)
            rows.append([modelname, mv, mt, dv, dt, history['mask_ratio'][-1]])
            
            del enc, dec, best_model, infer; gc.collect()

        with open(os.path.join(paths.ISIC_2018_CSV_DIR, f'results_ISIC_2018_UniMatch_{runid}.csv'), 'w', newline='') as f:
            w = csv.writer(f, delimiter=';'); w.writerow(['modelname', 'mIoU_val', 'mIoU_test', 'dice_val', 'dice_test', 'mask_ratio']); w.writerows(rows)