import sys, os, gc, csv, configparser
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.mixed_precision import LossScaleOptimizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unet import get_unet_unimatch_parts
from functions import train_multiclass_unimatch_parts, benchmark_multiclass, save_history_to_csv, initialize_unimatch_with_baseline
import paths
from Cityscapes_class_mapping import COLOR_TO_CLASS_MAPPING_CITYSCAPES

COLOR_TO_CLASS_MAPPING_CITYSCAPES = None 
tf.keras.mixed_precision.set_global_policy('mixed_float16')
config = configparser.ConfigParser()
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.ini')
config.read(config_dir)
CONFIDENCE_THRESHOLD = 0.95 

DATASET_LIST = ['CITYSCAPES']

with tf.device('/gpu:0'):
    for dataset_name in DATASET_LIST:
        print(f"\n====== Processing {dataset_name} ======\n")
        H = int(config[dataset_name]['IMAGE_HEIGHT'])
        W = int(config[dataset_name]['IMAGE_WIDTH'])
        C = int(config[dataset_name]['IMAGE_CHANNELS'])
        BS = 8 #int(config['DEFAULT']['BATCH_SIZE'])
        NUM_CLASSES = int(config[dataset_name]['NUM_CLASSES'])
        ALPHA = float(config[dataset_name].get('ALPHA', 1.0))
        ACTIFU = str(config[dataset_name].get('ACTIFU', 'relu'))
        
        mapping = COLOR_TO_CLASS_MAPPING_CITYSCAPES if dataset_name == 'CITYSCAPES' else COLOR_TO_CLASS_MAPPING_CITYSCAPES
        csv_dir = getattr(paths, f'{dataset_name}_CSV_DIR')
        model_dir = getattr(paths, f'{dataset_name}_MODEL_DIR')
        base_dir = getattr(paths, f'{dataset_name}_BASE_DIR')
        
        for runid in range(1, 4):
            rows = []
            baseline_path = os.path.join(model_dir, f'{dataset_name}_subset_{runid}_topK_1.h5')

            for i in range(5):
                print(f"\n--- {dataset_name} Run {runid}, Repetition {i} ---")
                tf.keras.backend.clear_session()
                modelname = f'{dataset_name}_UniMatch_{runid}_{i}'
                save_path = os.path.join(model_dir, f'{modelname}.h5')
                
                enc, dec = get_unet_unimatch_parts(H, W, C, NUM_CLASSES, ALPHA, ACTIFU, actifuout='linear')
                
                initialize_unimatch_with_baseline(enc, dec, baseline_path, H, W, C)

                opt = LossScaleOptimizer(tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4))
                
                history = train_multiclass_unimatch_parts(
                    enc, dec, save_path, modelname, H, W, C, NUM_CLASSES,
                    int(config['DEFAULT']['NUM_EPOCHS']), BS, opt,
                    getattr(paths, f'{dataset_name}_TRAIN_LABELED_IMAGES_DIR'),
                    getattr(paths, f'{dataset_name}_TRAIN_LABELED_MASKS_DIR'),
                    getattr(paths, f'{dataset_name}_TRAIN_UNLABELED_IMAGES_DIR'),
                    getattr(paths, f'{dataset_name}_VAL_IMAGES_DIR'),
                    getattr(paths, f'{dataset_name}_VAL_MASKS_DIR'),
                    dataset_name_for_aug=dataset_name,
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                )
                save_history_to_csv(history, os.path.join(csv_dir, f'history_{modelname}.csv'))
                
                best_model = tf.keras.models.load_model(save_path, compile=False)
                inp = best_model.input; out = tf.keras.layers.Softmax()(best_model.output)
                infer = tf.keras.Model(inp, out)
                
                val_pred = os.path.join(base_dir, 'val_predictions', 'unimatch', modelname)
                test_pred = os.path.join(base_dir, 'test_predictions', 'unimatch', modelname)
                unl_pred = os.path.join(base_dir, 'train_unlabeled_predictions', 'unimatch', modelname)
                
                mv, mi = benchmark_multiclass(infer, getattr(paths, f'{dataset_name}_VAL_IMAGES_DIR'), getattr(paths, f'{dataset_name}_VAL_MASKS_DIR'), val_pred, H, W, C, mapping, print_results=False)
                mt, mit = benchmark_multiclass(infer, getattr(paths, f'{dataset_name}_TEST_IMAGES_DIR'), getattr(paths, f'{dataset_name}_TEST_MASKS_DIR'), test_pred, H, W, C, mapping, print_results=False)
                try:
                    mu, miu = benchmark_multiclass(infer, getattr(paths, f'{dataset_name}_TRAIN_UNLABELED_IMAGES_DIR'), getattr(paths, f'{dataset_name}_TRAIN_UNLABELED_MASKS_DIR'), unl_pred, H, W, C, mapping, print_results=False)
                except: mu, miu = 0, 0
                
                rows.append([modelname, mv, mt, mu, mi, mit, miu, history['mask_ratio'][-1]])
                del enc, dec, best_model, infer; gc.collect()

            with open(os.path.join(csv_dir, f'results_{dataset_name}_UniMatch_{runid}.csv'), 'w', newline='') as f:
                w = csv.writer(f, delimiter=';'); w.writerow(['modelname', 'mPA_val', 'mPA_test', 'mPA_unlabeled', 'mIoU_val', 'mIoU_test', 'mIoU_unlabeled', 'final_mask_ratio']); w.writerows(rows)