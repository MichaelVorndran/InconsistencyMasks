
import sys
import os
import numpy as np
import cv2
import io
import contextlib
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow.keras import Model
import configparser
import random
import shutil
import gc
import csv
import math
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from unet import get_unet
import glob
from collections import deque
from typing import Tuple
import paths
from tensorflow.keras.utils import Sequence



CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_SCRIPT_DIR)

config_path = os.path.join(CURRENT_SCRIPT_DIR, 'config.ini')

config = configparser.ConfigParser()

if not os.path.exists(config_path):
    print(f"Error: Config file not found at {config_path}")
else:
    config.read(config_path)


SEED = int(config['DEFAULT']['SEED'])
BATCH_SIZE = int(config['DEFAULT']['BATCH_SIZE'])
LR = float(config['DEFAULT']['LR'])
WD = float(config['DEFAULT']['WD'])
THRESHOLD = float(config['DEFAULT']['THRESHOLD'])
NUM_EPOCHS = int(config['DEFAULT']['NUM_EPOCHS'])
NUM_EPOCHS_CS = int(config['DEFAULT']['NUM_EPOCHS_CS'])


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def delta_metric(threshold=1.25):
    def metric(y_true, y_pred):
        ratio_1 = y_pred / y_true
        ratio_2 = y_true / y_pred

        max_ratio = tf.maximum(ratio_1, ratio_2)
        return tf.reduce_mean(tf.cast(max_ratio < threshold, tf.float32))
    
    metric.__name__ = f'delta_{threshold}'
    return metric


class MeanIoU(tf.keras.metrics.Metric):
    '''
    This class represents a Mean Intersection over Union (IoU) metric for evaluating segmentation models.

    The Mean IoU is a common evaluation metric for semantic image segmentation, which calculates the mean of IoU scores over multiple classes.

    Attributes:
        num_classes (int): The number of classes in the segmentation task.
    
    Methods:
        compute_iou: Computes the IoU between the ground truth and predicted masks for a single class.
        update_state: Updates the state of the metric with new predictions and ground truths.
        result: Returns the current value of the metric.
        reset_state: Resets all state variables of the metric.
        get_config: Returns the configuration of the metric.
        from_config: Creates a new instance from the given configuration.
    '''
    
    def __init__(self, num_classes, **kwargs):
        super(MeanIoU, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.total_iou = self.add_weight(name="total_iou", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    
    def compute_iou(self, y_true, y_pred):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        return intersection / union

    def update_state(self, y_true, y_pred, sample_weight=None):
        iou_values = [self.compute_iou(y_true[:,:,:,k], y_pred[:,:,:,k]) for k in range(self.num_classes)]
        mean_iou_val = tf.reduce_mean(iou_values)
        self.total_iou.assign_add(mean_iou_val)
        self.count.assign_add(1.0)

    def result(self):
        return self.total_iou / self.count

    def reset_state(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super(MeanIoU, self).get_config()
        config.update({'num_classes': self.num_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ignore_im_categorical_crossentropy(tf.keras.losses.Loss):
    '''
    This class is a custom loss function for categorical cross-entropy that ignores the inconsistency mask.

    Methods:
        call: Calculates the loss, ignoring the IM class.
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        loss = tf.cast(self.cce(y_true, y_pred), tf.float32)
        mask = tf.cast(1. - y_true[..., 0], tf.float32)
        loss *= mask
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-7)



class ignore_im_dice_loss_multiclass(tf.keras.losses.Loss):
    '''
    This class is a custom loss function that ignores the contribution of the inconsistency mask.
    
    Assumes that the segmentation masks are given in one-hot encoded format.
    
    Args:
        y_true: Tensor of true masks.
        y_pred: Tensor of predicted masks.
    
    Returns:
        Tensor of the computed Dice loss.
    '''

    def call(self, y_true, y_pred):
        # Remove contribution of class 0
        y_true = y_true[:, :, 1:]
        y_pred = y_pred[:, :, 1:]

        # Intersection of the two sets
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        
        # Size of each set
        size_true = tf.reduce_sum(y_true, axis=[1, 2])
        size_pred = tf.reduce_sum(y_pred, axis=[1, 2])

        # Compute Dice loss
        dice_coefficient = (2. * intersection + 1e-7) / (size_true + size_pred + 1e-7)
        dice_loss = 1 - dice_coefficient

        return tf.reduce_mean(dice_loss)



def dice_loss(y_true, y_pred, smooth=1):
    '''
    Compute the Dice Loss for binary segmentation tasks.
    
    Args:
        y_true (tf.Tensor): Ground truth segmentation masks, shape (batch_size, height, width, 1).
        y_pred (tf.Tensor): Predicted segmentation masks, shape (batch_size, height, width, 1).
        smooth (float, optional): Smoothing factor to prevent division by zero. Defaults to 1.
    
    Returns:
        tf.Tensor: Dice loss value.
    '''

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])

    dice_coeff = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - tf.reduce_mean(dice_coeff)

    return dice_loss




def train_ISIC_2018(train_images_dir, 
                        val_images_dir, 
                        val_masks_dir, 
                        test_images_dir, 
                        test_masks_dir, 
                        unlabeled_images_dir, 
                        unlabeled_masks_dir, 
                        modelname, 
                        filepath_h5, 
                        model, 
                        loss_func, 
                        steps_per_epoch, 
                        h, w, c,
                        val_pred_dir, 
                        test_pred_dir, 
                        unlabeled_pred_dir, 
                        print_results=False):
    
    train_images_path = os.path.join(train_images_dir, '*.png')
    train_dataset = tf.data.Dataset.list_files(train_images_path, seed=SEED)
    train_dataset = train_dataset.map(parse_image_ISIC_2018).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    
    val_images_path = os.path.join(val_images_dir, '*.png')
    val_dataset = tf.data.Dataset.list_files(val_images_path, seed=SEED)
    val_dataset = val_dataset.map(parse_image_ISIC_2018).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    opt = tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
    model.compile(optimizer=opt, loss=loss_func, metrics=[dice_loss, tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath_h5, verbose=1, save_best_only=True, monitor='val_binary_io_u', mode='max')]
    model.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=callbacks, validation_data=val_dataset)
    
    best_model = tf.keras.models.load_model(filepath_h5, custom_objects={'dice_loss': dice_loss})

    mIoU_val, dice_score_val = benchmark_ISIC2018(best_model, val_images_dir, val_masks_dir, val_pred_dir, h, w, c, print_results=print_results) 
    mIoU_test, dice_score_test = benchmark_ISIC2018(best_model, test_images_dir, test_masks_dir, test_pred_dir, h, w, c, print_results=print_results) 
    mIoU_train_unlabeled, dice_score_train_unlabeled = benchmark_ISIC2018(best_model, unlabeled_images_dir, unlabeled_masks_dir, unlabeled_pred_dir, h, w, c, print_results=print_results) 
    
    print(f'{modelname} mIoU_val: {mIoU_val}')

    return mIoU_val, mIoU_test, mIoU_train_unlabeled, dice_score_val, dice_score_test, dice_score_train_unlabeled



def train_hela(train_images_dir, 
               val_images_dir,
               val_gt_dir, 
               test_gt_dir, 
               unlabeled_gt_dir, 
               modelname, 
               filepath_h5, 
               model,
               loss_func, 
               steps_per_epoch, 
               h, w, c,
               val_pred_dir, 
               test_pred_dir, 
               unlabeled_pred_dir): 
    
    train_images_path = os.path.join(train_images_dir, '*.png')
    train_dataset = tf.data.Dataset.list_files(train_images_path, seed=SEED)
    train_dataset = train_dataset.map(parse_image_hela).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    
    val_images_path = os.path.join(val_images_dir, '*.png')
    val_dataset = tf.data.Dataset.list_files(val_images_path, seed=SEED)
    val_dataset = val_dataset.map(parse_image_hela).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    opt = tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
    model.compile(optimizer=opt, loss=loss_func)#, metrics=['acc', MeanIoU(num_classes=n_classes)])   val_mean_io_u
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath_h5, verbose=1, save_best_only=True, monitor='val_loss', mode='min')]
    model.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=callbacks, validation_data=val_dataset)

    best_model = tf.keras.models.load_model(filepath_h5)

    mIoU_val, mIoU_ad_val, mean_cell_count_error_val = benchmark_hela(best_model, val_gt_dir, val_pred_dir, h, w, c)
    mIoU_test, mIoU_ad_test, mean_cell_count_error_test = benchmark_hela(best_model, test_gt_dir, test_pred_dir, h, w, c)
    mIoU_unlabeled, mIoU_ad_unlabeled, mean_cell_count_error_unlabeled = benchmark_hela(best_model, unlabeled_gt_dir, unlabeled_pred_dir, h, w, c)


    print(f'{modelname} mIoU_val: {mIoU_val}   mean_cell_count_error_val: {mean_cell_count_error_val}')

    return mIoU_val, mIoU_ad_val, mean_cell_count_error_val, mIoU_test, mIoU_ad_test, mean_cell_count_error_test, mIoU_unlabeled, mIoU_ad_unlabeled, mean_cell_count_error_unlabeled





def train_multiclass(train_images_dir, 
                        val_images_dir, 
                        val_masks_dir, 
                        test_images_dir, 
                        test_masks_dir, 
                        unlabeled_images_dir, 
                        unlabeled_masks_dir, 
                        modelname, 
                        filepath_h5, 
                        model,
                        loss_func, 
                        steps_per_epoch, 
                        h, w, c,
                        n_classes,
                        class_to_color_mapping,
                        val_pred_dir, 
                        test_pred_dir, 
                        unlabeled_pred_dir, 
                        print_results=False):
    
    train_images_path = os.path.join(train_images_dir, '*.png')
    train_dataset = tf.data.Dataset.list_files(train_images_path, seed=SEED)
    train_dataset = train_dataset.map(lambda x: parse_image_multiclass(x, n_classes=n_classes, image_channels=3)).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    
    val_images_path = os.path.join(val_images_dir, '*.png')
    val_dataset = tf.data.Dataset.list_files(val_images_path, seed=SEED)
    val_dataset = val_dataset.map(lambda x: parse_image_multiclass(x, n_classes=n_classes, image_channels=3)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    opt = tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
    model.compile(optimizer=opt, loss=loss_func, metrics=['acc', MeanIoU(num_classes=n_classes)])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath_h5, verbose=1, save_best_only=True, monitor='val_mean_io_u', mode='max')]
    model.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=callbacks, validation_data=val_dataset)

    best_model = tf.keras.models.load_model(filepath_h5, custom_objects={'MeanIoU': MeanIoU})
    best_model = tf.keras.models.load_model(filepath_h5, custom_objects={'MeanIoU': MeanIoU}, compile=False)

    mPA_val, mIoU_val = benchmark_multiclass(best_model, val_images_dir, val_masks_dir, val_pred_dir, h, w, c, class_to_color_mapping, print_results=print_results)
    mPA_test, mIoU_test  = benchmark_multiclass(best_model, test_images_dir, test_masks_dir, test_pred_dir, h, w, c, class_to_color_mapping, print_results=print_results)
    mPA_train_unlabeled, mIoU_train_unlabeled = benchmark_multiclass(best_model, unlabeled_images_dir, unlabeled_masks_dir, unlabeled_pred_dir, h, w, c, class_to_color_mapping, print_results=print_results)
    
    print(f'{modelname} mIoU_val: {mIoU_val}')

    return mPA_val, mPA_test, mPA_train_unlabeled, mIoU_val, mIoU_test, mIoU_train_unlabeled



def train_depth_map(train_images_dir, 
                        val_images_dir, 
                        test_images_dir, 
                        unlabeled_images_dir, 
                        modelname, 
                        filepath_h5, 
                        model, 
                        loss_func, 
                        steps_per_epoch, 
                        val_pred_dir, 
                        test_pred_dir, 
                        unlabeled_pred_dir):
    
    train_images_path = os.path.join(train_images_dir, '*.png')
    train_dataset = tf.data.Dataset.list_files(train_images_path, seed=SEED)
    train_dataset = train_dataset.map(parse_image_depth_map).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    
    val_images_path = os.path.join(val_images_dir, '*.png')
    val_dataset = tf.data.Dataset.list_files(val_images_path, seed=SEED)
    val_dataset = val_dataset.map(parse_image_depth_map).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    opt = tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
    model.compile(optimizer=opt, loss=loss_func, metrics=['mse', delta_metric(1.25)])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath_h5, verbose=1, save_best_only=True, monitor='val_loss', mode='min')]
    model.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=callbacks, validation_data=val_dataset)
    
    best_model = tf.keras.models.load_model(filepath_h5, custom_objects={'rmse': rmse, 'delta_1.25': delta_metric(1.25)})


    test_images_path = os.path.join(test_images_dir, '*.png')
    test_dataset = tf.data.Dataset.list_files(test_images_path, seed=SEED)
    test_dataset = test_dataset.map(parse_image_depth_map).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    unlabeled_images_path = os.path.join(unlabeled_images_dir, '*.png')
    unlabeled_dataset = tf.data.Dataset.list_files(unlabeled_images_path, seed=SEED)
    unlabeled_dataset = unlabeled_dataset.map(parse_image_depth_map).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    rmse_val, mse_val = benchmark_depth_map(best_model, val_dataset, val_images_dir, val_pred_dir)
    rmse_test, mse_test = benchmark_depth_map(best_model, test_dataset, test_images_dir, test_pred_dir)
    rmse_unlabeled, mse_unlabeled = benchmark_depth_map(best_model, unlabeled_dataset, unlabeled_images_dir, unlabeled_pred_dir)

    print(f'{modelname} rmse_val: {rmse_val}')

    return rmse_val, rmse_test, rmse_unlabeled, mse_val, mse_test, mse_unlabeled


def train_ISIC_2018_consistency_loss(train_images_dir, 
                                         train_masks_dir, 
                                         val_images_dir, 
                                         val_masks_dir, 
                                         test_images_dir, 
                                         test_masks_dir, 
                                         unlabeled_images_dir, 
                                         unlabeled_masks_dir, 
                                         modelname, 
                                         filepath_h5, 
                                         model, 
                                         loss_func, 
                                         h, w, c,
                                         val_pred_dir, 
                                         test_pred_dir, 
                                         unlabeled_pred_dir, 
                                         max_blur, max_noise, brightness_range_alpha, brightness_range_beta, 
                                         validation_frequency=1
                                         ):

    labeled_images, labeled_masks = load_labeled_data_single_mask(train_images_dir, train_masks_dir)
    unlabeled_images = load_unlabeled_images(unlabeled_images_dir)

    validation_images, validation_masks = load_labeled_data_single_mask(val_images_dir, val_masks_dir)

    opt = tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
    model.compile(optimizer=opt, loss=loss_func)    #, metrics=[dice_loss]

    current_validation_loss = 100


    for epoch in range(NUM_EPOCHS_CS):
        # Shuffle the data
        labeled_indices = np.random.permutation(len(labeled_images))
        unlabeled_indices = np.random.permutation(len(unlabeled_images))
        
        # Create batches for labeled and unlabeled data
        num_labeled_batches = int(np.ceil(len(labeled_images) / BATCH_SIZE))
        num_unlabeled_batches = int(np.ceil(len(unlabeled_images) / BATCH_SIZE))
        
        
        # Iterate through labeled data batches
        for batch_idx in range(num_labeled_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = (batch_idx + 1) * BATCH_SIZE
            batch_indices = labeled_indices[batch_start:batch_end]
        
            batch_labeled_images = labeled_images[batch_indices]
            batch_labeled_masks = labeled_masks[batch_indices]
        
            model.train_on_batch(batch_labeled_images, batch_labeled_masks)
        
        if epoch % validation_frequency == 0:
            validation_loss = model.evaluate(validation_images, validation_masks)
            print(f'Epoch: {epoch}, Validation Loss: {round(validation_loss,4)}  after labeled training')
        
            if validation_loss < current_validation_loss:
                current_validation_loss = validation_loss
                print(f' -------------------------- Model saved with val_loss {round(validation_loss,4)}')
                model.save(filepath_h5)


        # Iterate through unlabeled data batches
        for batch_idx in range(num_unlabeled_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = (batch_idx + 1) * BATCH_SIZE
            batch_indices = unlabeled_indices[batch_start:batch_end]
        
            batch_unlabeled_images = unlabeled_images[batch_indices]
        
            batch_unlabeled_images_rotated_flipped = np.array([apply_random_flip_and_rotation(image) for image in batch_unlabeled_images])
        
            augmented_unlabeled_images_1 = np.array([data_augmentation_image2tensor(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta ) for image in batch_unlabeled_images_rotated_flipped])
            augmented_unlabeled_images_2 = np.array([data_augmentation_image2tensor(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta ) for image in batch_unlabeled_images_rotated_flipped])
        
            with tf.GradientTape() as tape:
                predictions_1 = model(augmented_unlabeled_images_1, training=True)
                predictions_2 = model(augmented_unlabeled_images_2, training=True)
        
                consistency_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(predictions_1, predictions_2))
        
                current_weights = model.trainable_variables
        
                gradients = tape.gradient(consistency_loss, current_weights)
        
            opt.apply_gradients(zip(gradients, current_weights))
        
        
        # Evaluate the model on the validation set (optional)
        if epoch % validation_frequency == 0:
            validation_loss = model.evaluate(validation_images, validation_masks)
            print(f'Epoch: {epoch}, Validation Loss: {round(validation_loss,4)}  after unlabeled training')
        
            if validation_loss < current_validation_loss:
                current_validation_loss = validation_loss
                print(f' -------------------------- Model saved with val_loss {round(validation_loss,4)}')
                model.save(filepath_h5)


    best_model = tf.keras.models.load_model(filepath_h5, custom_objects={'dice_loss': dice_loss})

    mIoU_val, dice_score_val = benchmark_ISIC2018(best_model, val_images_dir, val_masks_dir, val_pred_dir, h, w, c) 
    mIoU_test, dice_score_test = benchmark_ISIC2018(best_model, test_images_dir, test_masks_dir, test_pred_dir, h, w, c) 
    mIoU_train_unlabeled, dice_score_train_unlabeled = benchmark_ISIC2018(best_model, unlabeled_images_dir, unlabeled_masks_dir, unlabeled_pred_dir, h, w, c) 
   
    print(f'{modelname} mIoU_val: {mIoU_val}')

    return mIoU_val, mIoU_test, mIoU_train_unlabeled, dice_score_val, dice_score_test, dice_score_train_unlabeled




def train_hela_consistency_loss(train_main_dir, 
                                val_main_dir, 
                                test_main_dir, 
                                unlabeled_main_dir,
                                modelname, 
                                filepath_h5, 
                                model, 
                                loss_func, 
                                h, w, c,
                                val_pred_dir, 
                                test_pred_dir, 
                                unlabeled_pred_dir, 
                                max_blur, max_noise, brightness_range_alpha, brightness_range_beta, 
                                validation_frequency=1):

    labeled_images, labeled_masks_alive, labeled_masks_dead, labeled_masks_pos  = load_images_and_masks_hela(train_main_dir)
    unlabeled_images = load_unlabeled_images_hela(os.path.join(unlabeled_main_dir, 'brightfield'))

    validation_images, validation_binary_masks_alive, validation_binary_masks_dead, validation_binary_masks_pos = load_images_and_masks_hela(val_main_dir)

    validation_binary_masks_alive = np.expand_dims(validation_binary_masks_alive, axis=-1)
    validation_binary_masks_dead = np.expand_dims(validation_binary_masks_dead, axis=-1)
    validation_binary_masks_pos = np.expand_dims(validation_binary_masks_pos, axis=-1)
    
    validation_masks = np.concatenate([validation_binary_masks_alive, validation_binary_masks_dead, validation_binary_masks_pos], axis=-1)

    opt = tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
    model.compile(optimizer=opt, loss=loss_func)    #, metrics=[dice_loss]

    current_validation_loss = 100

    for epoch in range(NUM_EPOCHS_CS):
        # Shuffle the data
        labeled_indices = np.random.permutation(len(labeled_images))
        unlabeled_indices = np.random.permutation(len(unlabeled_images))
        
        # Create batches for labeled and unlabeled data
        num_labeled_batches = int(np.ceil(len(labeled_images) / BATCH_SIZE))
        num_unlabeled_batches = int(np.ceil(len(unlabeled_images) / BATCH_SIZE))
        
        
        # Iterate through labeled data batches
        for batch_idx in range(num_labeled_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = (batch_idx + 1) * BATCH_SIZE
            batch_indices = labeled_indices[batch_start:batch_end]

            batch_labeled_images = labeled_images[batch_indices]
            batch_labeled_masks_alive = labeled_masks_alive[batch_indices]
            batch_labeled_masks_dead = labeled_masks_dead[batch_indices]
            batch_labeled_masks_pos = labeled_masks_pos[batch_indices]

            batch_labeled_masks_alive = np.expand_dims(batch_labeled_masks_alive, axis=-1)
            batch_labeled_masks_dead = np.expand_dims(batch_labeled_masks_dead, axis=-1)
            batch_labeled_masks_pos = np.expand_dims(batch_labeled_masks_pos, axis=-1)
            
            batch_labeled_masks = np.concatenate([batch_labeled_masks_alive, batch_labeled_masks_dead, batch_labeled_masks_pos], axis=-1)

            model.train_on_batch(batch_labeled_images, batch_labeled_masks)
        
        if epoch % validation_frequency == 0:
            validation_loss = model.evaluate(validation_images, validation_masks)
            print(f'Epoch: {epoch}, Validation Loss: {round(validation_loss,4)}  after labeled training')
        
            if validation_loss < current_validation_loss:
                current_validation_loss = validation_loss
                print(f' -------------------------- Model saved with val_loss {round(validation_loss,4)}')
                model.save(filepath_h5)


        # Iterate through unlabeled data batches
        for batch_idx in range(num_unlabeled_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = (batch_idx + 1) * BATCH_SIZE
            batch_indices = unlabeled_indices[batch_start:batch_end]
        
            batch_unlabeled_images = unlabeled_images[batch_indices]

            batch_unlabeled_images_rotated_flipped = np.array([apply_random_flip_and_rotation(image) for image in batch_unlabeled_images])

            augmented_unlabeled_images_1 = np.array([data_augmentation_image2tensor(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta) for image in batch_unlabeled_images_rotated_flipped])
            augmented_unlabeled_images_2 = np.array([data_augmentation_image2tensor(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta) for image in batch_unlabeled_images_rotated_flipped])
        
            with tf.GradientTape() as tape:
                predictions_1 = model(augmented_unlabeled_images_1, training=True)
                predictions_2 = model(augmented_unlabeled_images_2, training=True)
        
                consistency_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(predictions_1, predictions_2))
        
                current_weights = model.trainable_variables
        
                gradients = tape.gradient(consistency_loss, current_weights)
        
            opt.apply_gradients(zip(gradients, current_weights))


        # Evaluate the model on the validation set (optional)
        if epoch % validation_frequency == 0:
            validation_loss = model.evaluate(validation_images, validation_masks)
            print(f'Epoch: {epoch}, Validation Loss: {round(validation_loss,4)}  after unlabeled training')
        
            if validation_loss < current_validation_loss:
                current_validation_loss = validation_loss
                print(f' -------------------------- Model saved with val_loss {round(validation_loss,4)}')
                model.save(filepath_h5)


    best_model = tf.keras.models.load_model(filepath_h5)

    mIoU_val, mIoU_ad_val, mean_cell_count_error_val = benchmark_hela(best_model, val_main_dir, val_pred_dir, h, w, c)
    mIoU_test, mIoU_ad_test, mean_cell_count_error_test = benchmark_hela(best_model, test_main_dir, test_pred_dir, h, w, c)
    mIoU_unlabeled, mIoU_ad_unlabeled, mean_cell_count_error_unlabeled = benchmark_hela(best_model, unlabeled_main_dir, unlabeled_pred_dir, h, w, c)


    print(f'{modelname} mIoU_val: {mIoU_val}   mean_cell_count_error_val: {mean_cell_count_error_val}')

    return mIoU_val, mIoU_ad_val, mean_cell_count_error_val, mIoU_test, mIoU_ad_test, mean_cell_count_error_test, mIoU_unlabeled, mIoU_ad_unlabeled, mean_cell_count_error_unlabeled





def train_multiclass_consistency_loss(train_images_dir, 
                                      train_masks_dir, 
                                      val_images_dir, 
                                      val_masks_dir, 
                                      test_images_dir, 
                                      test_masks_dir, 
                                      unlabeled_images_dir, 
                                      unlabeled_masks_dir, 
                                      modelname, 
                                      filepath_h5, 
                                      model, 
                                      loss_func, 
                                      h, w, c,
                                      n_classes,
                                      class_to_color_mapping,
                                      val_pred_dir, 
                                      test_pred_dir, 
                                      unlabeled_pred_dir, 
                                      max_blur, max_noise, brightness_range_alpha, brightness_range_beta, 
                                      validation_frequency=1,
                                      print_results=False):

    labeled_images, labeled_masks = load_labeled_data_multiclass_mask(train_images_dir, train_masks_dir, n_classes)
    unlabeled_images = load_unlabeled_images(unlabeled_images_dir)

    validation_images, validation_masks = load_labeled_data_multiclass_mask(val_images_dir, val_masks_dir, n_classes)

    opt = tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
    model.compile(optimizer=opt, loss=loss_func)    #, metrics=[dice_loss]

    current_validation_loss = 100

    for epoch in range(NUM_EPOCHS_CS):
        # Shuffle the data
        labeled_indices = np.random.permutation(len(labeled_images))
        unlabeled_indices = np.random.permutation(len(unlabeled_images))
        
        # Create batches for labeled and unlabeled data
        num_labeled_batches = int(np.ceil(len(labeled_images) / BATCH_SIZE))
        num_unlabeled_batches = int(np.ceil(len(unlabeled_images) / BATCH_SIZE))
        
        
        # Iterate through labeled data batches
        for batch_idx in range(num_labeled_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = (batch_idx + 1) * BATCH_SIZE
            batch_indices = labeled_indices[batch_start:batch_end]
        
            batch_labeled_images = labeled_images[batch_indices]
            batch_labeled_masks = labeled_masks[batch_indices]
        
            model.train_on_batch(batch_labeled_images, batch_labeled_masks)
        
        if epoch % validation_frequency == 0:
            validation_loss = model.evaluate(validation_images, validation_masks)
            print(f'Epoch: {epoch}, Validation Loss: {round(validation_loss,4)}  after labeled training')
        
            if validation_loss < current_validation_loss:
                current_validation_loss = validation_loss
                print(f' -------------------------- Model saved with val_loss {round(validation_loss,4)}')
                model.save(filepath_h5)


        # Iterate through unlabeled data batches
        for batch_idx in range(num_unlabeled_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = (batch_idx + 1) * BATCH_SIZE
            batch_indices = unlabeled_indices[batch_start:batch_end]
        
            batch_unlabeled_images = unlabeled_images[batch_indices]

            augmented_unlabeled_images_1 = np.array([data_augmentation_image2tensor(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta) for image in batch_unlabeled_images])
            augmented_unlabeled_images_2 = np.array([data_augmentation_image2tensor(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta) for image in batch_unlabeled_images])
        
            with tf.GradientTape() as tape:
                predictions_1 = model(augmented_unlabeled_images_1, training=True)
                predictions_2 = model(augmented_unlabeled_images_2, training=True)
        
                consistency_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(predictions_1, predictions_2))
        
                current_weights = model.trainable_variables
        
                gradients = tape.gradient(consistency_loss, current_weights)
        
            opt.apply_gradients(zip(gradients, current_weights))


        # Evaluate the model on the validation set (optional)
        if epoch % validation_frequency == 0:
            validation_loss = model.evaluate(validation_images, validation_masks)
            print(f'Epoch: {epoch}, Validation Loss: {round(validation_loss,4)}  after unlabeled training')
        
            if validation_loss < current_validation_loss:
                current_validation_loss = validation_loss
                print(f' -------------------------- Model saved with val_loss {round(validation_loss,4)}')
                model.save(filepath_h5)


    best_model = tf.keras.models.load_model(filepath_h5, custom_objects={'dice_loss': dice_loss})
    
    mPA_val, mIoU_val = benchmark_multiclass(best_model, val_images_dir, val_masks_dir, val_pred_dir, h, w, c, class_to_color_mapping, print_results=print_results)
    mPA_test, mIoU_test  = benchmark_multiclass(best_model, test_images_dir, test_masks_dir, test_pred_dir, h, w, c, class_to_color_mapping, print_results=print_results)
    mPA_train_unlabeled, mIoU_train_unlabeled = benchmark_multiclass(best_model, unlabeled_images_dir, unlabeled_masks_dir, unlabeled_pred_dir, h, w, c, class_to_color_mapping, print_results=print_results)
   
    print(f'{modelname} mIoU_val: {mIoU_val}')

    return mPA_val, mPA_test, mPA_train_unlabeled, mIoU_val, mIoU_test, mIoU_train_unlabeled





def train_depth_map_consistency_loss(train_images_dir, 
                                         train_depth_maps_dir, 
                                         val_images_dir, 
                                         val_depth_maps_dir, 
                                         test_images_dir, 
                                         unlabeled_images_dir, 
                                         modelname, 
                                         filepath_h5, 
                                         model, 
                                         loss_func, 
                                         h, w, c,
                                         val_pred_dir, 
                                         test_pred_dir, 
                                         unlabeled_pred_dir, 
                                         max_blur, max_noise, brightness_range_alpha, brightness_range_beta, 
                                         validation_frequency=1
                                         ):

    labeled_images, labeled_depth_maps = load_labeled_data_depth_map(train_images_dir, train_depth_maps_dir)
    unlabeled_images = load_unlabeled_images(unlabeled_images_dir)

    validation_images, validation_depth_maps = load_labeled_data_depth_map(val_images_dir, val_depth_maps_dir)

    opt = tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
    model.compile(optimizer=opt, loss=loss_func)    #, metrics=[dice_loss]

    current_validation_loss = 100

    for epoch in range(NUM_EPOCHS_CS):
        # Shuffle the data
        labeled_indices = np.random.permutation(len(labeled_images))
        unlabeled_indices = np.random.permutation(len(unlabeled_images))
        
        # Create batches for labeled and unlabeled data
        num_labeled_batches = int(np.ceil(len(labeled_images) / BATCH_SIZE))
        num_unlabeled_batches = int(np.ceil(len(unlabeled_images) / BATCH_SIZE))
        
        
        # Iterate through labeled data batches
        for batch_idx in range(num_labeled_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = (batch_idx + 1) * BATCH_SIZE
            batch_indices = labeled_indices[batch_start:batch_end]
        
            batch_labeled_images = labeled_images[batch_indices]
            batch_labeled_depth_maps = labeled_depth_maps[batch_indices]
        
            model.train_on_batch(batch_labeled_images, batch_labeled_depth_maps)
        
        if epoch % validation_frequency == 0:
            validation_loss = model.evaluate(validation_images, validation_depth_maps)
            print(f'Epoch: {epoch}, Validation Loss: {round(validation_loss,4)}  after labeled training')
        
            if validation_loss < current_validation_loss:
                current_validation_loss = validation_loss
                print(f' -------------------------- Model saved with val_loss {round(validation_loss,4)}')
                model.save(filepath_h5)


        # Iterate through unlabeled data batches
        for batch_idx in range(num_unlabeled_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = (batch_idx + 1) * BATCH_SIZE
            batch_indices = unlabeled_indices[batch_start:batch_end]
        
            batch_unlabeled_images = unlabeled_images[batch_indices]

            augmented_unlabeled_images_1 = np.array([data_augmentation_image2tensor(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta) for image in batch_unlabeled_images])
            augmented_unlabeled_images_2 = np.array([data_augmentation_image2tensor(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta) for image in batch_unlabeled_images])
        
            with tf.GradientTape() as tape:
                predictions_1 = model(augmented_unlabeled_images_1, training=True)
                predictions_2 = model(augmented_unlabeled_images_2, training=True)
        
                consistency_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(predictions_1, predictions_2))
        
                current_weights = model.trainable_variables
        
                gradients = tape.gradient(consistency_loss, current_weights)
        
            opt.apply_gradients(zip(gradients, current_weights))


        # Evaluate the model on the validation set (optional)
        if epoch % validation_frequency == 0:
            validation_loss = model.evaluate(validation_images, validation_depth_maps)
            print(f'Epoch: {epoch}, Validation Loss: {round(validation_loss,4)}  after unlabeled training')
        
            if validation_loss < current_validation_loss:
                current_validation_loss = validation_loss
                print(f' -------------------------- Model saved with val_loss {round(validation_loss,4)}')
                model.save(filepath_h5)


    best_model = tf.keras.models.load_model(filepath_h5)

    val_images_path = os.path.join(val_images_dir, '*.png')
    val_dataset = tf.data.Dataset.list_files(val_images_path, seed=SEED)
    val_dataset = val_dataset.map(parse_image_depth_map).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_images_path = os.path.join(test_images_dir, '*.png')
    test_dataset = tf.data.Dataset.list_files(test_images_path, seed=SEED)
    test_dataset = test_dataset.map(parse_image_depth_map).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    unlabeled_images_path = os.path.join(unlabeled_images_dir, '*.png')
    unlabeled_dataset = tf.data.Dataset.list_files(unlabeled_images_path, seed=SEED)
    unlabeled_dataset = unlabeled_dataset.map(parse_image_depth_map).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    rmse_val, mse_val = benchmark_depth_map(best_model, val_dataset, val_pred_dir)
    rmse_test, mse_test = benchmark_depth_map(best_model, test_dataset, test_pred_dir)
    rmse_unlabeled, mse_unlabeled = benchmark_depth_map(best_model, unlabeled_dataset, unlabeled_pred_dir)


    print(f'{modelname} rmse_val: {rmse_val}')

    return rmse_val, rmse_test, rmse_unlabeled, mse_val, mse_test, mse_unlabeled








def load_labeled_data_single_mask(labeled_images_dir, labeled_masks_dir):
    '''
    Reads images and masks from specified directories, processes them, and returns them as numpy arrays.

    Args:
        labeled_images_dir (str): Directory containing labeled images.
        labeled_masks_dir (str): Directory containing corresponding masks for the images.

    Returns:
        tuple: A tuple containing two numpy arrays:
               - images: RGB images as numpy arrays.
               - binary_masks: Binary masks corresponding to the images.
    '''
    
    image_filenames = sorted(os.listdir(labeled_images_dir))
    mask_filenames = sorted(os.listdir(labeled_masks_dir))

    # Added the color conversion for each image after reading it
    images = [cv2.cvtColor(cv2.imread(os.path.join(labeled_images_dir, fname)), cv2.COLOR_BGR2RGB) for fname in image_filenames]
    
    masks = [cv2.imread(os.path.join(labeled_masks_dir, fname), cv2.IMREAD_GRAYSCALE) for fname in mask_filenames]
    images = np.array(images)
    masks = np.array(masks)

    # Normalize the masks by dividing them by 255
    masks_normalized = masks.astype(np.float32) / 255.0

    # Threshold the normalized masks to obtain binary masks (0 or 1)
    binary_masks = np.where(masks_normalized > 0.5, 1, 0)

    return images, binary_masks



def load_labeled_data_multiclass_mask(labeled_images_dir, labeled_masks_dir, num_classes):
    '''
    Reads images and masks from specified directories, processes them for multiclass segmentation tasks, and returns them as numpy arrays. Masks are converted to one-hot encoded format based on the number of classes.

    Args:
        labeled_images_dir (str): Directory containing labeled images.
        labeled_masks_dir (str): Directory containing corresponding masks for the images.
        num_classes (int): The number of classes for segmentation.

    Returns:
        tuple: A tuple containing two numpy arrays:
               - images: RGB images as numpy arrays.
               - masks: One-hot encoded masks corresponding to the images.
    '''
    
    image_filenames = sorted(os.listdir(labeled_images_dir))
    mask_filenames = sorted(os.listdir(labeled_masks_dir))

    # Added the color conversion for each image after reading it
    images = [cv2.cvtColor(cv2.imread(os.path.join(labeled_images_dir, fname)), cv2.COLOR_BGR2RGB) for fname in image_filenames]

    masks = [cv2.imread(os.path.join(labeled_masks_dir, fname), cv2.IMREAD_GRAYSCALE) for fname in mask_filenames]
    images = np.array(images)
    masks = np.array(masks)
    
    # One-hot encoding
    masks = to_categorical(masks, num_classes)

    return images, masks



def load_labeled_data_depth_map(labeled_images_dir, labeled_depth_maps_dir):
    '''
    Reads images and depth maps from specified directories, processes them, and returns them as numpy arrays. Depth maps are normalized to float values between 0 and 1.

    Args:
        labeled_images_dir (str): Directory containing labeled images.
        labeled_depth_maps_dir (str): Directory containing corresponding depth maps for the images.

    Returns:
        tuple: A tuple containing two numpy arrays:
               - images: Images as numpy arrays.
               - depth_maps_normalized: Normalized depth maps corresponding to the images.
    '''
    
    image_filenames = sorted(os.listdir(labeled_images_dir))
    mask_filenames = sorted(os.listdir(labeled_depth_maps_dir))

    images = [cv2.imread(os.path.join(labeled_images_dir, fname)) for fname in image_filenames]
    depth_maps = [cv2.imread(os.path.join(labeled_depth_maps_dir, fname), cv2.IMREAD_GRAYSCALE) for fname in mask_filenames]
    images = np.array(images)
    depth_maps = np.array(depth_maps)

    depth_maps_normalized = depth_maps.astype(np.float32) / 255.0

    return images, depth_maps_normalized


def load_unlabeled_images(unlabeled_images_dir, read_as_grayscale=False):
    '''
    Reads images from the given directory and can optionally convert them to grayscale. The images are returned as a numpy array.

    Args:
        unlabeled_images_dir (str): Directory containing unlabeled images.
        read_as_grayscale (bool, optional): Flag to indicate whether the images should be read in grayscale. Defaults to False.

    Returns:
        np.array: Array of images read from the directory.
    '''
    
    image_filenames = os.listdir(unlabeled_images_dir)

    if read_as_grayscale:
        unlabeled_images = [cv2.imread(os.path.join(unlabeled_images_dir, fname), cv2.IMREAD_GRAYSCALE) for fname in image_filenames]
    else:
        unlabeled_images = [cv2.imread(os.path.join(unlabeled_images_dir, fname)) for fname in image_filenames]     

    unlabeled_images = np.array(unlabeled_images)

    return unlabeled_images



def parse_image_ISIC_2018(image_dir, IMG_CHANNELS=3): 
    '''
    Parse an image and its corresponding mask from the ISIC 2018 dataset.

    Args:
        image_dir (str): Directory of the image file.
        IMG_CHANNELS (int, optional): Number of channels in the image (e.g., 3 for RGB). Defaults to 3.

    Returns:
        tuple: A tuple containing:
               - image: The decoded image tensor.
               - mask: The corresponding binary mask tensor.
    '''

    image = tf.io.read_file(image_dir)
    image = tf.image.decode_png(image, channels = IMG_CHANNELS)

    path_mask = tf.strings.regex_replace(image_dir, 'images', 'masks')
    mask = tf.io.read_file(path_mask)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast((mask / 255), tf.uint8)

    return image, mask


def parse_image_hela(path_brightfield, IMG_CHANNELS=1, Position_weight=3):  
    '''
    Reads a brightfield image along with its corresponding 'alive', 'dead', and 'position' masks.

    Args:
        path_brightfield (str): Path to the brightfield image.
        IMG_CHANNELS (int, optional): Number of channels in the brightfield image. Defaults to 1.
        Position_weight (int, optional): Weighting factor for the position image. Defaults to 3.

    Returns:
        tuple: A tuple containing:
               - bf_img: The brightfield image tensor.
               - comb_outputs: Combined tensor of 'alive', 'dead', and weighted 'position' images.
    '''

    bf_img = tf.io.read_file(path_brightfield)
    bf_img = tf.image.decode_png(bf_img, channels = IMG_CHANNELS)

    path_alive = tf.strings.regex_replace(path_brightfield, 'brightfield', 'alive')
    alive_img = tf.io.read_file(path_alive)
    alive_img = tf.image.decode_png(alive_img, channels=1)
    alive_img = tf.cast(alive_img / 255, tf.uint8)

    path_dead = tf.strings.regex_replace(path_brightfield, 'brightfield', 'dead')
    dead_img = tf.io.read_file(path_dead)
    dead_img = tf.image.decode_png(dead_img, channels=1)
    dead_img = tf.cast(dead_img / 255, tf.uint8)

    path_pos = tf.strings.regex_replace(path_brightfield, 'brightfield', 'mod_position')
    pos_img = tf.io.read_file(path_pos)
    pos_img = tf.image.decode_png(pos_img, channels=1)
    pos_img = tf.cast((pos_img / 255) *Position_weight, tf.uint8)

    comb_outputs = tf.squeeze(tf.stack([alive_img, dead_img, pos_img], axis=3))

    bf_img.set_shape([256, 256,1])
    comb_outputs.set_shape([256, 256,3])

    return bf_img, comb_outputs


def parse_image_multiclass(image_dir, n_classes, image_channels=3):      
    '''
    Reads an image and its mask from given directories, decodes them, and processes the mask for multiclass segmentation tasks using one-hot encoding.

    Args:
        image_dir (str): Directory of the image file.
        n_classes (int): The number of classes for segmentation.
        image_channels (int, optional): Number of channels in the image (e.g., 3 for RGB). Defaults to 3.

    Returns:
        tuple: A tuple containing:
               - image: The decoded image tensor.
               - mask: The corresponding one-hot encoded mask tensor.
    '''

    image = tf.io.read_file(image_dir)
    image = tf.image.decode_png(image, channels = image_channels)

    path_mask = tf.strings.regex_replace(image_dir, "images", "masks")
    mask = tf.io.read_file(path_mask)
    mask = tf.image.decode_png(mask, channels=1)
    
    # Convert the mask to one-hot encoding
    mask = tf.squeeze(mask)  # remove the channel dimension
    mask = tf.one_hot(mask, n_classes)  # create a new dimension for classes
    mask.set_shape([None, None, n_classes])  # Manually setting shape

    return image, mask


def parse_image_depth_map(image_dir, IMG_CHANNELS=3):    
    '''
    Parse an image and its corresponding depth map.
    
    Args:
        image_dir (str): Directory of the image file.
        IMG_CHANNELS (int, optional): Number of channels in the image (e.g., 3 for RGB). Defaults to 3.

    Returns:
        tuple: A tuple containing:
               - image: The decoded image tensor.
               - depth_map: The corresponding depth map tensor, normalized and converted to the appropriate data type.
    '''

    image = tf.io.read_file(image_dir)
    image = tf.image.decode_png(image, channels = IMG_CHANNELS)

    path_depth_map = tf.strings.regex_replace(image_dir, 'images', 'depth_maps')
    depth_map = tf.io.read_file(path_depth_map)
    depth_map = tf.image.decode_png(depth_map, channels=1)
    depth_map = tf.cast((depth_map / 255), tf.float32)  # check if tf.float16 is needed because of mixed_precision.set_global_policy('mixed_float16')

    return image, depth_map




def benchmark_ISIC2018(model, images_dir, masks_dir, pred_path, h, w, c, batch_size=64, create_images=True, print_results=False):
    '''
    Evaluates a given model on the ISIC 2018 dataset by computing Intersection over Union (IoU) and Dice scores. It also optionally saves the predicted masks and prints the results.

    Args:
        model: The model to be benchmarked.
        images_dir (str): Directory containing images.
        masks_dir (str): Directory containing corresponding ground truth masks.
        pred_path (str): Directory to save predicted masks.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        batch_size (int, optional): Size of the batch for model prediction. Defaults to 64.
        create_images (bool, optional): Flag to indicate whether to save predicted masks. Defaults to True.
        print_results (bool, optional): Flag to indicate whether to print the results. Defaults to False.

    Returns:
        tuple: A tuple containing mean IoU and mean Dice score.
    '''

    
    ious = []
    dice_scores = []
    batch_images = []
    batch_masks_masks = []
    batch_names = []

    os.makedirs(pred_path, exist_ok=True)

    for imagename in tqdm(os.listdir(images_dir)):
        input_image = cv2.imread(os.path.join(images_dir, imagename))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(os.path.join(masks_dir, imagename), 0)

        prepared_image = np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8)

        batch_images.append(prepared_image)
        batch_masks_masks.append(gt_mask)
        batch_names.append(imagename)

        if len(batch_images) == batch_size or imagename == os.listdir(images_dir)[-1]:
            batch_images_np = np.vstack(batch_images)
            with contextlib.redirect_stdout(io.StringIO()):
                pred_masks = model.predict(batch_images_np)
                pred_masks = ((pred_masks > 0.5)*255).astype(np.uint8)

            for idx, pred_mask in enumerate(pred_masks):
                pred_mask = pred_mask.squeeze()
                gt_mask = batch_masks_masks[idx]
                if create_images:
                    cv2.imwrite(os.path.join(pred_path, f'{batch_names[idx]}'), pred_mask)

                dice_score = round(dice_score_numpy_binary(gt_mask, pred_mask), 4)
                dice_scores.append(dice_score)

                iou = round(get_IoU_binary(gt_mask, pred_mask), 4) 
                ious.append(iou)

                if print_results:
                    print(f'{batch_names[idx]} IoU: {iou}    DS: {dice_score}')

            # Clear the batches
            batch_images = []
            batch_masks_masks = []
            batch_names = []

    sum_ious = np.sum(ious)
    mIoU = round((sum_ious / len(ious)) , 3)

    sum_dice_scores = np.sum(dice_scores)
    mdice_score = round((sum_dice_scores / len(dice_scores)) , 3)
    
    print(f'------------------------------------------------------------  mIoU: {mIoU}    mdice score: {mdice_score}  ------------------------------------------------------------')

    return mIoU, mdice_score




def benchmark_hela(model, gt_main_dir, pred_dir, h, w, c, threshold=0.5, batch_size=64, save_output=True, benchmark=True, mod_position=True):
    '''
    Evaluates a given model on HeLa cell images by computing mean Intersection over Union (mIoU) for 'alive', 'dead', and 'position' cells. It also saves the predicted masks and optionally benchmarks cell count accuracy.

    Args:
        model: The model to be benchmarked.
        gt_main_dir (str): Directory containing ground truth data.
        pred_dir (str): Directory to save predicted masks.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        threshold (float, optional): Threshold to apply for binary classification. Defaults to 0.5.
        batch_size (int, optional): Size of the batch for model prediction. Defaults to 64.
        save_output (bool, optional): Flag to indicate whether to save predicted masks. Defaults to True.
        benchmark (bool, optional): Flag to indicate whether to benchmark against ground truth. Defaults to True.
        mod_position (bool, optional): Flag to modify position mask size. Defaults to True.

    Returns:
        tuple: A tuple containing mean IoU for all classes, mean IoU for 'alive' and 'dead' classes, and mean cell count error.
    '''
    
    mIoUs = []
    mIoUs_ad = []
    cell_count_delta = 0
    os.makedirs(os.path.join(pred_dir, 'alive'), exist_ok=True)
    os.makedirs(os.path.join(pred_dir, 'dead'), exist_ok=True)

    if mod_position:
        os.makedirs(os.path.join(pred_dir, 'mod_position'), exist_ok=True)
    else:
        os.makedirs(os.path.join(pred_dir, 'position'), exist_ok=True)

    image_names = os.listdir(os.path.join(gt_main_dir, 'brightfield'))

    for batch_start in tqdm(range(0, len(image_names), batch_size)):
        batch_images = []
        batch_gt_alive = []
        batch_gt_dead = []
        batch_gt_pos = []
        batch_names = image_names[batch_start:batch_start + batch_size]

        for imagename in batch_names:
            img_gray = cv2.imread(os.path.join(gt_main_dir, 'brightfield', imagename), 0)
            preparedIMG = np.array(img_gray).reshape(-1, h, w, c).astype(np.float32)
            batch_images.append(preparedIMG)

            img_alive_gray = cv2.imread(os.path.join(gt_main_dir, 'alive', imagename), 0)
            batch_gt_alive.append(img_alive_gray)

            img_dead_gray = cv2.imread(os.path.join(gt_main_dir, 'dead', imagename), 0)
            batch_gt_dead.append(img_dead_gray)

            img_pos_gray = cv2.imread(os.path.join(gt_main_dir, 'mod_position', imagename), 0)
            batch_gt_pos.append(img_pos_gray)

        batch_images_np = np.vstack(batch_images)
        with contextlib.redirect_stdout(io.StringIO()):
            output_data = model.predict(batch_images_np)

        for idx, output in enumerate(output_data):
            alive, dead, pos = cv2.split(output)

            alive_uint = ((alive > threshold) * 255).astype(np.uint8).reshape(h, w)
            dead_uint = ((dead > threshold) * 255).astype(np.uint8).reshape(h, w)
            pos_uint = ((pos > threshold) * 255).astype(np.uint8).reshape(h, w)

            if mod_position:
                pos_uint = mod_pos_size(pos_uint)

            if benchmark:
                iou_alive = round(get_IoU_binary(batch_gt_alive[idx], alive_uint), 4)
                iou_dead = round(get_IoU_binary(batch_gt_dead[idx], dead_uint), 4)
                iou_pos = round(get_IoU_binary(batch_gt_pos[idx], pos_uint), 4)

                img_mIoU = (iou_alive + iou_dead + iou_pos) / 3
                mIoUs.append(img_mIoU)

                img_mIoU_ad = (iou_alive + iou_dead) / 2
                mIoUs_ad.append(img_mIoU_ad)

                pred_positions = get_pos_contours(pos_uint)
                pred_alive_count, pred_dead_count, _ = get_cell_count(pred_positions, alive_uint, dead_uint)

                gt_positions = get_pos_contours(batch_gt_pos[idx])
                gt_alive_count, gt_dead_count, _ = get_cell_count(gt_positions, batch_gt_alive[idx], batch_gt_dead[idx])

                alive_delta = abs(pred_alive_count - gt_alive_count)
                dead_delta = abs(pred_dead_count - gt_dead_count)
                total_delta_count = alive_delta + dead_delta
                cell_count_delta += total_delta_count

            if save_output:
                cv2.imwrite(os.path.join(pred_dir, 'alive', batch_names[idx]), alive_uint)
                cv2.imwrite(os.path.join(pred_dir, 'dead', batch_names[idx]), dead_uint)

                if mod_position:
                    cv2.imwrite(os.path.join(pred_dir, 'mod_position', batch_names[idx]), pos_uint)
                else:
                    cv2.imwrite(os.path.join(pred_dir, 'position', batch_names[idx]), pos_uint)

    mIoU = round(np.sum(mIoUs) / len(mIoUs), 3)
    mIoU_ad = round(np.sum(mIoUs_ad) / len(mIoUs_ad), 3)
    mcce = round(cell_count_delta / len(mIoUs), 3)

    print(f"------------------------------------------------------------   mIoU: {mIoU}      mIoU_ad: {mIoU_ad}      mcce: {mcce}  ------------------------------------------------------------")

    return mIoU, mIoU_ad, mcce




def benchmark_multiclass(model, image_path, gt_path, pred_path, h, w, c, class_to_color_mapping, batch_size=64, create_images=True, print_results=True):
    '''
    Evaluates a given model on a set of images by computing mean Intersection over Union (mIoU) and mean Pixel Accuracy (mPA). It also optionally saves the predicted masks both in class label and color-coded formats.

    Args:
        model: The model to be benchmarked.
        image_path (str): Path to the directory containing the images.
        gt_path (str): Path to the directory containing the ground truth masks.
        pred_path (str): Path to the directory where predicted masks will be saved.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        class_to_color_mapping (dict): A mapping from class labels to colors.
        batch_size (int, optional): Size of the batch for model prediction. Defaults to 64.
        create_images (bool, optional): Flag to indicate whether to save predicted masks. Defaults to True.
        print_results (bool, optional): Flag to indicate whether to print the results. Defaults to True.

    Returns:
        tuple: A tuple containing mean Pixel Accuracy (mPA) and mean IoU (mIoU).
    '''
    
    ious = []
    PAs = []    # Pixel Accuracy
    batch_images = []
    batch_gt_masks = []
    batch_names = []

    os.makedirs(pred_path, exist_ok=True)

    for imagename in tqdm(os.listdir(image_path)):
        input_image = cv2.imread(os.path.join(image_path, imagename))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(os.path.join(gt_path, imagename), 0)

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))

        batch_images.append(prepared_image)
        batch_gt_masks.append(gt_mask)
        batch_names.append(imagename)

        if len(batch_images) == batch_size or imagename == os.listdir(image_path)[-1]:
            batch_images_np = np.vstack(batch_images)
            with contextlib.redirect_stdout(io.StringIO()):
                pred_masks = model.predict(batch_images_np)
                pred_masks = np.argmax(pred_masks, axis=-1)

            for idx, pred_mask in enumerate(pred_masks):
                gt_mask = batch_gt_masks[idx]
                if create_images:
                    cv2.imwrite(os.path.join(pred_path, f'{batch_names[idx]}'), pred_mask)

                    if class_to_color_mapping is not None:
                        convert_class_to_color_mask(pred_mask, os.path.join(pred_path, f'{batch_names[idx][:-4]}_color.png'), class_to_color_mapping)

                pa = round(pixel_accuracy(pred_mask, gt_mask),4)    
                PAs.append(pa)

                iou = round(get_IoU_multi_unique(pred_mask, gt_mask),4) 
                ious.append(iou)

                if print_results:
                    print(f"{batch_names[idx]} IoU: {iou}    PA: {pa}")

            # Clear the batches
            batch_images = []
            batch_gt_masks = []
            batch_names = []

    sum_PAs = np.sum(PAs)
    mPA = round((sum_PAs / len(PAs)) , 3)

    sum_ious = np.sum(ious)
    mIoU = round((sum_ious / len(ious)) , 3)

    print(f"------------------------------------------------------------   mPA: {mPA}      mIoU: {mIoU}  ------------------------------------------------------------")

    return mPA, mIoU





def benchmark_depth_map(model, dataset, input_dir, output_dir):
    '''
    Evaluates a given model using a dataset, computes Root Mean Square Error (RMSE) and Mean Square Error (MSE), and saves the predicted depth maps.

    Args:
        model: The model to be benchmarked.
        dataset: The dataset used for evaluation.
        input_dir (str): Directory containing input images for prediction.
        output_dir (str): Directory to save the predicted depth maps.

    Returns:
        tuple: A tuple containing RMSE and MSE of the model evaluation.
    '''
    
    os.makedirs(output_dir, exist_ok=True)

    results = model.evaluate(dataset)

    rmse = results[0]
    mse = results[1]

    # Load images from input_dir
    input_images_path = os.path.join(input_dir, '*.png')
    input_dataset = tf.data.Dataset.list_files(input_images_path)
    input_dataset = input_dataset.map(load_and_preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    for img_paths, images in input_dataset:
        with contextlib.redirect_stdout(io.StringIO()):
            predictions = model.predict(images)
        
        # Convert predictions back to the [0, 255] scale for saving as PNG
        predictions_scaled = tf.clip_by_value(predictions * 255.0, 0, 255).numpy().astype(np.uint8)
        
        for pred, img_path in zip(predictions_scaled, img_paths):
            img_name = os.path.basename(img_path.numpy().decode('utf-8'))
            pred_img_path = os.path.join(output_dir, img_name)
            with tf.io.gfile.GFile(pred_img_path, 'wb') as f:
                f.write(tf.image.encode_png(pred).numpy())

    return rmse, mse



def load_and_preprocess_image(image_dir, IMG_CHANNELS=3):  
    '''
    Load and preprocess an image from a given directory.
    
    Args:
        image_dir (str): Directory of the image file.
        IMG_CHANNELS (int, optional): Number of channels to use in the image. Defaults to 3 (for RGB images).

    Returns:
        tuple: A tuple containing:
               - image_dir: The directory of the image.
               - image: The decoded and preprocessed image tensor.
    '''
    
    image = tf.io.read_file(image_dir)
    image = tf.image.decode_png(image, channels=IMG_CHANNELS)
    return image_dir, image




def input_ensemble_prediction(model, image, h, w, c, threshold, max_blur=3, max_noise=25, brightness_range_alpha=(0.5, 1.5), brightness_range_beta=(-25, 25), n=2, use_n_rnd_transformations=False):
    '''
    This function applies a set of transformations to the input image, uses a model to predict on these transformed images, and then combines these predictions into a single mask. 

    Args:
        model: The model used for prediction.
        image: The input image.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        threshold (float): Threshold for binary classification.
        max_blur (int, optional): Maximum blur to apply. Defaults to 3.
        max_noise (int, optional): Maximum noise to add. Defaults to 25.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.5, 1.5).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-25, 25).
        n (int, optional): Number of random transformations if 'use_n_rnd_transformations' is True. Defaults to 2.
        use_n_rnd_transformations (bool, optional): Flag to use N random transformations instead of all. Defaults to False.

    Returns:
        np.array: The predicted mask after combining the predictions from the transformed images.
    '''

    if use_n_rnd_transformations == True:
        transformed_images, transformations_applied = generate_random_transformations(image, n, max_blur, max_noise, brightness_range_alpha, brightness_range_beta)
    else:
        transformed_images = generate_all_transformations(image)
    
    prepared_images = [np.array((image).reshape(-1, h, w, c), dtype=np.uint8) for image in transformed_images]
    prepared_images = np.concatenate(prepared_images, axis=0)  # Combine into a single array

    with contextlib.redirect_stdout(io.StringIO()):
        masks = model.predict(prepared_images)
        masks[masks >= threshold] = 1
        masks[masks < threshold] = 0
        masks_2d = masks[:, :, :, 0]
        masks_2d = masks_2d.astype(np.uint8)

    masks_2d_list = [np.array(mask) for mask in masks_2d]


    if use_n_rnd_transformations == True:
        restored_masks = restore_random_transformations(masks_2d_list, transformations_applied)
    else:
        restored_masks = restore_all_transformations(masks_2d_list)

    mask_sum = np.sum(restored_masks, axis=0)
    pred_mask = np.zeros_like(mask_sum)
    pred_mask[mask_sum < len(transformed_images)] = 0
    pred_mask[mask_sum >= len(transformed_images)] = 255

    return pred_mask.astype(np.uint8)



def add_noise(image, max_noise=25):
    '''
    Add random noise to an image within a specified range.

    Args:
        image (np.array): The input image.
        max_noise (int, optional): The maximum magnitude of noise to add. Defaults to 25.

    Returns:
        np.array: The noisy image.
    '''
    
    noise = np.random.randint(max_noise*-1, max_noise, size=image.shape)
    image = np.clip(image.astype(np.int16) + noise, 0, 255)

    return image.astype(np.uint8)


def add_noise_and_blur(image, max_blur=3, max_noise=25):
    '''
    Applies Gaussian blur and random noise to a given image. The level of blur and noise can be controlled by the parameters.

    Args:
        image (np.array): The input image.
        max_blur (int, optional): The maximum kernel size for Gaussian blur. Defaults to 3.
        max_noise (int, optional): The maximum magnitude of noise to add. Defaults to 25.

    Returns:
        np.array: The modified image with noise and blur.
    '''

    rndint = random.randint(0,max_blur)
    if rndint == 1:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    elif rndint == 2:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    elif rndint == 3:
        image = cv2.GaussianBlur(image, (7, 7), 0)


    if max_noise > 0:
        image = add_noise(image, max_noise)

    return image


def apply_random_flip_and_rotation(image):
    '''
    Randomly flips an image vertically and/or horizontally, and applies random rotations (90, 180, or 270 degrees).

    Args:
        image (np.array): The input image.

    Returns:
        np.array: The transformed image.
    '''

    if random.randint(0,1) == 1:
        image = cv2.flip(image, 0)
    
    if random.randint(0,1) == 1:
        image = cv2.flip(image, 1)

    rndint = random.randint(0,3)
    if rndint > 0:
        if rndint == 1:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        if rndint == 2:
            image = cv2.rotate(image, cv2.ROTATE_180)

        if rndint == 3:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image



def data_augmentation_image2tensor(image, max_blur=3, max_noise=25, brightness_range_alpha=(0.5, 1.5), brightness_range_beta=(-25, 25)):
    '''
    Applies noise, blur, and random brightness adjustments to an image, followed by conversion to a TensorFlow tensor.

    Args:
        image (np.array): The input image.
        max_blur (int, optional): The maximum kernel size for Gaussian blur. Defaults to 3.
        max_noise (int, optional): The maximum magnitude of noise to add. Defaults to 25.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.5, 1.5).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-25, 25).

    Returns:
        tf.Tensor: The augmented image as a TensorFlow tensor.
    '''
    
    image = add_noise_and_blur(image, max_blur, max_noise)

    if random.randint(0,1) == 1:
        brightness_alpha  = np.random.uniform(brightness_range_alpha[0], brightness_range_alpha[1])
        brightness_beta  = np.random.uniform(brightness_range_beta[0], brightness_range_beta[1])

        image = cv2.convertScaleAbs(image, alpha=brightness_alpha, beta=brightness_beta)

    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=-1)

    return image


def data_augmentation_image(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta ):
    '''
    Apply noise, blur, and random brightness adjustments to an image.

    Args:
        image (np.array): The input image.
        max_blur (int): The maximum kernel size for Gaussian blur.
        max_noise (int): The maximum magnitude of noise to add.
        brightness_range_alpha (tuple): Range for alpha in brightness adjustment.
        brightness_range_beta (tuple): Range for beta in brightness adjustment.

    Returns:
        np.array: The augmented image.
    '''

    image = add_noise_and_blur(image, max_blur, max_noise)

    # Apply brightness and contrast adjustments
    if random.randint(0,1) == 1:
        brightness_alpha  = np.random.uniform(brightness_range_alpha[0], brightness_range_alpha[1])
        brightness_beta  = np.random.uniform(brightness_range_beta[0], brightness_range_beta[1])

        image = cv2.convertScaleAbs(image, alpha=brightness_alpha, beta=brightness_beta)

    return image


def generate_all_transformations(image):
    '''
    Generate all possible transformations of an image with flips and rotations.

    Args:
        image (np.array): The input image.

    Returns:
        list: A list of transformed images including the original image, flipped, and rotated versions.
    '''

    transformed_images = [image.copy()]

    for flip_horizontal in range(2):  # Loop through 0 (no flip) and 1 (flip)
        for flip_vertical in range(2):
            for rotation_type in range(1, 4):  # Loop through 1 (90°), 2 (180°) and 3 (270°)
                transformed_image = image.copy()

                if flip_horizontal == 1:
                    transformed_image = cv2.flip(transformed_image, 0)

                if flip_vertical == 1:
                    transformed_image = cv2.flip(transformed_image, 1)

                if rotation_type == 1:
                    transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_90_CLOCKWISE)
                elif rotation_type == 2:
                    transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_180)
                elif rotation_type == 3:
                    transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                transformed_images.append(transformed_image)

    return transformed_images





def restore_all_transformations(transformed_images):
    '''
    Restore all transformed images to their original orientation.

    This function reverses the transformations applied to a set of images, which includes flips and rotations, bringing them back to their original state.

    Args:
        transformed_images (list): A list of transformed images.

    Returns:
        list: A list of restored images.
    '''

    restored_images = [transformed_images.pop(0)]

    for flip_horizontal in range(2):  # Loop through 0 (no flip) and 1 (flip)
        for flip_vertical in range(2):
            for rotation_type in range(1, 4):  # Loop through 1 (90°), 2 (180°) and 3 (270°)
                transformed_image = transformed_images.pop(0)

                # Reverse transformations
                if rotation_type == 1:
                    transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation_type == 2:
                    transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_180)
                elif rotation_type == 3:
                    transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_90_CLOCKWISE)

                if flip_vertical == 1:
                    transformed_image = cv2.flip(transformed_image, 1)

                if flip_horizontal == 1:
                    transformed_image = cv2.flip(transformed_image, 0)

                restored_images.append(transformed_image)

    return restored_images


def generate_random_transformations(image, n, max_blur, max_noise, brightness_range_alpha, brightness_range_beta):
    '''
    Generate a set of randomly transformed images.

    This function applies a random combination of flips, rotations, and other augmentations to create multiple variations of the input image.

    Args:
        image (np.array): The input image.
        n (int): Number of random transformations to generate.
        max_blur (int): Maximum kernel size for Gaussian blur.
        max_noise (int): Maximum magnitude of noise to add.
        brightness_range_alpha (tuple): Range for alpha in brightness adjustment.
        brightness_range_beta (tuple): Range for beta in brightness adjustment.

    Returns:
        tuple: A tuple containing a list of transformed images and a list of applied transformations.
    '''

    all_transformations = []
    transformed_images = []
    transformations_applied = []

    # Generate all possible transformations
    for flip_horizontal in range(2):  # Loop through 0 (no flip) and 1 (flip)
        for flip_vertical in range(2):
            for rotation_type in range(1, 4):  # Loop through 1 (90°), 2 (180°) and 3 (270°)
                transformed_image = image.copy()

                if flip_horizontal == 1:
                    transformed_image = cv2.flip(transformed_image, 0)

                if flip_vertical == 1:
                    transformed_image = cv2.flip(transformed_image, 1)

                if rotation_type == 1:
                    transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_90_CLOCKWISE)
                elif rotation_type == 2:
                    transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_180)
                elif rotation_type == 3:
                    transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                all_transformations.append((transformed_image, (flip_horizontal, flip_vertical, rotation_type)))

    # Randomly select K transformations
    for _ in range(n):
        image, trans = random.choice(all_transformations)
        image = data_augmentation_image(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta)
        transformed_images.append(image)
        transformations_applied.append(trans)

    return transformed_images, transformations_applied



def restore_random_transformations(transformed_images, transformations_applied):
    '''
    Reverse transformations on a set of images based on the specified transformation parameters.

    Args:
        transformed_images (list): List of transformed images to be restored.
        transformations_applied (list of tuples): List of transformation parameters applied to each image, where each tuple contains flip_horizontal, flip_vertical, and rotation_type indicators.

    Returns:
        list: A list of images restored to their original state before the transformations were applied.
    '''

    restored_images = []

    for i in range(len(transformed_images)):
        flip_horizontal, flip_vertical, rotation_type = transformations_applied[i]
        transformed_image = transformed_images[i]

        if rotation_type == 1:
            transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_type == 2:
            transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_180)
        elif rotation_type == 3:
            transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_90_CLOCKWISE)

        if flip_vertical == 1:
            transformed_image = cv2.flip(transformed_image, 1)

        if flip_horizontal == 1:
            transformed_image = cv2.flip(transformed_image, 0)

        restored_images.append(transformed_image)

    return restored_images




def get_IoU_binary(gt, pred):
    '''
    Calculate the Intersection over Union (IoU) for binary images.

    Args:
        gt (np.array): Ground truth binary mask.
        pred (np.array): Predicted binary mask.

    Returns:
        float: The IoU score.
    '''
    
    mask_gt = np.array(gt)
    mask_pred = np.array(pred)

    intersection = np.logical_and(mask_gt, mask_pred).sum()

    union = np.logical_or(mask_gt, mask_pred).sum()

    iou = intersection / (union+1e-7)

    return iou


def get_IoU_multi_unique(pred, gt):
    '''
    Calculate the mean Intersection over Union (IoU) for multiclass segmentation.

    Args:
        pred (np.array): Predicted segmentation mask.
        gt (np.array): Ground truth segmentation mask.

    Returns:
        float: The mean IoU score across all unique classes in the ground truth mask.
    '''
    
    unique_classes = np.unique(gt)
    iou_list = []
    for i in unique_classes:
        temp_gt = np.array(gt == i, dtype=np.float32)
        temp_pred = np.array(pred == i, dtype=np.float32)

        intersection = np.logical_and(temp_gt, temp_pred).sum()
        union = np.logical_or(temp_gt, temp_pred).sum()

        iou = intersection / (union+1e-7)
        iou_list.append(iou)

    mean_iou = sum(iou_list) / len(unique_classes)
    return mean_iou



def pixel_accuracy(pred_mask, gt_mask):
    '''
    Calculate the pixel accuracy between the predicted mask and ground truth mask.

    Args:
        pred_mask (np.array): Predicted segmentation mask.
        gt_mask (np.array): Ground truth segmentation mask.

    Returns:
        float: The pixel accuracy score.
    '''
    
    correct_pixels = np.sum(pred_mask == gt_mask)
    total_pixels = np.prod(gt_mask.shape)
    return correct_pixels / total_pixels


def dice_score_numpy_binary(gt, pred, smooth=1, threshold=128):
    '''
    Calculate the Dice Score for binary segmentation masks.

    Args:
        gt (np.array): Ground truth binary mask.
        pred (np.array): Predicted binary mask.
        smooth (float, optional): Smoothing factor to avoid division by zero. Defaults to 1.
        threshold (int, optional): Threshold to binarize masks. Defaults to 128.

    Returns:
        float: The Dice Score.
    '''

    # Threshold the masks to convert them to binary
    gt = (gt >= threshold).astype(np.float32)
    pred = (pred >= threshold).astype(np.float32)

    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred)

    dice_coeff = (2 * intersection + smooth) / (union + smooth)
    dice_score = dice_coeff

    return dice_score


def create_pseudo_labels_model_ensemble_ISIC_2018(models, images_path, main_output_path, h, w, c, rgb=True,  threshold=0.5):
    '''
    Create pseudo labels for ISIC 2018 using an ensemble of models.

    This function reads images, processes them, and uses an ensemble of models to generate pseudo labels, saving both the images and the pseudo labels in a specified output path.

    Args:
        models (list): List of models for ensemble prediction.
        images_path (str): Directory containing input images.
        main_output_path (str): Main directory to save the processed images and pseudo labels.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

    Returns:
        None
    '''

    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)
    
    for imagename in tqdm(os.listdir(images_path)):
    
        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image
    
        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))

        ensemble_prediction = get_model_ensemble_prediction_ISIC_2018(models, prepared_image, h, w, threshold)    
    
        cv2.imwrite(os.path.join(images_path_out, imagename), image)
        cv2.imwrite(os.path.join(masks_path_out, imagename), ensemble_prediction)





def create_pseudo_labels_model_ensemble_hela(models, bf_images_path, main_output_path, h, w, c):
    '''
    Create pseudo labels for the HeLa dataset using an ensemble of models.

    Args:
        models (list): List of models for ensemble prediction.
        bf_images_path (str): Directory containing brightfield images.
        main_output_path (str): Main directory to save the processed images and pseudo labels.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.

    Returns:
        None
    '''

    bf_images_path_out = os.path.join(main_output_path, 'brightfield')
    alive_masks_path_out = os.path.join(main_output_path, 'alive')
    dead_masks_path_out = os.path.join(main_output_path, 'dead')
    pos_masks_path_out = os.path.join(main_output_path, 'mod_position')

    
    os.makedirs(bf_images_path_out, exist_ok=True)
    os.makedirs(alive_masks_path_out, exist_ok=True)
    os.makedirs(dead_masks_path_out, exist_ok=True)
    os.makedirs(pos_masks_path_out, exist_ok=True)
    
    for imagename in tqdm(os.listdir(bf_images_path)):
    
        bf_image = cv2.imread(os.path.join(bf_images_path, imagename), 0)
    
        prepared_image = (np.array((bf_image).reshape(-1, h, w, c), dtype=np.uint8))

        pred_alive_mask, pred_dead_mask, pred_pos_mask = get_model_ensemble_prediction_hela_soft(models, prepared_image)    
    
        cv2.imwrite(os.path.join(bf_images_path_out, imagename), bf_image)
        cv2.imwrite(os.path.join(alive_masks_path_out, imagename), pred_alive_mask)
        cv2.imwrite(os.path.join(dead_masks_path_out, imagename), pred_dead_mask)
        cv2.imwrite(os.path.join(pos_masks_path_out, imagename), pred_pos_mask)




def create_pseudo_labels_model_ensemble_multiclass(models, images_path, main_output_path, h, w, c, rgb=True):
    '''
    Generate pseudo labels for multiclass segmentation datasets using an ensemble of models.

    Args:
        models (list): List of models for ensemble prediction.
        images_path (str): Directory containing input images.
        main_output_path (str): Main directory to save the processed images and pseudo labels.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.

    Returns:
        None
    '''

    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)
    
    for imagename in tqdm(os.listdir(images_path)):
    
        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image
    
        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))

        ensemble_prediction = get_model_ensemble_prediction_multiclass_soft(models, prepared_image)    
    
        cv2.imwrite(os.path.join(images_path_out, imagename), image)
        cv2.imwrite(os.path.join(masks_path_out, imagename), ensemble_prediction)



def create_pseudo_labels_input_ensemble_ISIC_2018(model, images_path, main_output_path, h, w, c, n=2, rgb=True, use_n_rnd_transformations=True, threshold=0.5):
    '''
    Generate pseudo labels for ISIC 2018 using input ensemble predictions from a single model.

    Args:
        model: The model used for predictions.
        images_path (str): Directory containing input images.
        main_output_path (str): Main directory to save the processed images and pseudo labels.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        n (int, optional): Number of random transformations for ensemble. Defaults to 2.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.
        use_n_rnd_transformations (bool, optional): Flag to use N random transformations. Defaults to True.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

    Returns:
        None
    '''

    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)

    
    for imagename in tqdm(os.listdir(images_path)):
    
        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image
    
        ensemble_prediction = get_input_ensemble_prediction_ISIC_2018(model, input_image, h, w, c, threshold, n, use_n_rnd_transformations=use_n_rnd_transformations)    
    
        kernel = np.ones((5, 5), 'uint8')
        eroded_ensemble_prediction = cv2.erode(ensemble_prediction, kernel, iterations=1)

        predsize = np.sum(eroded_ensemble_prediction)

        if predsize > 0:
            cv2.imwrite(os.path.join(images_path_out, imagename), image)
            cv2.imwrite(os.path.join(masks_path_out, imagename), ensemble_prediction)



def create_pseudo_labels_input_ensemble_hela(model, bf_images_path, main_output_path, h, w, c, n=2, use_soft_voting=False):
    '''
    Generate pseudo labels for the HeLa dataset using input ensemble predictions for a single model.

    Args:
        model: The model used for predictions.
        bf_images_path (str): Directory containing brightfield images.
        main_output_path (str): Main directory to save the processed images and pseudo labels.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        n (int, optional): Number of transformations for ensemble. Defaults to 2.
        use_soft_voting (bool, optional): Flag to use soft voting for ensemble predictions. Defaults to False.

    Returns:
        None
    '''

    bf_images_path_out = os.path.join(main_output_path, 'brightfield')
    alive_masks_path_out = os.path.join(main_output_path, 'alive')
    dead_masks_path_out = os.path.join(main_output_path, 'dead')
    pos_masks_path_out = os.path.join(main_output_path, 'mod_position')

    
    os.makedirs(bf_images_path_out, exist_ok=True)
    os.makedirs(alive_masks_path_out, exist_ok=True)
    os.makedirs(dead_masks_path_out, exist_ok=True)
    os.makedirs(pos_masks_path_out, exist_ok=True)
    
    for imagename in tqdm(os.listdir(bf_images_path)):
    
        bf_image = cv2.imread(os.path.join(bf_images_path, imagename), 0)
    
        #prepared_image = (np.array((bf_image).reshape(-1, h, w, c), dtype=np.uint8))
        
        if use_soft_voting:
            pred_alive_mask, pred_dead_mask, pred_pos_mask = get_input_ensemble_prediction_hela_soft(model, bf_image, h, w, c, n)    
        else:
            pred_alive_mask, pred_dead_mask, pred_pos_mask = get_input_ensemble_prediction_hela_hard(model, bf_image, h, w, c, n)    
    
        cv2.imwrite(os.path.join(bf_images_path_out, imagename), bf_image)
        cv2.imwrite(os.path.join(alive_masks_path_out, imagename), pred_alive_mask)
        cv2.imwrite(os.path.join(dead_masks_path_out, imagename), pred_dead_mask)
        cv2.imwrite(os.path.join(pos_masks_path_out, imagename), pred_pos_mask)



def create_pseudo_labels_input_ensemble_multiclass(model, images_path, main_output_path, h, w, c, n=2, rgb=True):
    '''
    Generate pseudo labels for multiclass segmentation datasets using input ensemble predictions from a single model.

    Args:
        model: The model used for predictions.
        images_path (str): Directory containing input images.
        main_output_path (str): Main directory to save the processed images and pseudo labels.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        n (int, optional): Number of transformations for ensemble. Defaults to 2.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.

    Returns:
        None
    '''

    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)

    
    for imagename in tqdm(os.listdir(images_path)):
    
        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image
    
        ensemble_prediction = get_input_ensemble_prediction_multiclass_soft(model, input_image, h, w, c, n)      
    
        cv2.imwrite(os.path.join(images_path_out, imagename), image)
        cv2.imwrite(os.path.join(masks_path_out, imagename), ensemble_prediction)



def get_input_ensemble_prediction_ISIC_2018(model, image, h, w, c, threshold, n=2, max_blur=3, max_noise=25, brightness_range_alpha=(0.5, 1.5), brightness_range_beta=(-25, 25), use_n_rnd_transformations=True):
    '''
    Generate an ensemble prediction for an image using a model and input transformations.

    Args:
        model: The model used for prediction.
        image (np.array): The input image.
        h (int): Height of the image.
        w (int): Width of the image.
        c (int): Number of channels in the image.
        threshold (float): Threshold for binary classification.
        n (int, optional): Number of random transformations if 'use_n_rnd_transformations' is True. Defaults to 2.
        max_blur (int, optional): Maximum blur to apply. Defaults to 3.
        max_noise (int, optional): Maximum noise to add. Defaults to 25.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.5, 1.5).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-25, 25).
        use_n_rnd_transformations (bool, optional): Flag to use N random transformations instead of all. Defaults to True.

    Returns:
        np.array: The predicted mask after combining the predictions from the transformed images.
    '''
    
    if use_n_rnd_transformations == True:
        transformed_images, transformations_applied = generate_random_transformations(image, n, max_blur, max_noise, brightness_range_alpha, brightness_range_beta)
    else:
        transformed_images = generate_all_transformations(image)
    
    prepared_images = [np.array((image).reshape(-1, h, w, c), dtype=np.uint8) for image in transformed_images]
    prepared_images = np.concatenate(prepared_images, axis=0)

    with contextlib.redirect_stdout(io.StringIO()):
        masks = model.predict(prepared_images)
        masks[masks >= threshold] = 1
        masks[masks < threshold] = 0
        masks_2d = masks[:, :, :, 0]
        masks_2d = masks_2d.astype(np.uint8)

    masks_2d_list = [np.array(mask) for mask in masks_2d]

    if use_n_rnd_transformations == True:
        restored_masks = restore_random_transformations(masks_2d_list, transformations_applied)
    else:
        restored_masks = restore_all_transformations(masks_2d_list)

    mask_sum = np.sum(restored_masks, axis=0)
    pred_mask = np.zeros_like(mask_sum)
    pred_mask[mask_sum < len(transformed_images)] = 0
    pred_mask[mask_sum >= len(transformed_images)] = 255

    return pred_mask.astype(np.uint8)





def get_input_ensemble_prediction_multiclass(model, image, h, w, c, n=2, max_blur=1, max_noise=15, brightness_range_alpha=(0.7, 1.3), brightness_range_beta=(-15, 15)):
    '''
    Generate a multiclass dataset ensemble prediction for an image using a model with input transformations.

    Args:
        model: The model used for prediction.
        image (np.array): The input image.
        h (int): Height of the image.
        w (int): Width of the image.
        c (int): Number of channels in the image.
        n (int, optional): Number of transformations for ensemble. Defaults to 2.
        max_blur (int, optional): Maximum blur to apply. Defaults to 1.
        max_noise (int, optional): Maximum noise to add. Defaults to 15.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.7, 1.3).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-15, 15).

    Returns:
        np.array: The predicted multiclass mask.
    '''
    transformed_images= []
    for i in range(n+1):
        image = data_augmentation_image(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta)
        transformed_images.append(image)
    
    prepared_images = [np.array((image).reshape(-1, h, w, c), dtype=np.uint8) for image in transformed_images]
    prepared_images = np.concatenate(prepared_images, axis=0)

    with contextlib.redirect_stdout(io.StringIO()):
        masks = model.predict(prepared_images)

    restored_masks = [np.array(np.argmax(mask, axis=-1)) for mask in masks]

    reshaped_masks = np.stack(restored_masks, axis=-1)
    most_common = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=-1, arr=reshaped_masks)

    return most_common.astype(np.uint8)



def get_input_ensemble_prediction_hela_soft(model, image, h, w, c, n=2, max_blur=1, max_noise=15, brightness_range_alpha=(0.7, 1.3), brightness_range_beta=(-15, 15), threshold=0.5, max_pos_circle_size=8, min_pos_circle_size=3):
    '''
    Obtain input ensemble predictions for the HeLa dataset by performing multiple data augmentations on a single input image.
    
    Args:
    - model: The trained prediction model.
    - image (numpy array): The original input image.
    - h (int): Height of the image.
    - w (int): Width of the image.
    - c (int): Number of channels in the image (e.g., 3 for RGB).
    - n (int, optional): Number of times the input image will be augmented. Defaults to 2.
    - max_blur (int, optional): Maximum blur that can be applied during augmentation. Defaults to 1.
    - max_noise (int, optional): Maximum noise that can be added during augmentation. Defaults to 15.
    - brightness_range_alpha (tuple, optional): Range of multiplicative factors to change brightness during augmentation. Defaults to (0.7, 1.3).
    - brightness_range_beta (tuple, optional): Range of additive factors to change brightness during augmentation. Defaults to (-15, 15).
    - threshold (float, optional): Threshold value for classification. Defaults to 0.5.
    
    Returns:
    - tuple of numpy arrays: Binary masks (values 0 or 255) for each of the three classes: alive, dead, and pos.
    '''

    transformed_images= []
    for i in range(n+1):
        image = data_augmentation_image(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta)
        transformed_images.append(image)
    
    prepared_images = [np.array((image).reshape(-1, h, w, c), dtype=np.uint8) for image in transformed_images]
    prepared_images = np.concatenate(prepared_images, axis=0)
    
    # Extract image dimensions
    img_height, img_width = prepared_images[0].shape[1], prepared_images[0].shape[0]
    
    # Initialize the sum arrays for each class
    pred_sum_alive = np.zeros((img_width, img_height))
    pred_sum_dead = np.zeros((img_width, img_height))
    pred_sum_pos = np.zeros((img_width, img_height))

    with contextlib.redirect_stdout(io.StringIO()):
        output_data = model.predict(prepared_images)
        
        # Sum predictions over all images
        for pred in output_data:
            alive, dead, pos = cv2.split(pred)
            pred_sum_alive += alive
            pred_sum_dead += dead
            pred_sum_pos += pos

    # Average the probabilities
    avg_alive = pred_sum_alive / len(prepared_images)
    avg_dead = pred_sum_dead / len(prepared_images)
    avg_pos = pred_sum_pos / len(prepared_images)

    # Apply threshold and set to values 0 or 255
    final_alive_mask = (avg_alive > threshold).astype(np.uint8) * 255
    final_dead_mask = (avg_dead > threshold).astype(np.uint8) * 255
    temp_pos_mask = (avg_pos > threshold).astype(np.uint8) * 255

    positions = get_pos_contours(temp_pos_mask)

    final_pos_mask = np.zeros((img_height, img_width, 3), np.uint8)
               
    for pos in positions:
        if len(positions) > 1:
            min_dist = get_min_dist(pos, positions)
        else:
            min_dist = 99

        circle_size = int(min_dist // 4)
        circle_size = max(min(circle_size, max_pos_circle_size), min_pos_circle_size)
        cv2.circle(final_pos_mask, (pos[0], pos[1]), circle_size, (255, 255, 255), -1)

    return final_alive_mask, final_dead_mask, final_pos_mask


def get_input_ensemble_prediction_hela_hard(model, image, h, w, c, n=2, max_blur=1, max_noise=15, brightness_range_alpha=(0.7, 1.3), brightness_range_beta=(-15, 15), threshold=0.5, max_pos_circle_size=8, min_pos_circle_size=3):
    '''
    Obtain input ensemble predictions for the HeLa dataset by performing multiple data augmentations on a single input image.
    
    Args:
    - model: The trained prediction model.
    - image (numpy array): The original input image.
    - h (int): Height of the image.
    - w (int): Width of the image.
    - c (int): Number of channels in the image (e.g., 3 for RGB).
    - n (int, optional): Number of times the input image will be augmented. Defaults to 2.
    - max_blur (int, optional): Maximum blur that can be applied during augmentation. Defaults to 1.
    - max_noise (int, optional): Maximum noise that can be added during augmentation. Defaults to 15.
    - brightness_range_alpha (tuple, optional): Range of multiplicative factors to change brightness during augmentation. Defaults to (0.7, 1.3).
    - brightness_range_beta (tuple, optional): Range of additive factors to change brightness during augmentation. Defaults to (-15, 15).
    - threshold (float, optional): Threshold value for classification. Defaults to 0.5.
    
    Returns:
    - tuple of numpy arrays: Binary masks (values 0 or 255) for each of the three classes: alive, dead, and pos.
    '''

    transformed_images= []
    for i in range(n+1):
        image = data_augmentation_image(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta)
        transformed_images.append(image)
    
    prepared_images = [np.array((image).reshape(-1, h, w, c), dtype=np.uint8) for image in transformed_images]
    prepared_images = np.concatenate(prepared_images, axis=0)
    
    # Extract image dimensions
    img_height, img_width = prepared_images[0].shape[1], prepared_images[0].shape[0]
    
    # Initialize the sum arrays for each class
    pred_sum_alive = np.zeros((img_width, img_height))
    pred_sum_dead = np.zeros((img_width, img_height))
    pred_sum_pos = np.zeros((img_width, img_height))

    with contextlib.redirect_stdout(io.StringIO()):
        output_data = model.predict(prepared_images)
        
        # Sum predictions over all images
        for pred in output_data:
            alive, dead, pos = cv2.split(pred)
            pred_sum_alive += (alive > threshold).astype(np.uint8)
            pred_sum_dead += (dead > threshold).astype(np.uint8)
            pred_sum_pos += (pos > threshold).astype(np.uint8)

    final_alive_mask = np.where(pred_sum_alive == len(prepared_images), 255, 0).astype(np.uint8)
    final_dead_mask = np.where(pred_sum_dead == len(prepared_images), 255, 0).astype(np.uint8)
    temp_pos_mask = np.where(pred_sum_pos == len(prepared_images), 255, 0).astype(np.uint8)

    positions = get_pos_contours(temp_pos_mask)

    final_pos_mask = np.zeros((img_height, img_width, 3), np.uint8)
               
    for pos in positions:
        if len(positions) > 1:
            min_dist = get_min_dist(pos, positions)
        else:
            min_dist = 99

        circle_size = int(min_dist // 4)
        circle_size = max(min(circle_size, max_pos_circle_size), min_pos_circle_size)
        cv2.circle(final_pos_mask, (pos[0], pos[1]), circle_size, (255, 255, 255), -1)

    return final_alive_mask, final_dead_mask, final_pos_mask




def get_input_ensemble_prediction_multiclass_soft(model, image, h, w, c, n=2, max_blur=1, max_noise=15, brightness_range_alpha=(0.7, 1.3), brightness_range_beta=(-15, 15)):
    '''
    Generate a soft multiclass ensemble prediction for an image using a model and input transformations.

    Args:
        model: The model used for prediction.
        image (np.array): The input image.
        h (int): Height of the image.
        w (int): Width of the image.
        c (int): Number of channels in the image.
        n (int, optional): Number of transformations for ensemble. Defaults to 2.
        max_blur (int, optional): Maximum blur to apply. Defaults to 1.
        max_noise (int, optional): Maximum noise to add. Defaults to 15.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.7, 1.3).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-15, 15).

    Returns:
        np.array: The predicted multiclass mask based on soft voting.
    '''

    transformed_images= []
    for i in range(n+1):
        image = data_augmentation_image(image, max_blur, max_noise, brightness_range_alpha, brightness_range_beta)
        transformed_images.append(image)
    
    prepared_images = [np.array((image).reshape(-1, h, w, c), dtype=np.uint8) for image in transformed_images]
    prepared_images = np.concatenate(prepared_images, axis=0)

    # Capture the model's softmax outputs for all augmented images
    with contextlib.redirect_stdout(io.StringIO()):
        masks_prob = model.predict(prepared_images)

    # Average the softmax outputs
    averaged_mask_prob = np.mean(masks_prob, axis=0)

    # Convert averaged probabilities into class labels
    final_pred_mask = np.argmax(averaged_mask_prob, axis=-1)

    return final_pred_mask.astype(np.uint8)





def get_model_ensemble_prediction_ISIC_2018(models, prepared_image, image_width, image_height, threshold):
    '''
    Generate a binary mask prediction using an ensemble of models for the ISIC 2018 dataset.

    Args:
        models (list): List of models for ensemble prediction.
        prepared_image (np.array): Preprocessed image ready for prediction.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        threshold (float): Threshold for binary classification.

    Returns:
        np.array: The ensemble prediction mask.
    '''
    
    pred_sum = np.zeros((image_width, image_height))

    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):
            mask_pred = model.predict([prepared_image])
            mask_pred = np.reshape(mask_pred, (image_width, image_height))
            mask_pred_uint = ((mask_pred > threshold)).astype(np.uint8)
            pred_sum += mask_pred_uint

    pred_sum[pred_sum < len(models)] = 0
    pred_sum[pred_sum >= len(models)] = 255

    return pred_sum


def get_model_ensemble_prediction_multiclass_hard(models, prepared_image):
    '''
    Generate a multiclass mask prediction using a hard voting ensemble of models.

    Args:
        models (list): List of models for ensemble prediction.
        prepared_image (np.array): Preprocessed image ready for prediction.

    Returns:
        np.array: The ensemble prediction multiclass mask.
    '''

    pred_masks = []

    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):
            mask_pred = model.predict([prepared_image])
            mask_pred = np.argmax(mask_pred, axis=-1)

            pred_masks.append(mask_pred)

    pred_masks = np.stack(pred_masks, axis=0)

    # Perform the intersection operation
    final_pred_mask = np.where(np.all(pred_masks == pred_masks[0, :], axis=0), pred_masks[0, :], 0)

    final_pred_mask = np.squeeze(final_pred_mask).astype(np.uint8)

    return final_pred_mask



def get_model_ensemble_prediction_hela_soft(models, prepared_image, threshold=0.5, max_pos_circle_size=8, min_pos_circle_size=3):
    '''
    Get ensemble predictions for the HeLa dataset.
    
    Args:
    - models (list): List of trained models.
    - prepared_image (numpy array): The input image for prediction.
    - threshold (float, optional): Threshold value for classification. Defaults to 0.5.
    
    Returns:
    - tuple of numpy arrays: Binary masks (values 0 or 255) for each of the three classes: alive, dead, and pos.
    '''
    
    # Extract image dimensions
    img_height, img_width = prepared_image.shape[2], prepared_image.shape[1]
    
    # Initialize the sum arrays for each class
    pred_sum_alive = np.zeros((img_width, img_height))
    pred_sum_dead = np.zeros((img_width, img_height))
    pred_sum_pos = np.zeros((img_width, img_height))

    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):
            output_data = model.predict([prepared_image])
            
            # Split the prediction data into their respective classes
            alive, dead, pos = cv2.split(output_data[0])

            # Add the probabilities for each class
            pred_sum_alive += alive
            pred_sum_dead += dead
            pred_sum_pos += pos

    # Average the probabilities
    avg_alive = pred_sum_alive / len(models)
    avg_dead = pred_sum_dead / len(models)
    avg_pos = pred_sum_pos / len(models)

    # Apply threshold and set to values 0 or 255
    final_alive_mask = (avg_alive > threshold).astype(np.uint8) * 255
    final_dead_mask = (avg_dead > threshold).astype(np.uint8) * 255
    temp_pos_mask = (avg_pos > threshold).astype(np.uint8) * 255

    positions = get_pos_contours(temp_pos_mask)

    final_pos_mask = np.zeros((img_height, img_width, 3), np.uint8)
               
    for pos in positions:
        if len(positions) > 1:
            min_dist = get_min_dist(pos, positions)
        else:
            min_dist = 99

        circle_size = int(min_dist // 4)
        circle_size = max(min(circle_size, max_pos_circle_size), min_pos_circle_size)
        cv2.circle(final_pos_mask, (pos[0], pos[1]), circle_size, (255, 255, 255), -1)

    return final_alive_mask, final_dead_mask, final_pos_mask




def get_model_ensemble_prediction_multiclass_soft(models, prepared_image):
    '''
    Generate a multiclass mask prediction using a soft voting ensemble of models.

    Args:
        models (list): List of models for ensemble prediction.
        prepared_image (np.array): Preprocessed image ready for prediction.

    Returns:
        np.array: The ensemble prediction multiclass mask based on soft voting.
    '''

    pred_masks_prob = []

    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):
            mask_pred_prob = model.predict([prepared_image])
            pred_masks_prob.append(mask_pred_prob)

    # Stack predictions together for easier computation
    pred_masks_prob = np.stack(pred_masks_prob, axis=0)

    final_pred_mask_prob = np.mean(pred_masks_prob, axis=0)

    # Now, we apply np.argmax to generate final prediction mask
    final_pred_mask = np.argmax(final_pred_mask_prob, axis=-1)

    final_pred_mask = np.squeeze(final_pred_mask).astype(np.uint8)

    return final_pred_mask




def create_augment_images_and_masks_ISIC_2018(images_path, masks_path, main_output_path, num_images=9, copy_org=True, brightness_range_alpha=(0.5, 1.5), brightness_range_beta=(-25, 25), max_blur=3, max_noise=25, free_rotation=True):
    '''
    Augment images and masks in the ISIC 2018 dataset and save them to a specified output path.

    Args:
        images_path (str): Directory containing original images.
        masks_path (str): Directory containing corresponding masks.
        main_output_path (str): Directory to save augmented images and masks.
        num_images (int, optional): Number of augmented images to create for each original image. Defaults to 9.
        copy_org (bool, optional): Flag to copy original images to output path. Defaults to True.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.5, 1.5).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-25, 25).
        max_blur (int, optional): Maximum blur to apply. Defaults to 3.
        max_noise (int, optional): Maximum noise to add. Defaults to 25.
        free_rotation (bool, optional): Flag to apply free rotation. Defaults to True.

    Returns:
        None
    '''
    
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)

    if copy_org == True:
        for imagename in tqdm(os.listdir(images_path)):
            shutil.copy(os.path.join(images_path, imagename), os.path.join(images_path_out, imagename))
            shutil.copy(os.path.join(masks_path, imagename), os.path.join(masks_path_out, imagename))
    
    for imagename in tqdm(os.listdir(images_path)):
    
        image = cv2.imread(os.path.join(images_path, imagename))
        mask = cv2.imread(os.path.join(masks_path, imagename))

        for n in range(0,num_images):

            aug_image, aug_mask = augment_image_and_mask(image, mask, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation)        
    
            cv2.imwrite(os.path.join(images_path_out, f'{imagename[:-4]}_aug_{n}.png'), aug_image)
            cv2.imwrite(os.path.join(masks_path_out, f'{imagename[:-4]}_aug_{n}.png'), aug_mask)




def create_augment_images_and_masks_hela(main_input_path, main_output_path, num_images=9, copy_org=True, free_rotation=True, brightness_range_alpha=(0.7, 1.3), brightness_range_beta=(-15, 15), max_blur=3, max_noise=25):
    '''
    Augment HeLa cell images and masks and save them to a specified output path.

    Args:
        main_input_path (str): Directory containing original HeLa cell images and masks.
        main_output_path (str): Directory to save augmented images and masks.
        num_images (int, optional): Number of augmented images to create for each original image. Defaults to 9.
        copy_org (bool, optional): Flag to copy original images and masks to output path. Defaults to True.
        free_rotation (bool, optional): Flag to apply free rotation. Defaults to True.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.7, 1.3).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-15, 15).
        max_blur (int, optional): Maximum blur to apply. Defaults to 3.
        max_noise (int, optional): Maximum noise to add. Defaults to 25.

    Returns:
        None
    '''
    
    brightfield_path_in = os.path.join(main_input_path, 'brightfield')
    alive_path_in = os.path.join(main_input_path, 'alive')
    dead_path_in = os.path.join(main_input_path, 'dead')
    pos_path_in = os.path.join(main_input_path, 'mod_position')

    brightfield_path_out = os.path.join(main_output_path, 'brightfield')
    alive_path_out = os.path.join(main_output_path, 'alive')
    dead_path_out = os.path.join(main_output_path, 'dead')
    pos_path_out = os.path.join(main_output_path, 'mod_position')
    
    os.makedirs(brightfield_path_out, exist_ok=True)
    os.makedirs(alive_path_out, exist_ok=True)
    os.makedirs(dead_path_out, exist_ok=True)
    os.makedirs(pos_path_out, exist_ok=True)

    if copy_org == True:
        for imagename in tqdm(os.listdir(brightfield_path_in)):
            shutil.copy(os.path.join(brightfield_path_in, imagename), os.path.join(brightfield_path_out, imagename))
            shutil.copy(os.path.join(alive_path_in, imagename), os.path.join(alive_path_out, imagename))
            shutil.copy(os.path.join(dead_path_in, imagename), os.path.join(dead_path_out, imagename))
            shutil.copy(os.path.join(pos_path_in, imagename), os.path.join(pos_path_out, imagename))
    
    for imagename in tqdm(os.listdir(brightfield_path_in)):
    
        bf_image = cv2.imread(os.path.join(brightfield_path_in, imagename))
        alive_mask = cv2.imread(os.path.join(alive_path_in, imagename))
        dead_mask = cv2.imread(os.path.join(dead_path_in, imagename))
        pos_mask = cv2.imread(os.path.join(pos_path_in, imagename))

        masks = []
        masks.append(alive_mask)
        masks.append(dead_mask)
        masks.append(pos_mask)


        for n in range(0,num_images):

            aug_bf_image, aug_masks = augment_image_and_masks(bf_image, masks, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation)        
    
            cv2.imwrite(os.path.join(brightfield_path_out, f'{imagename[:-4]}_aug_{n}.png'), aug_bf_image)
            cv2.imwrite(os.path.join(alive_path_out, f'{imagename[:-4]}_aug_{n}.png'), aug_masks[0])
            cv2.imwrite(os.path.join(dead_path_out, f'{imagename[:-4]}_aug_{n}.png'), aug_masks[1])
            cv2.imwrite(os.path.join(pos_path_out, f'{imagename[:-4]}_aug_{n}.png'), aug_masks[2])



def create_augment_images_and_masks_multiclass(images_path, masks_path, main_output_path, num_images=9, copy_org=True, free_rotation=False, brightness_range_alpha=(0.5, 1.5), brightness_range_beta=(-25, 25), max_blur=3, max_noise=25):
    '''
    Augment multiclass images and masks and save them to a specified output path.

    Args:
        images_path (str): Directory containing original images.
        masks_path (str): Directory containing corresponding masks.
        main_output_path (str): Directory to save augmented images and masks.
        num_images (int, optional): Number of augmented images to create for each original image. Defaults to 9.
        copy_org (bool, optional): Flag to copy original images and masks to output path. Defaults to True.
        free_rotation (bool, optional): Flag to apply free rotation. Defaults to False.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.5, 1.5).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-25, 25).
        max_blur (int, optional): Maximum blur to apply. Defaults to 3.
        max_noise (int, optional): Maximum noise to add. Defaults to 25.

    Returns:
        None
    '''
    
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)

    if copy_org == True:
        for imagename in tqdm(os.listdir(images_path)):
            shutil.copy(os.path.join(images_path, imagename), os.path.join(images_path_out, imagename))
            shutil.copy(os.path.join(masks_path, imagename), os.path.join(masks_path_out, imagename))
    
    for imagename in tqdm(os.listdir(images_path)):
    
        image = cv2.imread(os.path.join(images_path, imagename))
        mask = cv2.imread(os.path.join(masks_path, imagename))

        for n in range(0,num_images):

            aug_image, aug_mask = augment_image_and_mask(image, mask, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation)        
    
            cv2.imwrite(os.path.join(images_path_out, f'{imagename[:-4]}_aug_{n}.png'), aug_image)
            cv2.imwrite(os.path.join(masks_path_out, f'{imagename[:-4]}_aug_{n}.png'), aug_mask)





def augment_image_and_masks(image, masks, brightness_range_alpha = (0.5, 1.5), brightness_range_beta = (-25, 25), max_blur=3, max_noise=25, free_rotation=True):
    '''
    Apply augmentation techniques to an image and its corresponding masks.

    Args:
        image (np.array): The input image to be augmented.
        masks (list of np.array): List of masks corresponding to the image.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.5, 1.5).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-25, 25).
        max_blur (int, optional): Maximum blur to apply. Defaults to 3.
        max_noise (int, optional): Maximum noise to add. Defaults to 25.
        free_rotation (bool, optional): Flag to apply random rotation and flips. Defaults to True.

    Returns:
        tuple: A tuple containing the augmented image and a list of augmented masks.
    '''
    
    augmented_masks = []
    
    if free_rotation:
        if random.randint(0,1) == 1:
            image = cv2.flip(image, 0)
            masks = [cv2.flip(mask, 0) for mask in masks]

    if random.randint(0,1) == 1:
        image = cv2.flip(image, 1)
        masks = [cv2.flip(mask, 1) for mask in masks]

    if free_rotation:
        rndint = random.randint(0,3)
        if rndint > 0:
            if rndint == 1:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                masks = [cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE) for mask in masks]

            if rndint == 2:
                image = cv2.rotate(image, cv2.ROTATE_180)
                masks = [cv2.rotate(mask, cv2.ROTATE_180) for mask in masks]

            if rndint == 3:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                masks = [cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE) for mask in masks]

    brightness_alpha  = np.random.uniform(brightness_range_alpha[0], brightness_range_alpha[1])
    brightness_beta  = np.random.uniform(brightness_range_beta[0], brightness_range_beta[1])

    if random.randint(0,1) == 1:
        image = cv2.convertScaleAbs(image, alpha=brightness_alpha, beta=brightness_beta)

    image_output = add_noise_and_blur(image, max_blur, max_noise)

    return image_output, masks


def augment_image_and_mask(image, mask, brightness_range_alpha = (0.5, 1.5), brightness_range_beta = (-25, 25), max_blur=3, max_noise=25, free_rotation=True):
    '''
    Apply augmentation techniques to an image and its corresponding mask.
    
    Args:
        image (np.array): The input image to be augmented.
        mask (np.array): The corresponding mask to be augmented.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.5, 1.5).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-25, 25).
        max_blur (int, optional): Maximum blur to apply. Defaults to 3.
        max_noise (int, optional): Maximum noise to add. Defaults to 25.
        free_rotation (bool, optional): Flag to apply random rotation and flips. Defaults to True.
    
    Returns:
        tuple: A tuple containing the augmented image and mask.
    '''
    
    if free_rotation:
        if random.randint(0,1) == 1:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

    if random.randint(0,1) == 1:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    if free_rotation:
        rndint = random.randint(0,3)
        if rndint > 0:
            if rndint == 1:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

            if rndint == 2:
                image = cv2.rotate(image, cv2.ROTATE_180)
                mask = cv2.rotate(mask, cv2.ROTATE_180)

            if rndint == 3:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

    brightness_alpha  = np.random.uniform(brightness_range_alpha[0], brightness_range_alpha[1])
    brightness_beta  = np.random.uniform(brightness_range_beta[0], brightness_range_beta[1])

    if random.randint(0,1) == 1:
        image = cv2.convertScaleAbs(image, alpha=brightness_alpha, beta=brightness_beta)

    image_output = add_noise_and_blur(image, max_blur, max_noise)

    return image_output, mask



def create_pseudo_labels_im_ISIC_2018(models, h, w, c, images_path, main_output_path, rgb=True, erode_kernel=5, dilate_kernel=5, block_input=True, block_output=True, filter_bad_predictions=True):

    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    im_path_out = os.path.join(main_output_path, 'im')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)
    os.makedirs(im_path_out, exist_ok=True)

    im_sizes = {}
    
    for imagename in tqdm(os.listdir(images_path)):

        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image
    
        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))

        pred_mask_sum, im, im_size, predsize = get_im_prediction_binary(models, prepared_image, THRESHOLD)

        im_sizes[imagename[:-4]] = im_size

        if erode_kernel > 0:
            kernel = np.ones((erode_kernel, erode_kernel), 'uint8')
            im = cv2.erode(im, kernel, iterations=1)

        if dilate_kernel > 0:
            kernel = np.ones((dilate_kernel, dilate_kernel), 'uint8')
            im = cv2.dilate(im, kernel, iterations=1)


        if block_input == True:
            if c == 3:
                image[im > 0] = [0,0,0]
            else:
                image[im > 0] = 0

        if block_output == True:
            pred_mask_sum[im > 0] = 0

        write_images = False

        if filter_bad_predictions:
            if predsize > im_size and predsize > 0:
                write_images = True
        else:
            write_images = True

        if write_images:
            cv2.imwrite(os.path.join(images_path_out, imagename), image)
            cv2.imwrite(os.path.join(masks_path_out, imagename), pred_mask_sum)
        cv2.imwrite(os.path.join(im_path_out, imagename), im)

    mean_im_size = round(sum(im_sizes.values()) / len(im_sizes),0)

    return mean_im_size



def create_pseudo_labels_im_hela(models, h, w, c, images_path, main_output_path, erode_kernel=5, dilate_kernel=5, block_input=True, block_output=True, max_pos_circle_size=8, min_pos_circle_size=3):
    '''
    Create pseudo labels for HeLa dataset images using IM predictions.

    Args:
        models (list): List of models for ensemble prediction.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        images_path (str): Directory containing input images.
        main_output_path (str): Main directory to save the processed images and pseudo labels.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.
        erode_kernel (int, optional): Size of the kernel used for erosion. Defaults to 5.
        dilate_kernel (int, optional): Size of the kernel used for dilation. Defaults to 5.
        block_input (bool, optional): Flag to block certain areas in input images. Defaults to True.
        block_output (bool, optional): Flag to block certain areas in output masks. Defaults to True.
        filter_bad_predictions (bool, optional): Flag to filter out bad predictions. Defaults to True.

    Returns:
        float: The average size of the inconsistency masks.
    '''
    
    brightfield_path_out = os.path.join(main_output_path, 'brightfield')
    alive_path_out = os.path.join(main_output_path, 'alive')
    dead_path_out = os.path.join(main_output_path, 'dead')
    pos_path_out = os.path.join(main_output_path, 'mod_position')
    im_path_out = os.path.join(main_output_path, 'im')
    
    os.makedirs(brightfield_path_out, exist_ok=True)
    os.makedirs(alive_path_out, exist_ok=True)
    os.makedirs(dead_path_out, exist_ok=True)
    os.makedirs(pos_path_out, exist_ok=True)
    os.makedirs(im_path_out, exist_ok=True)


    im_sizes = {}
    
    for imagename in tqdm(os.listdir(images_path)):

        image_gray = cv2.imread(os.path.join(images_path, imagename), 0)

        prepared_image = (np.array((image_gray).reshape(-1, h, w, c), dtype=np.uint8))
    
        final_alive_mask, final_dead_mask, final_pos_mask_raw, combined_im, im_size = get_im_prediction_hela(models, prepared_image)

        im_sizes[imagename[:-4]] = im_size

        if erode_kernel > 0:
            kernel = np.ones((erode_kernel, erode_kernel), 'uint8')
            combined_im = cv2.erode(combined_im, kernel, iterations=1)

            final_alive_mask = dilate_mask(final_alive_mask)
            final_dead_mask = dilate_mask(final_dead_mask)

        if dilate_kernel > 0:
            kernel = np.ones((dilate_kernel, dilate_kernel), 'uint8')
            combined_im = cv2.dilate(combined_im, kernel, iterations=1)

        positions = get_pos_contours(final_pos_mask_raw)
        final_pos_mask = np.zeros((h, w, 3), np.uint8)
        
        
        for pos in positions:
            if len(positions) > 1:
                min_dist = get_min_dist(pos, positions)
            else:
                min_dist = 99

            circle_size = int(min_dist // 4)
            circle_size = max(min(circle_size, max_pos_circle_size), min_pos_circle_size)
            cv2.circle(final_pos_mask, (pos[0], pos[1]), circle_size, (255, 255, 255), -1)
        

        if block_input == True:
            image_gray[combined_im > 0] = 0

        if block_output == True:
            final_alive_mask[combined_im > 0] = 0
            final_dead_mask[combined_im > 0] = 0
            final_pos_mask[combined_im > 0] = 0

        cv2.imwrite(os.path.join(brightfield_path_out, imagename), image_gray)
        cv2.imwrite(os.path.join(alive_path_out, imagename), final_alive_mask)
        cv2.imwrite(os.path.join(dead_path_out, imagename), final_dead_mask)
        cv2.imwrite(os.path.join(pos_path_out, imagename), final_pos_mask)
        cv2.imwrite(os.path.join(im_path_out, imagename), combined_im)

    mean_im_size = round(sum(im_sizes.values()) / len(im_sizes),0)

    return mean_im_size



def create_pseudo_labels_im_multiclass(models, h, w, c, images_path, main_output_path, rgb=True, erode_kernel=5, dilate_kernel=5, block_input=True, block_output=True, filter_unequal_class_pred=False):
    '''
    Create pseudo labels for multiclass segmentation using IM predictions.

    Args:
        models (list): List of models for ensemble prediction.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        images_path (str): Directory containing input images.
        main_output_path (str): Main directory to save the processed images and pseudo labels.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.
        erode_kernel (int, optional): Size of the kernel used for erosion. Defaults to 5.
        dilate_kernel (int, optional): Size of the kernel used for dilation. Defaults to 5.
        block_input (bool, optional): Flag to block certain areas in input images. Defaults to True.
        block_output (bool, optional): Flag to block certain areas in output masks. Defaults to True.
        filter_unequal_class_pred (bool, optional): Flag to filter out predictions where class distributions are unequal. Defaults to False.

    Returns:
        float: The average size of the inconsistency masks.
    '''
    
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    im_path_out = os.path.join(main_output_path, 'im')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)
    os.makedirs(im_path_out, exist_ok=True)

    im_sizes = {}
    
    for imagename in tqdm(os.listdir(images_path)):

        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))
    
        pred_mask_sum, im, im_size, lists_equal = get_im_prediction_multiclass(models, prepared_image, filter_unequal_class_pred)


        write_data = False

        if filter_unequal_class_pred and lists_equal:
            write_data = True

        if filter_unequal_class_pred == False:
            write_data = True

        im_sizes[imagename[:-4]] = im_size

        if erode_kernel > 0:
            kernel = np.ones((erode_kernel, erode_kernel), 'uint8')
            im = cv2.erode(im, kernel, iterations=1)

            pred_mask_sum = dilate_mask(pred_mask_sum)

        if dilate_kernel > 0:
            kernel = np.ones((dilate_kernel, dilate_kernel), 'uint8')
            im = cv2.dilate(im, kernel, iterations=1)


        if block_input == True:
            if c == 3:
                image[im > 0] = [0,0,0]
            else:
                image[im > 0] = 0

        if block_output == True:
            pred_mask_sum[im > 0] = 0

        if write_data:
            cv2.imwrite(os.path.join(images_path_out, imagename), image)
            cv2.imwrite(os.path.join(masks_path_out, imagename), pred_mask_sum)
        cv2.imwrite(os.path.join(im_path_out, imagename), im)

    mean_im_size = round(sum(im_sizes.values()) / len(im_sizes),0)

    return mean_im_size




def dilate_mask(mask, kernel_size=3, iterations=1):
    '''
    Dilate the mask for each class separately in a multiclass mask.

    Args:
        mask (np.array): The input multiclass mask.
        kernel_size (int, optional): Size of the dilation kernel. Defaults to 3.
        iterations (int, optional): Number of dilation iterations. Defaults to 1.

    Returns:
        np.array: The dilated multiclass mask.
    '''
    
    kernel = np.ones((kernel_size, kernel_size),np.uint8)
    unique_classes = np.unique(mask)
    dilated_mask = np.zeros_like(mask)
    
    for u in unique_classes:
        if u == 0:
            continue
            
        binary_mask = (mask == u).astype(np.uint8)
        dilated_binary_mask = cv2.dilate(binary_mask, kernel, iterations=iterations)
        dilated_mask[dilated_binary_mask == 1] = u

    return dilated_mask



def pred_masks_to_im_binary(pred_masks):
    pred_masks = np.stack(pred_masks, axis=0)

    summed_masks = np.sum(pred_masks, axis=0)

    num_masks = pred_masks.shape[0]
    final_pred_mask = np.where(summed_masks == num_masks, 1, 0)

    final_inconsistency_mask = np.where((summed_masks != 0) & (summed_masks != num_masks), 1, 0)

    pred_size = np.sum(final_pred_mask)
    inconsistency_size = np.sum(final_inconsistency_mask)

    final_pred_mask[final_pred_mask==1] = 255
    final_inconsistency_mask[final_inconsistency_mask==1] = 255

    return final_pred_mask.squeeze().astype(np.uint8), final_inconsistency_mask.squeeze().astype(np.uint8), inconsistency_size, pred_size


def pred_masks_to_im_multiclass(pred_masks):
    # Convert the list of masks to a numpy array
    pred_masks = np.stack(pred_masks, axis=0)

    # Perform the intersection operation
    final_pred_mask = np.where(np.all(pred_masks == pred_masks[0, :], axis=0), pred_masks[0, :], 0)

    # Perform the operation to get 1 where model outputs don't match
    inconsistency_mask = np.where(np.all(pred_masks == pred_masks[0, :], axis=0), 0, 1)

    inconsistency_size = np.sum(inconsistency_mask)

    inconsistency_mask[inconsistency_mask==1] = 255

    return np.squeeze(final_pred_mask).astype(np.uint8), np.squeeze(inconsistency_mask).astype(np.uint8), inconsistency_size


def get_im_prediction_binary(models, prepared_image, threshold=0.5):
    '''
    Generate binary predictions and inconsistency mask using an ensemble of models.

    Args:
        models (list): List of models for ensemble prediction.
        prepared_image (np.array): Preprocessed image ready for prediction.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

    Returns:
        tuple: A tuple containing the final prediction mask, inconsistency mask, size of inconsistency areas, and prediction size.
    '''
    
    pred_masks = []

    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):
            pred_mask = model.predict([prepared_image])[0] > threshold
            pred_masks.append(pred_mask.astype(int))

    final_pred_mask, final_inconsistency_mask, inconsistency_size, pred_size = pred_masks_to_im_binary(pred_masks)

    return final_pred_mask, final_inconsistency_mask, inconsistency_size, pred_size


def get_im_prediction_hela(models, prepared_image, threshold=0.5):
    '''
    Generate predictions for alive, dead, and positional masks for the HeLa dataset with inconsistency mask using an ensemble of models.

    Args:
        models (list): List of models for ensemble prediction.
        prepared_image (np.array): Preprocessed image ready for prediction.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

    Returns:
        tuple: A tuple containing the final masks for alive, dead, and positional predictions, a combined inconsistency mask, and the total inconsistency size.
    '''
    
    alive_masks = []
    dead_masks = []
    pos_masks = []
    
    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):
            output_data = model.predict([prepared_image])[0]
            alive, dead, pos = cv2.split(output_data)
            
            alive_mask = (alive >= threshold).astype(int)
            dead_mask = (dead >= threshold).astype(int)
            pos_mask = (pos >= threshold).astype(int)
            
            alive_masks.append(alive_mask)
            dead_masks.append(dead_mask)
            pos_masks.append(pos_mask)

    final_alive_mask, im_alive, im_alive_size, _ = pred_masks_to_im_binary(alive_masks)
    final_dead_mask, im_dead, im_dead_size, _ = pred_masks_to_im_binary(dead_masks)
    final_pos_mask, im_pos, im_pos_size, _ = pred_masks_to_im_binary(pos_masks)

    combined_im = np.maximum(np.maximum(im_alive, im_dead), im_pos)
    im_size = im_alive_size + im_dead_size + im_pos_size

    return final_alive_mask, final_dead_mask, final_pos_mask, combined_im, im_size



def get_im_prediction_multiclass(models, prepared_image, filter_unequal_class_pred=False):
    '''
    Generate multiclass predictions and inconsistency mask using an ensemble of models.

    Args:
        models (list): List of models for ensemble prediction.
        prepared_image (np.array): Preprocessed image ready for prediction.
        filter_unequal_class_pred (bool, optional): Flag to filter out predictions if class distributions are unequal. Defaults to False.

    Returns:
        tuple: A tuple containing the final multiclass prediction mask, inconsistency mask, inconsistency size, and a boolean indicating if class distributions are consistent across models.
    '''
    
    pred_masks = []
    unique_classes = []

    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):
            mask_pred = model.predict([prepared_image])
            mask_pred = np.argmax(mask_pred, axis=-1)
            unique_classes.append(np.unique(mask_pred))

            pred_masks.append(mask_pred)


    if filter_unequal_class_pred:
        lists_equal = all(set(unique_classes[0]) == set(sublist) for sublist in unique_classes)
    else:
        lists_equal = True

    final_pred_mask, im, im_size = pred_masks_to_im_multiclass(pred_masks)

    return final_pred_mask, im, im_size, lists_equal




def create_pseudo_labels_noisy_student_ISIC_2018(model, h, w, c, images_path, main_output_path, rgb=True, brightness_range_alpha=(0.5, 1.5), brightness_range_beta=(-25, 25), max_blur=3, max_noise=25, free_rotation=True):
    '''
    Generate augmented pseudo labels using the Noisy Student approach for ISIC 2018 dataset.

    Args:
        model: The model used for generating pseudo labels.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        images_path (str): Directory containing input images.
        main_output_path (str): Main directory to save the augmented images and masks.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.5, 1.5).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-25, 25).
        max_blur (int, optional): Maximum blur to apply. Defaults to 3.
        max_noise (int, optional): Maximum noise to add. Defaults to 25.
        free_rotation (bool, optional): Flag to apply free rotation. Defaults to True.

    Returns:
        None
    '''
    
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)
    
    for imagename in tqdm(os.listdir(images_path)):
    
        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image
    
        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))

        with contextlib.redirect_stdout(io.StringIO()):
            mask = model.predict([prepared_image])

        mask = ((mask > 0.5)*255).astype(np.uint8)
        mask = np.reshape(mask, (h, w))

        aug_image, aug_mask = augment_image_and_mask(image, mask, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation)     
        
        cv2.imwrite(os.path.join(images_path_out, imagename), aug_image)
        cv2.imwrite(os.path.join(masks_path_out, imagename), aug_mask)



def create_pseudo_labels_noisy_student_hela(model, h, w, c, images_path, main_output_path, brightness_range_alpha=(0.5, 1.5), brightness_range_beta=(-25, 25), max_blur=3, max_noise=25, free_rotation=True, max_pos_circle_size=8, min_pos_circle_size=3):
    '''
    Generate augmented pseudo labels using the Noisy Student approach for the HeLa dataset.

    Args:
        model: The model used for generating pseudo labels.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        images_path (str): Directory containing brightfield images.
        main_output_path (str): Main directory to save the augmented images and masks.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.5, 1.5).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-25, 25).
        max_blur (int, optional): Maximum blur to apply. Defaults to 3.
        max_noise (int, optional): Maximum noise to add. Defaults to 25.
        free_rotation (bool, optional): Flag to apply free rotation. Defaults to True.
        max_pos_circle_size (int, optional): Maximum size of circles in positional mask. Defaults to 8.
        min_pos_circle_size (int, optional): Minimum size of circles in positional mask. Defaults to 3.

    Returns:
        None
    '''
    
    brightfield_path_out = os.path.join(main_output_path, 'brightfield')
    alive_path_out = os.path.join(main_output_path, 'alive')
    dead_path_out = os.path.join(main_output_path, 'dead')
    pos_path_out = os.path.join(main_output_path, 'mod_position')
    
    os.makedirs(brightfield_path_out, exist_ok=True)
    os.makedirs(alive_path_out, exist_ok=True)
    os.makedirs(dead_path_out, exist_ok=True)
    os.makedirs(pos_path_out, exist_ok=True)
    
    for imagename in tqdm(os.listdir(images_path)):
    
        image = cv2.imread(os.path.join(images_path, imagename), 0)
    
        prepared_image = (np.array((image).reshape(-1, h, w, c), dtype=np.uint8))

        pred_masks = []

        with contextlib.redirect_stdout(io.StringIO()):
            output_data = model.predict([prepared_image])
            alive, dead, pos = cv2.split(output_data[0])

            pred_masks.append(alive)
            pred_masks.append(dead)
            pred_masks.append(pos)

        aug_image, aug_masks = augment_image_and_masks(image, pred_masks, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation)     
        
        cv2.imwrite(os.path.join(brightfield_path_out, f'{imagename[:-4]}_aug.png'), aug_image)
        cv2.imwrite(os.path.join(alive_path_out, f'{imagename[:-4]}_aug.png'), (aug_masks[0] >= 0.5).astype(np.int)*255)
        cv2.imwrite(os.path.join(dead_path_out, f'{imagename[:-4]}_aug.png'), (aug_masks[1] >= 0.5).astype(np.int)*255)

        temp_pos = ((aug_masks[2] >= 0.5)*255).astype(np.uint8)

        positions = get_pos_contours(temp_pos)
        #h, w = final_pos_mask_raw.shape
        final_pos_mask = np.zeros((h, w, 3), np.uint8)
        
        
        for pos in positions:
            if len(positions) > 1:
                min_dist = get_min_dist(pos, positions)
            else:
                min_dist = 99
        
            circle_size = int(min_dist // 4)
            circle_size = max(min(circle_size, max_pos_circle_size), min_pos_circle_size)
            cv2.circle(final_pos_mask, (pos[0], pos[1]), circle_size, (255, 255, 255), -1)
        

        cv2.imwrite(os.path.join(pos_path_out, f'{imagename[:-4]}_aug.png'), final_pos_mask)   



def create_pseudo_labels_noisy_student_multiclass(model, h, w, c, images_path, main_output_path, rgb=True, brightness_range_alpha=(0.5, 1.5), brightness_range_beta=(-25, 25), max_blur=3, max_noise=25, free_rotation=True):
    '''
    Generate augmented pseudo labels using the Noisy Student approach for multiclass segmentation.

    Args:
        model: The model used for generating pseudo labels.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        images_path (str): Directory containing input images.
        main_output_path (str): Main directory to save the augmented images and masks.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.5, 1.5).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-25, 25).
        max_blur (int, optional): Maximum blur to apply. Defaults to 3.
        max_noise (int, optional): Maximum noise to add. Defaults to 25.
        free_rotation (bool, optional): Flag to apply free rotation. Defaults to True.

    Returns:
        None
    '''
    
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)
    
    for imagename in tqdm(os.listdir(images_path)):
    
        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image
    
        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))

        with contextlib.redirect_stdout(io.StringIO()):
            mask_pred = model.predict([prepared_image])
            mask_pred = np.argmax(mask_pred, axis=-1)[0]

        aug_image, aug_mask = augment_image_and_mask(image, mask_pred, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation)     
        
        cv2.imwrite(os.path.join(images_path_out, imagename), aug_image)
        cv2.imwrite(os.path.join(masks_path_out, imagename), aug_mask)


def create_training_data_evalnet_ISIC_2018(model, h, w, c, images_path, masks_path, main_output_path, i, rgb=True):
    '''
    Generate training data for EvalNet using predictions from a model on the ISIC 2018 dataset.

    Args:
        model: The model used for generating pseudo labels.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        images_path (str): Directory containing input images.
        masks_path (str): Directory containing corresponding masks.
        main_output_path (str): Directory to save the generated data and IoU scores.
        i (int): Iteration number for naming the output files.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.

    Returns:
        None
    '''
    
    imagename_ious = []

    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)


    for imagename in tqdm(os.listdir(images_path)):
    
        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        mask = cv2.imread(os.path.join(masks_path, imagename))
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))
        
        with contextlib.redirect_stdout(io.StringIO()):
            mask = model.predict([prepared_image])
        
            mask = ((mask > 0.5)*255).astype(np.uint8)
            mask = np.reshape(mask, (h, w))

        if i >= 10:
            if 'aug' in imagename:
                pred_name = f'{imagename[:-10]}___{i}_{imagename[-6:-4]}.png'
            else:
                pred_name = f'{imagename[:-4]}___{i}.png'
        else:
            pred_name = f'{imagename[:-4]}___{i}.png'
        cv2.imwrite(os.path.join(masks_path_out, pred_name), mask)

        iou = round(get_IoU_binary(mask_gray, mask),4)

        imagename_ious.append((pred_name, iou))


    if i == 0:
        for imagename in tqdm(os.listdir(images_path)):
            imagename_ious.append((imagename, 1.0))

            shutil.copy(os.path.join(images_path, imagename), os.path.join(images_path_out, imagename))
            shutil.copy(os.path.join(masks_path, imagename), os.path.join(masks_path_out, imagename))

        
    with open(os.path.join(main_output_path, 'labels.csv'), 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')

        for row in imagename_ious:
            writer.writerow(row)



def create_training_data_evalnet_multiclass(model, h, w, c, images_path, masks_path, main_output_path, i, rgb=True):
    '''
    Generate training data for a multiclass EvalNet using predictions from a model.

    Args:
        model: The model used for generating pseudo labels.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        images_path (str): Directory containing input images.
        masks_path (str): Directory containing corresponding masks.
        main_output_path (str): Directory to save the generated data and IoU scores.
        i (int): Iteration number for naming the output files.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.

    Returns:
        None
    '''
    
    imagename_ious = []

    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)


    for imagename in tqdm(os.listdir(images_path)):
    
        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        mask = cv2.imread(os.path.join(masks_path, imagename))
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))
        
        with contextlib.redirect_stdout(io.StringIO()):
            mask_pred = model.predict([prepared_image])
            mask_pred = np.argmax(mask_pred, axis=-1)[0]
        
        if i >= 10:
            if 'aug' in imagename:
                pred_name = f'{imagename[:-10]}___{i}_{imagename[-6:-4]}.png'
            else:
                pred_name = f'{imagename[:-4]}___{i}.png'
        else:
            pred_name = f'{imagename[:-4]}___{i}.png'
        cv2.imwrite(os.path.join(masks_path_out, pred_name), mask_pred)

        iou = round(get_IoU_multi_unique(mask_gray, mask_pred),4)

        imagename_ious.append((pred_name, iou))


    if i == 0:
        for imagename in tqdm(os.listdir(images_path)):
            imagename_ious.append((imagename, 1.0))

            shutil.copy(os.path.join(images_path, imagename), os.path.join(images_path_out, imagename))
            shutil.copy(os.path.join(masks_path, imagename), os.path.join(masks_path_out, imagename))

        
    with open(os.path.join(main_output_path, 'labels.csv'), 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')

        for row in imagename_ious:
            writer.writerow(row)




def create_training_data_evalnet_im_binary(models, h, w, c, images_path, masks_path, main_output_path, num_loops, n_min_models=2, n_max_models=4, rgb=True, brightness_range_alpha=(0.6, 1.4), brightness_range_beta=(-20, 20), max_blur=3, max_noise=20, free_rotation=False):
    '''
    Generate training data for EvalNet using binary predictions with inconsistency masks from an ensemble of models.

    Args:
        models (list): List of models for ensemble prediction.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        images_path (str): Directory containing input images.
        masks_path (str): Directory containing corresponding masks.
        main_output_path (str): Directory to save the generated data and IoU scores.
        num_loops (int): Number of augmentation loops to run.
        n_min_models (int): Minimum number of models to use in each prediction. Defaults to 2.
        n_max_models (int): Maximum number of models to use in each prediction. Defaults to 4.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.6, 1.4).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-20, 20).
        max_blur (int, optional): Maximum blur to apply. Defaults to 3.
        max_noise (int, optional): Maximum noise to add. Defaults to 20.
        free_rotation (bool, optional): Flag to apply free rotation. Defaults to False.

    Returns:
        None
    '''
    
    imagename_ious = []
    kernel_list = [0, 3, 5]

    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)

    for nl in range(num_loops):
        for imagename in tqdm(os.listdir(images_path)):
        
            image = cv2.imread(os.path.join(images_path, imagename))
            if rgb == True:
                input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                input_image = image

            mask_gray = cv2.imread(os.path.join(masks_path, imagename), 0)

            prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))

            n_selected_models = random.randint(n_min_models, n_max_models)

            selected_models = random.sample(models, n_selected_models)

            pred_mask_sum, im, im_size, lists_equal = get_im_prediction_binary(selected_models, prepared_image)

            
            erode_kernel = random.choice(kernel_list)
            if erode_kernel > 0:
                kernel = np.ones((erode_kernel, erode_kernel), 'uint8')
                im = cv2.erode(im, kernel, iterations=1)
            
            dilate_kernel = random.choice(kernel_list)
            if dilate_kernel > 0:
                kernel = np.ones((dilate_kernel, dilate_kernel), 'uint8')
                im = cv2.dilate(im, kernel, iterations=1)
            
            
            if c == 3:
                image[im > 0] = [0,0,0]
            else:
                image[im > 0] = 0
            
            pred_mask_sum[im > 0] = 0


            iou = round(get_IoU_binary(mask_gray, pred_mask_sum),4)

            pred_name = f'{imagename[:-4]}_aug_{nl}.png'
            imagename_ious.append((pred_name, iou))

            image_to_save = image
            mask_to_save = pred_mask_sum

            # rnd if aug or not
            if random.random() < 0.5:
                image_to_save, mask_to_save = augment_image_and_mask(image, pred_mask_sum, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation=free_rotation)
            
            cv2.imwrite(os.path.join(images_path_out, pred_name), image_to_save)
            cv2.imwrite(os.path.join(masks_path_out, pred_name), mask_to_save)

        
        tf.keras.backend.clear_session()

    with open(os.path.join(main_output_path, 'labels.csv'), 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')

        for row in imagename_ious:
            writer.writerow(row)




def create_training_data_evalnet_im_multiclass(models, h, w, c, images_path, masks_path, main_output_path, num_loops, n_min_models=2, n_max_models=4, rgb=True, brightness_range_alpha=(0.6, 1.4), brightness_range_beta=(-20, 20), max_blur=3, max_noise=20, free_rotation=False):
    '''
    Generate training data for EvalNet using multiclass predictions with inconsistency masks from an ensemble of models.

    Args:
        models (list): List of models for ensemble prediction.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        images_path (str): Directory containing input images.
        masks_path (str): Directory containing corresponding masks.
        main_output_path (str): Directory to save the generated data and IoU scores.
        num_loops (int): Number of augmentation loops to run.
        n_min_models (int): Minimum number of models to use in each prediction. Defaults to 2.
        n_max_models (int): Maximum number of models to use in each prediction. Defaults to 4.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.6, 1.4).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-20, 20).
        max_blur (int, optional): Maximum blur to apply. Defaults to 3.
        max_noise (int, optional): Maximum noise to add. Defaults to 20.
        free_rotation (bool, optional): Flag to apply free rotation. Defaults to False.

    Returns:
        None
    '''
    
    imagename_ious = []
    kernel_list = [0, 3, 5]

    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)

    for nl in range(num_loops):
        for imagename in tqdm(os.listdir(images_path)):
        
            image = cv2.imread(os.path.join(images_path, imagename))
            if rgb == True:
                input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                input_image = image

            mask_gray = cv2.imread(os.path.join(masks_path, imagename), 0)

            prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))

            n_selected_models = random.randint(n_min_models, n_max_models)

            selected_models = random.sample(models, n_selected_models)

            pred_mask_sum, im, im_size, lists_equal = get_im_prediction_multiclass(selected_models, prepared_image)

            
            erode_kernel = random.choice(kernel_list)
            if erode_kernel > 0:
                kernel = np.ones((erode_kernel, erode_kernel), 'uint8')
                im = cv2.erode(im, kernel, iterations=1)
            
            dilate_kernel = random.choice(kernel_list)
            if dilate_kernel > 0:
                kernel = np.ones((dilate_kernel, dilate_kernel), 'uint8')
                im = cv2.dilate(im, kernel, iterations=1)
            
            
            if c == 3:
                image[im > 0] = [0,0,0]
            else:
                image[im > 0] = 0
            
            pred_mask_sum[im > 0] = 0


            iou = round(get_IoU_multi_unique(mask_gray, pred_mask_sum),4)

            pred_name = f'{imagename[:-4]}_aug_{nl}.png'
            imagename_ious.append((pred_name, iou))

            image_to_save = image
            mask_to_save = pred_mask_sum

            # rnd if aug or not
            if random.random() < 0.5:
                image_to_save, mask_to_save = augment_image_and_mask(image, pred_mask_sum, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation=free_rotation)
            
            cv2.imwrite(os.path.join(images_path_out, pred_name), image_to_save)
            cv2.imwrite(os.path.join(masks_path_out, pred_name), mask_to_save)

        
        tf.keras.backend.clear_session()

    with open(os.path.join(main_output_path, 'labels.csv'), 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')

        for row in imagename_ious:
            writer.writerow(row)



def create_training_data_evalnet_miou_im_multiclass(models, h, w, c, num_classes, images_path, masks_path, main_output_path, num_loops, n_min_models=2, n_max_models=4, rgb=True, brightness_range_alpha=(0.8, 1.2), brightness_range_beta=(-10, 10), max_blur=1, max_noise=10, free_rotation=False):
    '''
    Generate training data for an multiclass EvalNet with mIoU scores for each class, using predictions with inconsistency masks from an ensemble of models.

    Args:
        models (list): List of models for ensemble prediction.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        num_classes (int): Number of classes in the segmentation.
        images_path (str): Directory containing input images.
        masks_path (str): Directory containing corresponding masks.
        main_output_path (str): Directory to save the generated data, mIoU scores, and class detection.
        num_loops (int): Number of augmentation loops to run.
        n_min_models (int): Minimum number of models to use in each prediction. Defaults to 2.
        n_max_models (int): Maximum number of models to use in each prediction. Defaults to 4.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.8, 1.2).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-10, 10).
        max_blur (int, optional): Maximum blur to apply. Defaults to 1.
        max_noise (int, optional): Maximum noise to add. Defaults to 10.
        free_rotation (bool, optional): Flag to apply free rotation. Defaults to False.

    Returns:
        None
    '''
    
    imagename_ious_detection = []
    kernel_list = [0, 3, 5]

    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)

    for nl in range(num_loops):
        for imagename in tqdm(os.listdir(images_path)):
        
            image = cv2.imread(os.path.join(images_path, imagename))
            if rgb == True:
                input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                input_image = image

            mask_gray = cv2.imread(os.path.join(masks_path, imagename), 0)

            prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))

            n_selected_models = random.randint(n_min_models, n_max_models)

            selected_models = random.sample(models, n_selected_models)

            pred_mask_sum, im, im_size, lists_equal = get_im_prediction_multiclass(selected_models, prepared_image)

            
            erode_kernel = random.choice(kernel_list)
            if erode_kernel > 0:
                kernel = np.ones((erode_kernel, erode_kernel), 'uint8')
                im = cv2.erode(im, kernel, iterations=1)
            
            dilate_kernel = random.choice(kernel_list)
            if dilate_kernel > 0:
                kernel = np.ones((dilate_kernel, dilate_kernel), 'uint8')
                im = cv2.dilate(im, kernel, iterations=1)
            
            
            if c == 3:
                image[im > 0] = [0,0,0]
            else:
                image[im > 0] = 0
            
            pred_mask_sum[im > 0] = 0

            ious = compute_classwise_IoU(pred_mask_sum, mask_gray, num_classes)

            gt_class_counts = np.zeros(num_classes)
            bins = np.bincount(mask_gray.ravel(), minlength=num_classes)
            gt_class_counts[:len(bins)] += bins

            mask_gray[im > 0] = 0
            detected_class = compute_classwise_detection_im(mask_gray, num_classes, gt_class_counts, 0.3)
            
            pred_name = f'{imagename[:-4]}_aug_{nl}.png'
            imagename_ious_detection.append((pred_name, *ious, *detected_class))

            image_to_save = image
            mask_to_save = pred_mask_sum

            # rnd if aug or not
            if random.random() < 0.5:
                image_to_save, mask_to_save = augment_image_and_mask(image, pred_mask_sum, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation=free_rotation)
            
            cv2.imwrite(os.path.join(images_path_out, pred_name), image_to_save)
            cv2.imwrite(os.path.join(masks_path_out, pred_name), mask_to_save)

        
        tf.keras.backend.clear_session()

    with open(os.path.join(main_output_path, 'labels.csv'), 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')

        for row in imagename_ious_detection:
            writer.writerow(row)




def create_training_data_evalnet_miou_im_hela(models, h, w, c, main_input_path, main_output_path, num_loops, n_min_models=2, n_max_models=4, brightness_range_alpha=(0.8, 1.2), brightness_range_beta=(-10, 10), max_blur=1, max_noise=10, free_rotation=False):
    '''
    Generate training data for EvalNet with mIoU scores and class detection for the HeLa dataset using predictions with inconsistency masks from an ensemble of models.

    Args:
        models (list): List of models for ensemble prediction.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        main_input_path (str): Directory containing input HeLa cell images and masks.
        main_output_path (str): Directory to save the generated data, mIoU scores, and class detection.
        num_loops (int): Number of augmentation loops to run.
        n_min_models (int): Minimum number of models to use in each prediction. Defaults to 2.
        n_max_models (int): Maximum number of models to use in each prediction. Defaults to 4.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.
        brightness_range_alpha (tuple, optional): Range for alpha in brightness adjustment. Defaults to (0.8, 1.2).
        brightness_range_beta (tuple, optional): Range for beta in brightness adjustment. Defaults to (-10, 10).
        max_blur (int, optional): Maximum blur to apply. Defaults to 1.
        max_noise (int, optional): Maximum noise to add. Defaults to 10.
        free_rotation (bool, optional): Flag to apply free rotation. Defaults to False.

    Returns:
        None
    '''
    
    imagename_ious_detection = []
    kernel_list = [0, 3, 5]

    bf_images_path_in = os.path.join(main_input_path, 'brightfield')
    alive_masks_path_in = os.path.join(main_input_path, 'alive')
    dead_masks_path_in = os.path.join(main_input_path, 'dead')
    pos_masks_path_in = os.path.join(main_input_path, 'mod_position')

    bf_images_path_out = os.path.join(main_output_path, 'brightfield')
    alive_masks_path_out = os.path.join(main_output_path, 'alive')
    dead_masks_path_out = os.path.join(main_output_path, 'dead')
    pos_masks_path_out = os.path.join(main_output_path, 'mod_position')

    os.makedirs(bf_images_path_out, exist_ok=True)
    os.makedirs(alive_masks_path_out, exist_ok=True)
    os.makedirs(dead_masks_path_out, exist_ok=True)
    os.makedirs(pos_masks_path_out, exist_ok=True)

    for nl in range(num_loops):
        for imagename in tqdm(os.listdir(bf_images_path_in)):

            bf_image_gray = cv2.imread(os.path.join(bf_images_path_in, imagename), 0)
            alive_mask = cv2.imread(os.path.join(alive_masks_path_in, imagename), 0)
            dead_mask = cv2.imread(os.path.join(dead_masks_path_in, imagename), 0)
            pos_mask = cv2.imread(os.path.join(pos_masks_path_in, imagename), 0)
            
            prepared_image = (np.array((bf_image_gray).reshape(-1, h, w, c), dtype=np.uint8))

            n_selected_models = random.randint(n_min_models, n_max_models)

            selected_models = random.sample(models, n_selected_models)

            final_alive_mask, final_dead_mask, final_pos_mask, combined_im, im_size = get_im_prediction_hela(selected_models, prepared_image)

            final_alive_mask = final_alive_mask * 255
            final_dead_mask = final_dead_mask * 255
            final_pos_mask = final_pos_mask * 255
            combined_im = combined_im * 255

            
            erode_kernel = random.choice(kernel_list)
            if erode_kernel > 0:
                kernel = np.ones((erode_kernel, erode_kernel), 'uint8')
                combined_im = cv2.erode(combined_im, kernel, iterations=1)
            
            dilate_kernel = random.choice(kernel_list)
            if dilate_kernel > 0:
                kernel = np.ones((dilate_kernel, dilate_kernel), 'uint8')
                combined_im = cv2.dilate(combined_im, kernel, iterations=1)
            
            
            bf_image_gray[combined_im > 0] = 0
            
            final_alive_mask[combined_im > 0] = 0
            final_dead_mask[combined_im > 0] = 0
            final_pos_mask[combined_im > 0] = 0

            iou_alive = get_IoU_binary(alive_mask, final_alive_mask)
            iou_dead = get_IoU_binary(dead_mask, final_dead_mask)
            iou_pos = get_IoU_binary(pos_mask, final_pos_mask)
            
            detection_alive = 0
            detection_dead = 0
            detection_pos = 0
            
            if np.count_nonzero(alive_mask) >= np.prod(alive_mask.shape) * 0.01:
                detection_alive = 1
            
            if np.count_nonzero(dead_mask) >= np.prod(dead_mask.shape) * 0.01:
                detection_dead = 1
            
            if np.count_nonzero(pos_mask) >= np.prod(pos_mask.shape) * 0.001:
                detection_pos = 1

            imagename_out = f'{imagename[:-4]}_aug_{nl}.png'
            imagename_ious_detection.append((imagename_out, iou_alive, iou_dead, iou_pos, detection_alive, detection_dead, detection_pos))

            # rnd if aug or not
            if random.random() < 0.5:

                masks = [final_alive_mask, final_dead_mask, final_pos_mask]
                aug_bf_image, aug_masks = augment_image_and_masks(bf_image_gray, masks, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation)        
    
                cv2.imwrite(os.path.join(bf_images_path_out, imagename_out), aug_bf_image)
                cv2.imwrite(os.path.join(alive_masks_path_out, imagename_out), aug_masks[0])
                cv2.imwrite(os.path.join(dead_masks_path_out, imagename_out), aug_masks[1])
                cv2.imwrite(os.path.join(pos_masks_path_out, imagename_out), aug_masks[2])


            cv2.imwrite(os.path.join(bf_images_path_out, imagename_out), bf_image_gray)
            cv2.imwrite(os.path.join(alive_masks_path_out, imagename_out), final_alive_mask)
            cv2.imwrite(os.path.join(dead_masks_path_out, imagename_out), final_dead_mask)
            cv2.imwrite(os.path.join(pos_masks_path_out, imagename_out), final_pos_mask)
        
        tf.keras.backend.clear_session()

    with open(os.path.join(main_output_path, 'labels.csv'), 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')

        for row in imagename_ious_detection:
            writer.writerow(row)




def create_training_data_evalnet_miou_hela(model, h, w, c, main_input_path, main_output_path, i, threshold=0.5):
    '''
    Generate training data for EvalNet with mIoU scores and class detection for the HeLa dataset using predictions from a model.

    Args:
        model: The model used for generating predictions.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        main_input_path (str): Directory containing input HeLa cell images and masks.
        main_output_path (str): Directory to save the generated data, mIoU scores, and class detection.
        i (int): Iteration number for naming the output files.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

    Returns:
        None
    '''
    
    imagename_ious_detection = []

    bf_images_path_in = os.path.join(main_input_path, 'brightfield')
    alive_masks_path_in = os.path.join(main_input_path, 'alive')
    dead_masks_path_in = os.path.join(main_input_path, 'dead')
    pos_masks_path_in = os.path.join(main_input_path, 'mod_position')

    bf_images_path_out = os.path.join(main_output_path, 'brightfield')
    alive_masks_path_out = os.path.join(main_output_path, 'alive')
    dead_masks_path_out = os.path.join(main_output_path, 'dead')
    pos_masks_path_out = os.path.join(main_output_path, 'mod_position')

    os.makedirs(bf_images_path_out, exist_ok=True)
    os.makedirs(alive_masks_path_out, exist_ok=True)
    os.makedirs(dead_masks_path_out, exist_ok=True)
    os.makedirs(pos_masks_path_out, exist_ok=True)


    for imagename in tqdm(os.listdir(bf_images_path_in)):
    
        bf_image_gray = cv2.imread(os.path.join(bf_images_path_in, imagename), 0)
        alive_mask = cv2.imread(os.path.join(alive_masks_path_in, imagename), 0)
        dead_mask = cv2.imread(os.path.join(dead_masks_path_in, imagename), 0)
        pos_mask = cv2.imread(os.path.join(pos_masks_path_in, imagename), 0)

        prepared_image = (np.array((bf_image_gray).reshape(-1, h, w, c), dtype=np.uint8))

        with contextlib.redirect_stdout(io.StringIO()):
            output_data = model.predict(prepared_image)
            alive, dead, pos = cv2.split(output_data[0])

            # Apply threshold and set to values 0 or 255
            final_alive_mask = (alive > threshold).astype(np.uint8) * 255
            final_dead_mask = (dead > threshold).astype(np.uint8) * 255
            final_pos_mask = (pos > threshold).astype(np.uint8) * 255
        
        if i >= 10:
            if 'aug' in imagename:
                pred_name = f'{imagename[:-10]}___{i}_{imagename[-6:-4]}.png'
            else:
                pred_name = f'{imagename[:-4]}___{i}.png'
        else:
            pred_name = f'{imagename[:-4]}___{i}.png'

        cv2.imwrite(os.path.join(alive_masks_path_out, pred_name), final_alive_mask)
        cv2.imwrite(os.path.join(dead_masks_path_out, pred_name), final_dead_mask)
        cv2.imwrite(os.path.join(pos_masks_path_out, pred_name), final_pos_mask)

        iou_alive = 0
        iou_dead = 0
        iou_pos = 0

        detection_alive = 0
        detection_dead = 0
        detection_pos = 0

        if np.count_nonzero(alive_mask) >= np.prod(alive_mask.shape) * 0.01:
            detection_alive = 1
            iou_alive = get_IoU_binary(alive_mask, final_alive_mask)

        if np.count_nonzero(dead_mask) >= np.prod(dead_mask.shape) * 0.01:
            detection_dead = 1
            iou_dead = get_IoU_binary(dead_mask, final_dead_mask)

        if np.count_nonzero(pos_mask) >= np.prod(pos_mask.shape) * 0.001:
            detection_pos = 1
            iou_pos = get_IoU_binary(pos_mask, final_pos_mask)

        imagename_ious_detection.append((pred_name, iou_alive, iou_dead, iou_pos, detection_alive, detection_dead, detection_pos))


    if i == 0:
        for imagename in tqdm(os.listdir(bf_images_path_in)):

            iou_alive = 0
            iou_dead = 0
            iou_pos = 0

            detection_alive = 0
            detection_dead = 0
            detection_pos = 0
            
            if np.count_nonzero(alive_mask) >= np.prod(alive_mask.shape) * 0.01:
                detection_alive = 1
                iou_alive = 1
            
            if np.count_nonzero(dead_mask) >= np.prod(dead_mask.shape) * 0.01:
                detection_dead = 1
                iou_dead = 1
            
            if np.count_nonzero(pos_mask) >= np.prod(pos_mask.shape) * 0.001:
                detection_pos = 1
                iou_pos = 1

            imagename_ious_detection.append((imagename, iou_alive, iou_dead, iou_pos, detection_alive, detection_dead, detection_pos))

            shutil.copy(os.path.join(bf_images_path_in, imagename), os.path.join(bf_images_path_out, imagename))
            shutil.copy(os.path.join(alive_masks_path_in, imagename), os.path.join(alive_masks_path_out, imagename))
            shutil.copy(os.path.join(dead_masks_path_in, imagename), os.path.join(dead_masks_path_out, imagename))
            shutil.copy(os.path.join(pos_masks_path_in, imagename), os.path.join(pos_masks_path_out, imagename))

        
    with open(os.path.join(main_output_path, 'labels.csv'), 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')

        for row in imagename_ious_detection:
            writer.writerow(row)



def create_training_data_evalnet_miou_hela_no_pos(model, h, w, c, main_input_path, main_output_path, i, threshold=0.5):
    '''
    Generate training data for EvalNet with mIoU scores and class detection for the HeLa dataset excluding positional masks, using predictions from a model.

    Args:
        model: The model used for generating predictions.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        main_input_path (str): Directory containing input HeLa cell images and masks.
        main_output_path (str): Directory to save the generated data and mIoU scores.
        i (int): Iteration number for naming the output files.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

    Returns:
        None
    '''
    
    imagename_ious_detection = []

    bf_images_path_in = os.path.join(main_input_path, 'brightfield')
    alive_masks_path_in = os.path.join(main_input_path, 'alive')
    dead_masks_path_in = os.path.join(main_input_path, 'dead')

    bf_images_path_out = os.path.join(main_output_path, 'brightfield')
    alive_masks_path_out = os.path.join(main_output_path, 'alive')
    dead_masks_path_out = os.path.join(main_output_path, 'dead')

    os.makedirs(bf_images_path_out, exist_ok=True)
    os.makedirs(alive_masks_path_out, exist_ok=True)
    os.makedirs(dead_masks_path_out, exist_ok=True)


    for imagename in tqdm(os.listdir(bf_images_path_in)):
    
        bf_image_gray = cv2.imread(os.path.join(bf_images_path_in, imagename), 0)
        alive_mask = cv2.imread(os.path.join(alive_masks_path_in, imagename), 0)
        dead_mask = cv2.imread(os.path.join(dead_masks_path_in, imagename), 0)

        prepared_image = (np.array((bf_image_gray).reshape(-1, h, w, c), dtype=np.uint8))

        with contextlib.redirect_stdout(io.StringIO()):
            output_data = model.predict(prepared_image)
            alive, dead, pos = cv2.split(output_data[0])

            # Apply threshold and set to values 0 or 255
            final_alive_mask = (alive > threshold).astype(np.uint8) * 255
            final_dead_mask = (dead > threshold).astype(np.uint8) * 255
        
        if i >= 10:
            if 'aug' in imagename:
                pred_name = f'{imagename[:-10]}___{i}_{imagename[-6:-4]}.png'
            else:
                pred_name = f'{imagename[:-4]}___{i}.png'
        else:
            pred_name = f'{imagename[:-4]}___{i}.png'

        cv2.imwrite(os.path.join(alive_masks_path_out, pred_name), final_alive_mask)
        cv2.imwrite(os.path.join(dead_masks_path_out, pred_name), final_dead_mask)

        iou_alive = 0
        iou_dead = 0

        detection_alive = 0
        detection_dead = 0

        if np.count_nonzero(alive_mask) >= np.prod(alive_mask.shape) * 0.01:
            detection_alive = 1
            iou_alive = get_IoU_binary(alive_mask, final_alive_mask)

        if np.count_nonzero(dead_mask) >= np.prod(dead_mask.shape) * 0.01:
            detection_dead = 1
            iou_dead = get_IoU_binary(dead_mask, final_dead_mask)

        imagename_ious_detection.append((pred_name, iou_alive, iou_dead, detection_alive, detection_dead))


    if i == 0:
        for imagename in tqdm(os.listdir(bf_images_path_in)):

            iou_alive = 0
            iou_dead = 0

            detection_alive = 0
            detection_dead = 0
            
            if np.count_nonzero(alive_mask) >= np.prod(alive_mask.shape) * 0.01:
                detection_alive = 1
                iou_alive = 1
            
            if np.count_nonzero(dead_mask) >= np.prod(dead_mask.shape) * 0.01:
                detection_dead = 1
                iou_dead = 1

            imagename_ious_detection.append((imagename, iou_alive, iou_dead, detection_alive, detection_dead))

            shutil.copy(os.path.join(bf_images_path_in, imagename), os.path.join(bf_images_path_out, imagename))
            shutil.copy(os.path.join(alive_masks_path_in, imagename), os.path.join(alive_masks_path_out, imagename))
            shutil.copy(os.path.join(dead_masks_path_in, imagename), os.path.join(dead_masks_path_out, imagename))
        
    with open(os.path.join(main_output_path, 'labels.csv'), 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')

        for row in imagename_ious_detection:
            writer.writerow(row)




def create_training_data_evalnet_miou_multiclass(model, h, w, c, num_classes, images_path, masks_path, main_output_path, i, rgb=True):
    '''
    Generate training data for EvalNet with mIoU scores and class detection for multiclass datasets dataset using predictions from a model.

    Args:
        model: The model used for generating predictions.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        num_classes (int): Number of classes in the segmentation.
        images_path (str): Directory containing input images.
        masks_path (str): Directory containing corresponding masks.
        main_output_path (str): Directory to save the generated data, mIoU scores, and class detection.
        i (int): Iteration number for naming the output files.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.

    Returns:
        None
    '''
    
    imagename_ious_detection = []

    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)


    for imagename in tqdm(os.listdir(images_path)):
    
        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        mask_gt = cv2.imread(os.path.join(masks_path, imagename), 0)

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))
        
        with contextlib.redirect_stdout(io.StringIO()):
            mask_pred = model.predict([prepared_image])
            mask_pred = np.argmax(mask_pred, axis=-1)[0]
        
        if i >= 10:
            if 'aug' in imagename:
                pred_name = f'{imagename[:-10]}___{i}_{imagename[-6:-4]}.png'
            else:
                pred_name = f'{imagename[:-4]}___{i}.png'
        else:
            pred_name = f'{imagename[:-4]}___{i}.png'
        cv2.imwrite(os.path.join(masks_path_out, pred_name), mask_pred)

        ious = compute_classwise_IoU(mask_gt, mask_pred, num_classes)
        detected_class = compute_classwise_detection(mask_gt, num_classes)

        imagename_ious_detection.append((pred_name, *ious, *detected_class))


    if i == 0:
        for imagename in tqdm(os.listdir(images_path)):
            ious = compute_classwise_IoU(mask_gt, mask_gt, num_classes)
            detected_class = compute_classwise_detection(mask_gt, num_classes)

            imagename_ious_detection.append((imagename, *ious, *detected_class))

            shutil.copy(os.path.join(images_path, imagename), os.path.join(images_path_out, imagename))
            shutil.copy(os.path.join(masks_path, imagename), os.path.join(masks_path_out, imagename))

        
    with open(os.path.join(main_output_path, 'labels.csv'), 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')

        for row in imagename_ious_detection:
            writer.writerow(row)




def compute_classwise_IoU(pred, gt, num_classes):
    '''
    Compute class-wise Intersection over Union (IoU) for multiclass segmentation.

    Args:
        pred (np.array): Predicted segmentation mask.
        gt (np.array): Ground truth segmentation mask.
        num_classes (int): Number of classes in the segmentation.

    Returns:
        list: List containing IoU scores for each class.
    '''
    
    iou_list = [0] * num_classes  # prefill list with zeros

    if (pred == 0).sum() > 0:
        iou_list[0]=1
    
    for cls in range(num_classes):
        if cls in gt:  # Only compute IoU if class exists in gt
            temp_gt = np.array(gt == cls, dtype=np.float32)
            temp_pred = np.array(pred == cls, dtype=np.float32)

            intersection = np.logical_and(temp_gt, temp_pred).sum()
            union = np.logical_or(temp_gt, temp_pred).sum()

            if union > 0:  # to avoid division by zero
                iou = intersection / union
                iou_list[cls] = round(iou,4)

    return iou_list

def compute_classwise_confluence(gt, num_classes):
    '''
    Compute class-wise confluence, i.e., the proportion of each class in the total area, for multiclass segmentation.

    Args:
        gt (np.array): Ground truth segmentation mask.
        num_classes (int): Number of classes in the segmentation.

    Returns:
        list: List containing confluence scores for each class, representing their proportion in the total area.
    '''
    
    total_pixels = gt.size
    confluence_list = [0] * num_classes  # prefill list with zeros

    for cls in range(num_classes):
        class_pixel_count = (gt == cls).sum()
        confluence_list[cls] = round(class_pixel_count / total_pixels, 4)

    return confluence_list

def get_confluence_binary(gt):
    '''
    Compute the confluence, i.e., the proportion of foreground pixels to the total pixels, for binary segmentation.

    Args:
        gt (np.array): Ground truth binary segmentation mask.

    Returns:
        float: Confluence score representing the proportion of foreground pixels in the total area.
    '''
    
    total_pixels = gt.size
    foreground_pixel_count = gt.sum()
    
    return round(foreground_pixel_count / total_pixels, 4)




def compute_classwise_detection(mask, num_classes):
    '''
    Compute class-wise detection for multiclass segmentation, determining if each class is significantly represented.

    Args:
        mask (np.array): Segmentation mask (either ground truth or predicted).
        num_classes (int): Number of classes in the segmentation.

    Returns:
        list: List indicating detection (1) or non-detection (0) for each class, based on a threshold of 1% of total pixels.
    '''
    
    total_pixels = mask.size
    detected_classes_list = [0] * num_classes

    for cls in range(num_classes):
        pred_class_pixel_count = (mask == cls).sum()

        if pred_class_pixel_count > total_pixels * 0.01:
            detected_classes_list[cls] = 1

    return detected_classes_list


def compute_classwise_detection_im(pred_mask, num_classes, gt_class_counts, threshold):
    '''
    Compute class-wise detection for multiclass segmentation with inconsistency masks based on the proportion of pixels for each class in the predicted mask compared to ground truth class counts.

    Args:
        pred_mask (np.array): Predicted segmentation mask.
        num_classes (int): Number of classes in the segmentation.
        gt_class_counts (list): Counts of pixels for each class in the ground truth mask.
        threshold (float): Minimum ratio of predicted class pixel count to ground truth class pixel count for class detection.

    Returns:
        list: List indicating detection (1) or non-detection (0) for each class based on the threshold and an additional check for classes with significant presence.
    '''
    
    total_pixels = pred_mask.size
    detected_classes_list = [0] * num_classes

    for cls in range(num_classes):
        pred_class_pixel_count = (pred_mask == cls).sum()

        # Handle division by zero
        if gt_class_counts[cls] == 0:
            pixel_ratio = 0
        else:
            pixel_ratio = pred_class_pixel_count / gt_class_counts[cls]

        # Special handling for class 0 (assumed background)
        if cls == 0 and pred_class_pixel_count > 0:
            detected_classes_list[cls] = 1
        else:
            if pixel_ratio >= threshold:
                detected_classes_list[cls] = 1
            elif pred_class_pixel_count / total_pixels >= 0.1: 
                detected_classes_list[cls] = 1

    return detected_classes_list




def train_evalnet_ISIC_2018(model, train_main_path, val_main_path, filepath_h5, batch_size, epochs):
    '''
    Train the EvalNet model for ISIC 2018 dataset and evaluate its performance.

    Args:
        model: The EvalNet model to be trained.
        train_main_path (str): Directory containing training data, images, masks, and labels.
        val_main_path (str): Directory containing validation data, images, masks, and labels.
        filepath_h5 (str): Path to save the best model.
        batch_size (int): Batch size for training and validation.
        epochs (int): Number of epochs to train the model.

    Returns:
        tuple: A tuple containing the mean squared error (mse) and mean absolute error (mae) on the validation dataset.
    '''
    
    train_images_path = os.path.join(train_main_path, 'images')
    train_masks_path = os.path.join(train_main_path, 'masks')

    val_images_path = os.path.join(val_main_path, 'images')
    val_masks_path = os.path.join(val_main_path, 'masks')

    df_train = pd.read_csv(os.path.join(train_main_path, 'labels.csv'), header=None, sep=';')
    df_val = pd.read_csv(os.path.join(val_main_path, 'labels.csv'), header=None, sep=';')
    
    train_generator = generate_images_batch_ISIC_2018(df_train, train_images_path, train_masks_path, batch_size)
    val_generator = generate_images_batch_ISIC_2018(df_val, val_images_path, val_masks_path, batch_size)
                  
    opt = tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath_h5, verbose=1, save_best_only=True, monitor='val_mean_absolute_error', mode='min')]
    
    model.fit(train_generator, validation_data=val_generator, steps_per_epoch=len(df_train)//batch_size, validation_steps=len(df_val)//batch_size, epochs=epochs, callbacks=callbacks)

    best_model = tf.keras.models.load_model(filepath_h5)

    results = best_model.evaluate(val_generator, steps=len(df_val)//batch_size)

    mse = results[0]
    mae = results[1]

    return mse, mae


def train_evalnet_multiclass(model, train_main_path, val_main_path, filepath_h5, batch_size, num_classes, epochs):
    '''
    Train the EvalNet model for multiclass segmentation and evaluate its performance.

    Args:
        model: The EvalNet model to be trained for multiclass segmentation.
        train_main_path (str): Directory containing training data, images, masks, and labels.
        val_main_path (str): Directory containing validation data, images, masks, and labels.
        filepath_h5 (str): Path to save the best model.
        batch_size (int): Batch size for training and validation.
        num_classes (int): Number of classes in the segmentation.
        epochs (int): Number of epochs to train the model.

    Returns:
        tuple: A tuple containing the mean squared error (mse) and mean absolute error (mae) on the validation dataset.
    '''
    
    train_images_path = os.path.join(train_main_path, 'images')
    train_masks_path = os.path.join(train_main_path, 'masks')

    val_images_path = os.path.join(val_main_path, 'images')
    val_masks_path = os.path.join(val_main_path, 'masks')

    df_train = pd.read_csv(os.path.join(train_main_path, 'labels.csv'), header=None, sep=';')
    df_val = pd.read_csv(os.path.join(val_main_path, 'labels.csv'), header=None, sep=';')
    
    train_generator = generate_images_batch_multiclass(df_train, train_images_path, train_masks_path, num_classes, batch_size)
    val_generator = generate_images_batch_multiclass(df_val, val_images_path, val_masks_path, num_classes, batch_size)
                  
    opt = tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath_h5, verbose=1, save_best_only=True, monitor='val_mean_absolute_error', mode='min')]
    
    model.fit(train_generator, validation_data=val_generator, steps_per_epoch=len(df_train)//batch_size, validation_steps=len(df_val)//batch_size, epochs=epochs, callbacks=callbacks)

    best_model = tf.keras.models.load_model(filepath_h5)

    results = best_model.evaluate(val_generator, steps=len(df_val)//batch_size)

    mse = results[0]
    mae = results[1]

    return mse, mae



def train_evalnet_miou_multiclass(segnet_models, 
                                  evalnet_model, 
                                  evalnet_name,
                                  h, w, c, 
                                  train_labeled_main_dir_train, 
                                  main_dir_evalnet_train, 
                                  train_labeled_main_dir_val, 
                                  main_dir_evalnet_val, 
                                  num_loops_train, 
                                  num_loops_val,  
                                  model_dir, 
                                  csv_dir, 
                                  batch_size, 
                                  num_classes, 
                                  epochs,
                                  runid,
                                  gen):

    '''
    Train the EvalNet model for multiclass segmentation using mIoU scores and class detection, and save the top performing models.

    Args:
        segnet_models (list): List of SegNet models for ensemble prediction.
        evalnet_model: The EvalNet model to be trained for multiclass segmentation.
        evalnet_name (str): Name of the EvalNet model.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        train_labeled_main_dir_train (str): Directory containing labeled training data.
        main_dir_evalnet_train (str): Directory for storing EvalNet training data.
        train_labeled_main_dir_val (str): Directory containing labeled validation data.
        main_dir_evalnet_val (str): Directory for storing EvalNet validation data.
        num_loops_train (int): Number of loops for generating training data.
        num_loops_val (int): Number of loops for generating validation data.
        model_dir (str): Directory to save trained models.
        csv_dir (str): Directory to save CSV results.
        batch_size (int): Batch size for training.
        num_classes (int): Number of classes in the segmentation.
        epochs (int): Number of epochs to train the model.
        runid (str): Unique identifier for the training run.
        gen (int): Generation number.

    Returns:
        None
    '''                                      



    create_training_data_evalnet_miou_im_multiclass(segnet_models,
                                                 h, w, c,
                                                 num_classes,
                                                 os.path.join(train_labeled_main_dir_train, 'images'),
                                                 os.path.join(train_labeled_main_dir_train, 'masks'),
                                                 main_dir_evalnet_train,
                                                 num_loops_train)
    
    create_training_data_evalnet_miou_im_multiclass(segnet_models,
                                                 h, w, c,
                                                 num_classes,
                                                 os.path.join(train_labeled_main_dir_val, 'images'),
                                                 os.path.join(train_labeled_main_dir_val, 'masks'),
                                                 main_dir_evalnet_val,
                                                 num_loops_val)
    
    
    modelname_evalnet_benchmarks = []

    initial_weights_evalnet = evalnet_model.get_weights()
    
    for i in range(0,5):
    
        modelname_evalnet_im = f'{evalnet_name}_{runid}_gen{gen}_{i}'
        model_filepath_h5 = os.path.join(model_dir, f'{modelname_evalnet_im}.h5')

        evalnet_model.set_weights(initial_weights_evalnet)
        
        total_loss, iou_loss, detection_loss, iou_mae, detection_acc = train_evalnet_miou_model_multiclass(evalnet_model, 
                                                                                                 h, w, 
                                                                                                 main_dir_evalnet_train, 
                                                                                                 main_dir_evalnet_val, 
                                                                                                 model_filepath_h5, 
                                                                                                 batch_size, 
                                                                                                 num_classes, 
                                                                                                 epochs)  
    
        modelname_evalnet_benchmarks.append((modelname_evalnet_im, total_loss, iou_loss, detection_loss, iou_mae, detection_acc))
    
        tf.keras.backend.clear_session()
        gc.collect()
    
    
    sorted_mae = sorted(modelname_evalnet_benchmarks, key=lambda x: x[1], reverse=False)
    
    top_K_maes = sorted_mae[:4] #TOP_Ks
    
    print(top_K_maes)
    
    
    for i, top_k in enumerate(top_K_maes, start=1):
        old_filename = os.path.join(model_dir, f'{top_k[0]}.h5')
        new_filename = os.path.join(model_dir, f'{top_k[0][:-2]}_topK_{i}.h5')
        
        os.rename(old_filename, new_filename)
    
    
    Header = ['modelname', 'total_loss', 'iou_loss', 'detection_loss', 'iou_mae', 'detection_acc']
    
    os.makedirs(csv_dir, exist_ok=True)
    
    with open(os.path.join(csv_dir, f'results_{modelname_evalnet_im}.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(Header)
        for row in modelname_evalnet_benchmarks:
            writer.writerow(row)



def train_evalnet_miou_model_hela(model, train_main_path, val_main_path, filepath_h5, batch_size, epochs):
    '''
    Train the EvalNet model for the HeLa dataset using mIoU scores and class detection, and evaluate its performance.

    Args:
        model: The EvalNet model to be trained for HeLa cell segmentation.
        train_main_path (str): Directory containing training data, images, masks, and labels.
        val_main_path (str): Directory containing validation data, images, masks, and labels.
        filepath_h5 (str): Path to save the best model.
        batch_size (int): Batch size for training and validation.
        epochs (int): Number of epochs to train the model.

    Returns:
        tuple: A tuple containing total loss, IoU loss, detection loss, IoU mean absolute error, and detection accuracy on the validation dataset.
    '''
    
    train_bf_images_path = os.path.join(train_main_path, 'brightfield')
    train_alive_masks_path = os.path.join(train_main_path, 'alive')
    train_dead_masks_path = os.path.join(train_main_path, 'dead')
    train_pos_masks_path = os.path.join(train_main_path, 'mod_position')

    val_bf_images_path = os.path.join(val_main_path, 'brightfield')
    val_alive_masks_path = os.path.join(val_main_path, 'alive')
    val_dead_masks_path = os.path.join(val_main_path, 'dead')
    val_pos_masks_path = os.path.join(val_main_path, 'mod_position')

    df_train = pd.read_csv(os.path.join(train_main_path, 'labels.csv'), header=None, sep=';')
    df_val = pd.read_csv(os.path.join(val_main_path, 'labels.csv'), header=None, sep=';')
    
    train_generator = generate_images_batch_evalnet_miou_hela(df_train, train_bf_images_path, train_alive_masks_path, train_dead_masks_path, train_pos_masks_path, batch_size)
    val_generator = generate_images_batch_evalnet_miou_hela(df_val, val_bf_images_path, val_alive_masks_path, val_dead_masks_path, val_pos_masks_path, batch_size)        

    opt = tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
    model.compile(optimizer=opt, loss=['mse', 'binary_crossentropy'], metrics=[['mae'], ['acc']])
    
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath_h5, verbose=1, save_best_only=True, monitor='val_loss', mode='min')]   
    
    model.fit(train_generator, validation_data=val_generator, steps_per_epoch=len(df_train)//batch_size, validation_steps=len(df_val)//batch_size, epochs=epochs, callbacks=callbacks)

    best_model = tf.keras.models.load_model(filepath_h5)

    results = best_model.evaluate(val_generator, steps=len(df_val)//batch_size)

    total_loss = results[0]
    iou_loss = results[1]
    detection_loss = results[2]
    iou_mae = results[3]
    detection_acc = results[4]

    return total_loss, iou_loss, detection_loss, iou_mae, detection_acc



def train_evalnet_miou_model_multiclass(model, h, w, train_main_path, val_main_path, filepath_h5, batch_size, num_classes, epochs):
    '''
    Train the EvalNet model for multiclass segmentation using mIoU scores and class detection, and evaluate its performance.

    Args:
        model: The EvalNet model to be trained for multiclass segmentation.
        h (int): Height of the input images.
        w (int): Width of the input images.
        train_main_path (str): Directory containing training data, images, masks, and labels.
        val_main_path (str): Directory containing validation data, images, masks, and labels.
        filepath_h5 (str): Path to save the best model.
        batch_size (int): Batch size for training and validation.
        num_classes (int): Number of classes in the segmentation.
        epochs (int): Number of epochs to train the model.

    Returns:
        tuple: A tuple containing total loss, IoU loss, detection loss, IoU mean absolute error, and detection accuracy on the validation dataset.
    '''
    
    train_images_path = os.path.join(train_main_path, 'images')
    train_masks_path = os.path.join(train_main_path, 'masks')

    val_images_path = os.path.join(val_main_path, 'images')
    val_masks_path = os.path.join(val_main_path, 'masks')

    df_train = pd.read_csv(os.path.join(train_main_path, 'labels.csv'), header=None, sep=';')
    df_val = pd.read_csv(os.path.join(val_main_path, 'labels.csv'), header=None, sep=';')
    
    train_generator = generate_images_batch_evalnet_miou_multiclass(df_train, h, w, train_images_path, train_masks_path, num_classes, batch_size)
    val_generator = generate_images_batch_evalnet_miou_multiclass(df_val, h, w, val_images_path, val_masks_path, num_classes, batch_size)
                  
    opt = tfa.optimizers.AdamW(learning_rate=LR, weight_decay=WD)
    model.compile(optimizer=opt, loss=['mse', 'binary_crossentropy'], metrics=[['mae'], ['acc']])
    
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath_h5, verbose=1, save_best_only=True, monitor='val_loss', mode='min')]   
    
    model.fit(train_generator, validation_data=val_generator, steps_per_epoch=len(df_train)//batch_size, validation_steps=len(df_val)//batch_size, epochs=epochs, callbacks=callbacks)

    best_model = tf.keras.models.load_model(filepath_h5)

    results = best_model.evaluate(val_generator, steps=len(df_val)//batch_size)

    total_loss = results[0]
    iou_loss = results[1]
    detection_loss = results[2]
    iou_mae = results[3]
    detection_acc = results[4]

    return total_loss, iou_loss, detection_loss, iou_mae, detection_acc



def generate_images_batch_ISIC_2018(dataframe, images_path, masks_path, batch_size=32):
    '''
    Generator to yield batches of images and masks from the ISIC 2018 dataset along with their labels for training.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image names and labels.
        images_path (str): Path to the directory containing images.
        masks_path (str): Path to the directory containing corresponding masks.
        batch_size (int, optional): Size of the batch to generate. Defaults to 32.

    Yields:
        tuple: A tuple containing a batch of images and masks, and their corresponding labels.
    '''
    
    while True:

        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(dataframe), batch_size):
            batch_images = []
            batch_gt = []
            batch_labels = []
            for j in range(i, min(i+batch_size, len(dataframe))):
                mask_name = dataframe.iloc[j, 0]
                label = dataframe.iloc[j, 1]

                # Determine the corresponding color image name

                if '___' in mask_name:
                    color_image_name = mask_name.split('___')[0] + '.png'
                else:
                    color_image_name = mask_name
            
                # Load and preprocess the image
                image = img_to_array(load_img(os.path.join(images_path, color_image_name)))
                batch_images.append(image)

                # Load and preprocess the gt
                gt = img_to_array(load_img(os.path.join(masks_path, mask_name), color_mode='grayscale'))
                batch_gt.append(gt)

                batch_labels.append(label)

            yield [np.array(batch_images), np.array(batch_gt)], np.array(batch_labels)


def generate_images_batch_evalnet_miou_hela(dataframe, bf_images_path, alive_masks_path, dead_masks_path, pos_masks_path, batch_size=32, num_classes=3):
    '''
    Generator to yield batches of the HeLa dataset images and their corresponding masks along with IoU and detection labels for EvalNet training.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image names and labels.
        bf_images_path (str): Path to the directory containing brightfield images.
        alive_masks_path (str): Path to the directory containing alive masks.
        dead_masks_path (str): Path to the directory containing dead masks.
        pos_masks_path (str): Path to the directory containing positional masks.
        batch_size (int, optional): Size of the batch to generate. Defaults to 32.
        num_classes (int): Number of classes (excluding background) in the segmentation.

    Yields:
        tuple: A tuple containing a batch of images and masks, and their corresponding IoU and detection labels.
    '''
    
    while True:

        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(dataframe), batch_size):
            batch_images = []
            batch_gt = []
            #batch_labels = []
            batch_labels_iou = []  
            batch_labels_detect = []

            for j in range(i, min(i+batch_size, len(dataframe))):
                mask_name = dataframe.iloc[j, 0]

                labels = dataframe.iloc[j, 1:1+2*num_classes].values

                iou_labels = labels[:num_classes]
                detect_labels = labels[num_classes:]

                batch_labels_iou.append(iou_labels.astype(np.float32))
                batch_labels_detect.append(detect_labels.astype(np.float32))

                # Determine the corresponding bf image name

                if '___' in mask_name:
                    image_name = mask_name.split('___')[0] + '.png'
                else:
                    image_name = mask_name
            
                bf_image = img_to_array(load_img(os.path.join(bf_images_path, image_name), color_mode='grayscale'))
                batch_images.append(bf_image)

                gt_alive = cv2.imread(os.path.join(alive_masks_path, mask_name), 0)
                gt_dead = cv2.imread(os.path.join(dead_masks_path, mask_name), 0)
                gt_pos = cv2.imread(os.path.join(pos_masks_path, mask_name), 0)

                # Stack along the third dimension
                gt = np.stack((gt_alive, gt_dead, gt_pos), axis=-1)

                batch_gt.append(gt)

                #batch_labels.append(labels.astype(np.float32))

            yield [np.array(batch_images), np.array(batch_gt)], [np.array(batch_labels_iou), np.array(batch_labels_detect)]



def generate_images_batch_multiclass(dataframe, images_path, masks_path, num_classes, batch_size=32):
    '''
    Generator to yield batches of images and one-hot encoded masks for multiclass segmentation along with their labels.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image names and labels.
        images_path (str): Path to the directory containing images.
        masks_path (str): Path to the directory containing corresponding masks.
        num_classes (int): Number of classes in the segmentation.
        batch_size (int, optional): Size of the batch to generate. Defaults to 32.

    Yields:
        tuple: A tuple containing a batch of images and one-hot encoded masks, and their corresponding labels.
    '''
    
    while True:

        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(dataframe), batch_size):
            batch_images = []
            batch_masks = []
            batch_labels = []
            for j in range(i, min(i+batch_size, len(dataframe))):
                mask_name = dataframe.iloc[j, 0]
                label = dataframe.iloc[j, 1]

                # Determine the corresponding color image name

                if '___' in mask_name:
                    color_image_name = mask_name.split('___')[0] + '.png'
                else:
                    color_image_name = mask_name
            
                # Load and preprocess the image
                image = img_to_array(load_img(os.path.join(images_path, color_image_name)))
                batch_images.append(image)

                # Load and preprocess the mask
                mask = img_to_array(load_img(os.path.join(masks_path, mask_name), color_mode='grayscale'))
                mask = mask.reshape((256, 256))
                one_hot_encoded = np.stack([(mask == cls).astype(int) for cls in range(num_classes)], axis=-1)
                batch_masks.append(one_hot_encoded)

                batch_labels.append(label)

            yield [np.array(batch_images), np.array(batch_masks)], np.array(batch_labels)



def generate_images_batch_evalnet_miou_multiclass(dataframe, h, w, images_path, masks_path, num_classes, batch_size=32):
    '''
    Generator to yield batches of images and one-hot encoded masks for multiclass segmentation along with IoU and detection labels for EvalNet training.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image names and labels.
        h (int): Height of the input images.
        w (int): Width of the input images.
        images_path (str): Path to the directory containing images.
        masks_path (str): Path to the directory containing corresponding masks.
        num_classes (int): Number of classes in the segmentation.
        batch_size (int, optional): Size of the batch to generate. Defaults to 32.

    Yields:
        tuple: A tuple containing a batch of images and one-hot encoded masks, and their corresponding IoU and detection labels.
    '''
    
    while True:

        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(dataframe), batch_size):
            batch_images = []
            batch_masks = []
            batch_labels = []
            for j in range(i, min(i+batch_size, len(dataframe))):
                mask_name = dataframe.iloc[j, 0]

                # Extract 2*num_classes labels for each image
                labels = dataframe.iloc[j, 1:1+2*num_classes].values

                # Determine the corresponding color image name
                if '___' in mask_name:
                    color_image_name = mask_name.split('___')[0] + '.png'
                else:
                    color_image_name = mask_name
            
                # Load and preprocess the image
                image = img_to_array(load_img(os.path.join(images_path, color_image_name)))
                batch_images.append(image)

                # Load and preprocess the mask
                mask = img_to_array(load_img(os.path.join(masks_path, mask_name), color_mode='grayscale'))
                mask = mask.reshape((h, w))
                one_hot_encoded = np.stack([(mask == cls).astype(int) for cls in range(num_classes)], axis=-1)
                batch_masks.append(one_hot_encoded)

                batch_labels.append(labels)

            batch_labels_array = np.array(batch_labels).astype(np.float32)
            yield [np.array(batch_images), np.array(batch_masks)], [batch_labels_array[:, :num_classes], batch_labels_array[:, num_classes:]]






def create_training_data_for_segnet_ISIC_2018(evalnet_model, h,w,c , images_path, mask_paths, main_output_path, threshold, last_gen_main_path='', rgb=True):
    '''
    Create training data for segnet (Student U-Net) by selecting images and masks with the highest IoU scores predicted by EvalNet.

    Args:
        evalnet_model: EvalNet model used to predict IoU scores.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        images_path (str): Path to the directory containing images.
        mask_paths (list[str]): List of paths to directories containing different sets of masks.
        main_output_path (str): Path to the directory where the selected images and masks will be saved.
        threshold (float): IoU score threshold for selecting an image-mask pair.
        last_gen_main_path (str, optional): Path to the directory containing images and masks from the last generation. Defaults to an empty string.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.

    Returns:
        None
    '''

    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    images_path_last_gen = os.path.join(last_gen_main_path, 'images')
    masks_path_last_gen = os.path.join(last_gen_main_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)

    
    if last_gen_main_path != '':
        for imagename in os.listdir(images_path_last_gen):
            shutil.copy(os.path.join(images_path_last_gen, imagename), os.path.join(images_path_out, imagename))
            shutil.copy(os.path.join(masks_path_last_gen, imagename), os.path.join(masks_path_out, imagename))


    for imagename in tqdm(os.listdir(images_path)):  
    
        masks = []

        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        for maskspath in mask_paths:
            mask = cv2.imread(os.path.join(maskspath, imagename))
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            masks.append(mask_gray)

        path_gt_last_gen_mask = os.path.join(masks_path_out, imagename)
        if os.path.isfile(path_gt_last_gen_mask):
            mask = cv2.imread(path_gt_last_gen_mask)
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            masks.append(mask_gray)

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))
        preparedMasks = [np.array(mask.reshape(-1, h, w, 1), dtype=np.uint8) for mask in masks]
        prepared_images = np.repeat(prepared_image, len(preparedMasks), axis=0)
        preparedMasks = np.concatenate(preparedMasks, axis=0)
        
        with contextlib.redirect_stdout(io.StringIO()):
            pred_ious = evalnet_model.predict([prepared_images, preparedMasks])

        best_iou_index = np.argmax(pred_ious)
        best_iou = pred_ious[best_iou_index]

        if best_iou >= threshold:

            best_mask = masks[best_iou_index]

            shutil.copy(os.path.join(images_path, imagename), os.path.join(images_path_out, imagename))
            cv2.imwrite(os.path.join(masks_path_out, imagename), best_mask)






def create_training_data_for_segnet_with_ensemble_binary(evalnets, h,w,c , images_path, mask_paths, main_output_path, threshold, last_gen_main_path='', rgb=True):
    '''
    Create training data for segnet (Student U-Net) using an ensemble of EvalNet models to select images and masks with the highest average IoU scores.

    Args:
        evalnets (list): List of EvalNet models used to predict IoU scores.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        images_path (str): Path to the directory containing images.
        mask_paths (list[str]): List of paths to directories containing different sets of masks.
        main_output_path (str): Path to the directory where the selected images and masks will be saved.
        threshold (float): Average IoU score threshold for selecting an image-mask pair.
        last_gen_main_path (str, optional): Path to the directory containing images and masks from the last generation. Defaults to an empty string.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.

    Returns:
        None
    '''
    
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    images_path_last_gen = os.path.join(last_gen_main_path, 'images')
    masks_path_last_gen = os.path.join(last_gen_main_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)

    
    if last_gen_main_path != '':
        for imagename in os.listdir(images_path_last_gen):
            shutil.copy(os.path.join(images_path_last_gen, imagename), os.path.join(images_path_out, imagename))
            shutil.copy(os.path.join(masks_path_last_gen, imagename), os.path.join(masks_path_out, imagename))


    for imagename in tqdm(os.listdir(images_path)):  
    
        masks = []

        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        for maskspath in mask_paths:
            mask = cv2.imread(os.path.join(maskspath, imagename))
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            masks.append(mask_gray)

        path_gt_last_gen_mask = os.path.join(masks_path_out, imagename)
        if os.path.isfile(path_gt_last_gen_mask):
            mask = cv2.imread(path_gt_last_gen_mask)
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            masks.append(mask_gray)

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))
        preparedMasks = [np.array(mask.reshape(-1, h, w, 1), dtype=np.uint8) for mask in masks]
        prepared_images = np.repeat(prepared_image, len(preparedMasks), axis=0)
        preparedMasks = np.concatenate(preparedMasks, axis=0)
        

        pred_ious_lists = []

        for model in evalnets:
            with contextlib.redirect_stdout(io.StringIO()):
                pred_ious = model.predict([prepared_images, preparedMasks])
                pred_ious_lists.append(pred_ious)

        pred_ious_lists_stacked = np.stack(pred_ious_lists, axis=0)

        mean_pred_ious = np.mean(pred_ious_lists_stacked, axis=0)


        best_iou_index = np.argmax(mean_pred_ious)
        best_iou = mean_pred_ious[best_iou_index]

        if best_iou >= threshold:

            best_mask = masks[best_iou_index]

            shutil.copy(os.path.join(images_path, imagename), os.path.join(images_path_out, imagename))
            cv2.imwrite(os.path.join(masks_path_out, imagename), best_mask)


            


def create_training_data_for_segnet_multiclass(evalnet_model, h,w,c, num_classes, images_path, mask_paths, main_output_path, threshold, last_gen_main_path='', rgb=True):
    '''
    Create training data for segnet (Student U-Net) for multiclass segmentation by selecting images and masks with the highest IoU scores predicted by EvalNet.

    Args:
        evalnet_model: EvalNet model used to predict IoU scores for multiclass segmentation.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        num_classes (int): Number of classes in the segmentation.
        images_path (str): Path to the directory containing images.
        mask_paths (list[str]): List of paths to directories containing different sets of masks.
        main_output_path (str): Path to the directory where the selected images and masks will be saved.
        threshold (float): IoU score threshold for selecting an image-mask pair.
        last_gen_main_path (str, optional): Path to the directory containing images and masks from the last generation. Defaults to an empty string.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.

    Returns:
        None
    '''

    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    images_path_last_gen = os.path.join(last_gen_main_path, 'images')
    masks_path_last_gen = os.path.join(last_gen_main_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)

    
    if last_gen_main_path != '':
        for imagename in os.listdir(images_path_last_gen):
            shutil.copy(os.path.join(images_path_last_gen, imagename), os.path.join(images_path_out, imagename))
            shutil.copy(os.path.join(masks_path_last_gen, imagename), os.path.join(masks_path_out, imagename))


    for imagename in tqdm(os.listdir(images_path)):  
    
        masks = []

        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        for maskspath in mask_paths:
            mask_gray = cv2.imread(os.path.join(maskspath, imagename), 0)
            masks.append(mask_gray)

        path_gt_last_gen_mask = os.path.join(masks_path_out, imagename)
        if os.path.isfile(path_gt_last_gen_mask):
            mask_gray = cv2.imread(path_gt_last_gen_mask, 0)
            masks.append(mask_gray)

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))
        prepared_images = np.repeat(prepared_image, len(masks), axis=0)

        masks_array = np.array(masks).reshape(-1, h, w, 1)
        preparedMasks_one_hot = np.stack([(masks_array == cls).astype(np.int32) for cls in range(num_classes)], axis=-1).squeeze(axis=-2)
        
        with contextlib.redirect_stdout(io.StringIO()):
            pred_ious = evalnet_model.predict([prepared_images, preparedMasks_one_hot])


        best_iou_index = np.argmax(pred_ious)
        best_iou = pred_ious[best_iou_index]

        if best_iou >= threshold:

            best_mask = masks[best_iou_index]

            shutil.copy(os.path.join(images_path, imagename), os.path.join(images_path_out, imagename))
            cv2.imwrite(os.path.join(masks_path_out, imagename), best_mask)




def create_training_data_for_segnet_with_ensemble_multiclass(evalnets, h,w,c, num_classes, images_path, mask_paths, main_output_path, threshold, last_gen_main_path='', rgb=True):
    '''
    Create training data for segnet (Student U-Net) for multiclass segmentation using an ensemble of EvalNet models to select the best image-mask pairs based on the highest average IoU scores.

    Args:
        evalnets (list): List of EvalNet models used to predict IoU scores.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        num_classes (int): Number of classes in the segmentation.
        images_path (str): Path to the directory containing images.
        mask_paths (list[str]): List of paths to directories containing different sets of masks.
        main_output_path (str): Path to the directory where the selected images and masks will be saved.
        threshold (float): Average IoU score threshold for selecting an image-mask pair.
        last_gen_main_path (str, optional): Path to the directory containing images and masks from the last generation. Defaults to an empty string.
        rgb (bool, optional): Flag to convert images to RGB. Defaults to True.

    Returns:
        None
    '''
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    images_path_last_gen = os.path.join(last_gen_main_path, 'images')
    masks_path_last_gen = os.path.join(last_gen_main_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)

    
    if last_gen_main_path != '':
        for imagename in os.listdir(images_path_last_gen):
            shutil.copy(os.path.join(images_path_last_gen, imagename), os.path.join(images_path_out, imagename))
            shutil.copy(os.path.join(masks_path_last_gen, imagename), os.path.join(masks_path_out, imagename))


    for imagename in tqdm(os.listdir(images_path)):  
    
        masks = []

        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        for maskspath in mask_paths:
            mask_gray = cv2.imread(os.path.join(maskspath, imagename), 0)
            masks.append(mask_gray)

        path_gt_last_gen_mask = os.path.join(masks_path_out, imagename)
        if os.path.isfile(path_gt_last_gen_mask):
            mask_gray = cv2.imread(path_gt_last_gen_mask, 0)
            masks.append(mask_gray)

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))
        prepared_images = np.repeat(prepared_image, len(masks), axis=0)

        masks_array = np.array(masks).reshape(-1, h, w, 1)
        preparedMasks_one_hot = np.stack([(masks_array == cls).astype(np.int32) for cls in range(num_classes)], axis=-1).squeeze(axis=-2)
        

        pred_ious_lists = []

        for model in evalnets:
            with contextlib.redirect_stdout(io.StringIO()):
                pred_ious = model.predict([prepared_images, preparedMasks_one_hot])
                pred_ious_lists.append(pred_ious)

        pred_ious_lists_stacked = np.stack(pred_ious_lists, axis=0)

        mean_pred_ious = np.mean(pred_ious_lists_stacked, axis=0)


        best_iou_index = np.argmax(mean_pred_ious)
        best_iou = mean_pred_ious[best_iou_index]

        if best_iou >= threshold:

            best_mask = masks[best_iou_index]

            shutil.copy(os.path.join(images_path, imagename), os.path.join(images_path_out, imagename))
            cv2.imwrite(os.path.join(masks_path_out, imagename), best_mask)





def create_training_data_for_segnet_with_miou_ensemble_hela(evalnets, h,w,c, bf_images_path_in, mask_paths_in, main_output_path, threshold, last_gen_main_path='', max_pos_circle_size=8, min_pos_circle_size=3):
    '''
    Create training data for segnet (Student U-Net) for the HeLa dataset using an ensemble of EvalNet models with mean IoU (Intersection over Union) evaluation. 
    This function selects the best image-mask pairs based on the highest average IoU scores across multiple predictions.

    Args:
        evalnets (list): List of EvalNet models used to predict IoU scores and detection confidence.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images (typically 1 for grayscale images).
        bf_images_path_in (str): Path to the directory containing brightfield images.
        mask_paths_in (list[str]): List of paths to directories containing different sets of masks for alive, dead, and position.
        main_output_path (str): Path to the directory where the selected images and masks will be saved.
        threshold (float): Mean IoU score threshold for selecting an image-mask pair.
        last_gen_main_path (str, optional): Path to the directory containing images and masks from the last generation. Defaults to an empty string.
        max_pos_circle_size (int, optional): Maximum size of the circle to represent a cell's position. Defaults to 8.
        min_pos_circle_size (int, optional): Minimum size of the circle to represent a cell's position. Defaults to 3.

    Returns:
        None: The function saves the selected images and masks in the specified output directory.
    '''
    
    bf_images_path_out = os.path.join(main_output_path, 'brightfield')
    alive_masks_path_out = os.path.join(main_output_path, 'alive')
    dead_masks_path_out = os.path.join(main_output_path, 'dead')
    pos_masks_path_out = os.path.join(main_output_path, 'mod_position')

    bf_images_path_last_gen_in = os.path.join(last_gen_main_path, 'brightfield')
    alive_masks_path_last_gen_in = os.path.join(last_gen_main_path, 'alive')
    dead_masks_path_last_gen_in = os.path.join(last_gen_main_path, 'dead')
    pos_masks_path_last_gen_in = os.path.join(last_gen_main_path, 'mod_position')
    
    os.makedirs(bf_images_path_out, exist_ok=True)
    os.makedirs(alive_masks_path_out, exist_ok=True)
    os.makedirs(dead_masks_path_out, exist_ok=True)
    os.makedirs(pos_masks_path_out, exist_ok=True)

    
    if last_gen_main_path != '':
        for imagename in os.listdir(bf_images_path_last_gen_in):
            shutil.copy(os.path.join(bf_images_path_last_gen_in, imagename), os.path.join(bf_images_path_out, imagename))
            shutil.copy(os.path.join(alive_masks_path_last_gen_in, imagename), os.path.join(alive_masks_path_out, imagename))
            shutil.copy(os.path.join(dead_masks_path_last_gen_in, imagename), os.path.join(dead_masks_path_out, imagename))
            shutil.copy(os.path.join(pos_masks_path_last_gen_in, imagename), os.path.join(pos_masks_path_out, imagename))


    for imagename in tqdm(os.listdir(bf_images_path_in)):  
    
        stacked_masks = []

        image_gray = cv2.imread(os.path.join(bf_images_path_in, imagename), 0)

        for masks_path in mask_paths_in:
            alive_mask = cv2.imread(os.path.join(masks_path, 'alive', imagename), 0) / 255.0
            dead_mask = cv2.imread(os.path.join(masks_path, 'dead', imagename), 0) / 255.0
            pos_mask = cv2.imread(os.path.join(masks_path, 'mod_position', imagename), 0) / 255.0
            stacked_mask = np.stack((alive_mask, dead_mask, pos_mask), axis=-1)

            stacked_masks.append(stacked_mask)


        if last_gen_main_path != '':

            if os.path.exists(os.path.join(alive_masks_path_out, imagename)):
                alive_mask = cv2.imread(os.path.join(alive_masks_path_out, imagename), 0) / 255.0
                dead_mask = cv2.imread(os.path.join(dead_masks_path_out, imagename), 0) / 255.0
                pos_mask = cv2.imread(os.path.join(pos_masks_path_out, imagename), 0) / 255.0
                stacked_mask = np.stack((alive_mask, dead_mask, pos_mask), axis=-1)

                stacked_masks.append(stacked_mask)

        prepared_image = (np.array((image_gray).reshape(-1, h, w, c), dtype=np.uint8))
        prepared_images = np.repeat(prepared_image, len(stacked_masks), axis=0)

        prepared_masks = np.array(stacked_masks).reshape(-1, h, w, 3)        

        pred_ious_lists = []
        pred_detection_lists = []

        for model in evalnets:
            with contextlib.redirect_stdout(io.StringIO()):
                predictions = model.predict([prepared_images, prepared_masks])
                pred_ious = predictions[0]
                pred_detection = predictions[1]

                pred_ious_lists.append(pred_ious)
                pred_detection_lists.append(pred_detection)


        pred_ious_lists_stacked = np.stack(pred_ious_lists, axis=0)
        pred_detection_lists_stacked = np.stack(pred_detection_lists, axis=0)


        mIoU_list = []
        mean_ious_per_class = np.mean(pred_ious_lists_stacked, axis=0)
        mean_conf_per_class = np.mean(pred_detection_lists_stacked, axis=0)
         
        # Iterate over each image
        for i in range(mean_ious_per_class.shape[0]):
            valid_iou_values = []
        
            # Iterate over each class
            for class_idx in range(mean_ious_per_class.shape[1]):
                if mean_conf_per_class[i, class_idx] >= 0.5:
                    valid_iou_values.append(mean_ious_per_class[i, class_idx])
        
            # Compute mean IoU for the current image
            if valid_iou_values:  # Check if the list is not empty
                mIoU = sum(valid_iou_values) / len(valid_iou_values)
                mIoU_list.append(mIoU)
            else:
                mIoU_list.append(0.0)  # If no class passed the conf check, append 0
        
        
        best_miou_index = np.argmax(mIoU_list)
        best_miou = mIoU_list[best_miou_index]
        
        if best_miou >= threshold:

            best_masks = prepared_masks[best_miou_index]
            best_alive = best_masks[:, :, 0]*255
            best_dead = best_masks[:, :, 1]*255
            best_pos_temp = best_masks[:, :, 2]*255

            positions = get_pos_contours(best_pos_temp)
            
            final_pos_mask = np.zeros((h, w, 3), np.uint8)
                       
            for pos in positions:
                if len(positions) > 1:
                    min_dist = get_min_dist(pos, positions)
                else:
                    min_dist = 99
            
                circle_size = int(min_dist // 4)
                circle_size = max(min(circle_size, max_pos_circle_size), min_pos_circle_size)
                cv2.circle(final_pos_mask, (pos[0], pos[1]), circle_size, (255, 255, 255), -1)


            shutil.copy(os.path.join(bf_images_path_in, imagename), os.path.join(bf_images_path_out, imagename))
            cv2.imwrite(os.path.join(alive_masks_path_out, imagename), best_alive)
            cv2.imwrite(os.path.join(dead_masks_path_out, imagename), best_dead)
            cv2.imwrite(os.path.join(pos_masks_path_out, imagename), final_pos_mask)


def create_training_data_for_segnet_with_miou_ensemble_multiclass(evalnets, h,w,c, num_classes, images_path, mask_paths, main_output_path, threshold, last_gen_main_path='', rgb=True):
    '''
    Create training data for segnet (Student U-Net) for multiclass segmentation using an ensemble of EvalNet models with mean IoU evaluation. 
    This function selects the best image-mask pairs based on the highest average IoU scores across multiple predictions.

    Args:
        evalnets (list): List of EvalNet models used to predict IoU scores and detection confidence.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images (typically 1 for grayscale images).
        num_classes (int): Number of classes for segmentation.
        images_path (str): Path to the directory containing images.
        mask_paths (list[str]): List of paths to directories containing different sets of masks.
        main_output_path (str): Path to the directory where the selected images and masks will be saved.
        threshold (float): Mean IoU score threshold for selecting an image-mask pair.
        last_gen_main_path (str, optional): Path to the directory containing images and masks from the last generation. Defaults to an empty string.
        rgb (bool, optional): True if the images are in RGB format, False if grayscale. Defaults to True.

    Returns:
        None: The function saves the selected images and masks in the specified output directory.
    '''
    
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    images_path_last_gen = os.path.join(last_gen_main_path, 'images')
    masks_path_last_gen = os.path.join(last_gen_main_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)

    
    if last_gen_main_path != '':
        for imagename in os.listdir(images_path_last_gen):
            shutil.copy(os.path.join(images_path_last_gen, imagename), os.path.join(images_path_out, imagename))
            shutil.copy(os.path.join(masks_path_last_gen, imagename), os.path.join(masks_path_out, imagename))


    for imagename in tqdm(os.listdir(images_path)):  
    
        masks = []

        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        for maskspath in mask_paths:
            mask_gray = cv2.imread(os.path.join(maskspath, imagename), 0)
            masks.append(mask_gray)

        path_gt_last_gen_mask = os.path.join(masks_path_out, imagename)
        if os.path.isfile(path_gt_last_gen_mask):
            mask_gray = cv2.imread(path_gt_last_gen_mask, 0)
            masks.append(mask_gray)

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))
        prepared_images = np.repeat(prepared_image, len(masks), axis=0)

        masks_array = np.array(masks).reshape(-1, h, w, 1)
        preparedMasks_one_hot = np.stack([(masks_array == cls).astype(np.int32) for cls in range(num_classes)], axis=-1).squeeze(axis=-2)
        

        pred_ious_lists = []
        pred_detection_lists = []

        for model in evalnets:
            with contextlib.redirect_stdout(io.StringIO()):
                predictions = model.predict([prepared_images, preparedMasks_one_hot])
                pred_ious = predictions[0]
                pred_detection = predictions[1]

                pred_ious_lists.append(pred_ious)
                pred_detection_lists.append(pred_detection)


        pred_ious_lists_stacked = np.stack(pred_ious_lists, axis=0)
        pred_detection_lists_stacked = np.stack(pred_detection_lists, axis=0)


        mIoU_list = []
        mean_ious_per_class = np.mean(pred_ious_lists_stacked, axis=0)
        mean_conf_per_class = np.mean(pred_detection_lists_stacked, axis=0)
         
        # Iterate over each image
        for i in range(mean_ious_per_class.shape[0]):
            valid_iou_values = []
        
            # Iterate over each class
            for class_idx in range(mean_ious_per_class.shape[1]):
                if mean_conf_per_class[i, class_idx] >= 0.5:
                    valid_iou_values.append(mean_ious_per_class[i, class_idx])
        
            # Compute mean IoU for the current image
            if valid_iou_values:  # Check if the list is not empty
                mIoU = sum(valid_iou_values) / len(valid_iou_values)
                mIoU_list.append(mIoU)
            else:
                mIoU_list.append(0.0)  # If no class passed the conf check, append 0
        
        
        best_miou_index = np.argmax(mIoU_list)
        best_miou = mIoU_list[best_miou_index]
        
        if best_miou >= threshold:

            best_mask = masks[best_miou_index]

            shutil.copy(os.path.join(images_path, imagename), os.path.join(images_path_out, imagename))
            cv2.imwrite(os.path.join(masks_path_out, imagename), best_mask)





def create_training_data_by_evalnet_miou_for_segnet_multiclass(evalnet_model, h,w,c, num_classes, images_path, mask_paths, main_output_path, miou_threshold, last_gen_main_path='', rgb=True):
    '''
    Create training data for segnet (Student U-Net) for multiclass segmentation by selecting the best image-mask pairs based on mean IoU scores predicted by an EvalNet model.

    Args:
        evalnet_model (tf.keras.Model): EvalNet model used to predict IoU scores and detection confidence.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images (typically 1 for grayscale, 3 for RGB).
        num_classes (int): Number of classes for segmentation.
        images_path (str): Path to the directory containing images.
        mask_paths (list[str]): List of paths to directories containing different sets of masks.
        main_output_path (str): Path to the directory where the selected images and masks will be saved.
        miou_threshold (float): Threshold for mean IoU score to select an image-mask pair.
        last_gen_main_path (str, optional): Path to the directory containing images and masks from the last generation. Defaults to an empty string.
        rgb (bool, optional): True if the images are in RGB format, False if grayscale. Defaults to True.

    Returns:
        None: The function saves the selected images and masks in the specified output directory based on the miou_threshold criteria.
    '''
    
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    images_path_last_gen = os.path.join(last_gen_main_path, 'images')
    masks_path_last_gen = os.path.join(last_gen_main_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)

    
    if last_gen_main_path != '':
        for imagename in os.listdir(images_path_last_gen):
            shutil.copy(os.path.join(images_path_last_gen, imagename), os.path.join(images_path_out, imagename))
            shutil.copy(os.path.join(masks_path_last_gen, imagename), os.path.join(masks_path_out, imagename))


    for imagename in tqdm(os.listdir(images_path)):  
    
        masks = []

        image = cv2.imread(os.path.join(images_path, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        for maskspath in mask_paths:
            mask_gray = cv2.imread(os.path.join(maskspath, imagename), 0)
            masks.append(mask_gray)

        path_gt_last_gen_mask = os.path.join(masks_path_out, imagename)
        if os.path.isfile(path_gt_last_gen_mask):
            mask_gray = cv2.imread(path_gt_last_gen_mask, 0)
            masks.append(mask_gray)

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))
        prepared_images = np.repeat(prepared_image, len(masks), axis=0)

        masks_array = np.array(masks).reshape(-1, h, w, 1)
        preparedMasks_one_hot = np.stack([(masks_array == cls).astype(np.int32) for cls in range(num_classes)], axis=-1).squeeze(axis=-2)
        
        with contextlib.redirect_stdout(io.StringIO()):
            predictions = evalnet_model.predict([prepared_images, preparedMasks_one_hot])
            pred_ious = predictions[0]
            pred_conf = predictions[1]
        
        mIoU_list = []
        mean_conf_per_class = np.mean(pred_conf, axis=0)
        
        # Iterate over each image
        for i in range(pred_ious.shape[0]):
            valid_iou_values = []
        
            # Iterate over each class
            for class_idx in range(pred_ious.shape[1]):
                if mean_conf_per_class[class_idx] >= 0.03:
                    valid_iou_values.append(pred_conf[i, class_idx])
        
            # Compute mean IoU for the current image
            if valid_iou_values:  # Check if the list is not empty
                mIoU = sum(valid_iou_values) / len(valid_iou_values)
                mIoU_list.append(mIoU)
            else:
                mIoU_list.append(0.0)  # If no class passed the conf check, append 0


        best_miou_index = np.argmax(mIoU_list)
        best_miou = mIoU_list[best_miou_index]

        if best_miou >= miou_threshold:

            best_mask = masks[best_miou_index]

            shutil.copy(os.path.join(images_path, imagename), os.path.join(images_path_out, imagename))
            cv2.imwrite(os.path.join(masks_path_out, imagename), best_mask)






def create_augment_images_and_masks_with_evalnet_ensemble_binary(evalnets, h, w, c, min_threshold, max_threshold, main_input_path, main_output_path, brightness_range_alpha=(0.6, 1.4), brightness_range_beta=(-20, 20), max_blur=3, max_noise=20, free_rotation=True, rgb=True):
    '''
    Create and augment images and masks using an ensemble of binary EvalNet models.
    The number of augmentations is determined based on the mean IoU predicted by EvalNets.

    Args:
        evalnets (list): List of EvalNet models.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        min_threshold (float): Minimum IoU threshold for augmentation.
        max_threshold (float): Maximum IoU threshold for augmentation.
        main_input_path (str): Directory containing the original images and masks.
        main_output_path (str): Output directory for augmented images and masks.
        brightness_range_alpha (tuple): Range of alpha for brightness adjustment (default (0.6, 1.4)).
        brightness_range_beta (tuple): Range of beta for brightness adjustment (default (-20, 20)).
        max_blur (int): Maximum blur to apply (default 3).
        max_noise (int): Maximum noise to apply (default 20).
        free_rotation (bool): Flag to allow free rotation (default True).
        rgb (bool): Flag indicating if images are in RGB format (default True).

    Returns:
        None: Augmented images and masks are saved in the specified output directory.
    '''
    
    images_path_in = os.path.join(main_input_path, 'images')
    masks_path_in = os.path.join(main_input_path, 'masks')
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)


    for imagename in tqdm(os.listdir(images_path_in)):  

        image = cv2.imread(os.path.join(images_path_in, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        mask_gray = cv2.imread(os.path.join(masks_path_in, imagename), 0)

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))
        prepared_mask = np.array(mask_gray).reshape(-1, h, w, 1)

        pred_ious_lists = []

        for model in evalnets:
            with contextlib.redirect_stdout(io.StringIO()):
                pred_ious = model.predict([prepared_image, prepared_mask])
                pred_ious_lists.append(pred_ious)

        pred_ious_lists_stacked = np.stack(pred_ious_lists, axis=0)

        mean_pred_ious = np.mean(pred_ious_lists_stacked, axis=0)

        threshold_step = (max_threshold - min_threshold) / 5

        if mean_pred_ious > max_threshold:
            num_augs = 5
        elif mean_pred_ious > min_threshold:
            num_augs = 1 + int((mean_pred_ious - min_threshold) / threshold_step)
        else:
            num_augs = 1
        
        num_augs = min(num_augs, 5)

        for j in range(num_augs):
            aug_image, aug_mask = augment_image_and_mask(image, mask_gray, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation=free_rotation)

            cv2.imwrite(os.path.join(images_path_out, f'{imagename[:-4]}___{j}.png'), aug_image)
            cv2.imwrite(os.path.join(masks_path_out, f'{imagename[:-4]}___{j}.png'), aug_mask)




def create_augment_images_and_masks_with_evalnet_multiclass(evalnet_model, h, w, c, num_classes, min_threshold, max_threshold, main_input_path, main_output_path, brightness_range_alpha=(0.6, 1.4), brightness_range_beta=(-20, 20), max_blur=3, max_noise=20, free_rotation=False, rgb=True):
    '''
    Create and augment images and masks using EvalNet model for multiclass segmentation.
    The number of augmentations is determined based on the IoU predicted by EvalNet.

    Args:
        evalnet_model (tf.keras.Model): EvalNet model for prediction.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        num_classes (int): Number of classes in segmentation.
        min_threshold (float): Minimum IoU threshold for augmentation.
        max_threshold (float): Maximum IoU threshold for augmentation.
        main_input_path (str): Directory containing the original images and masks.
        main_output_path (str): Output directory for augmented images and masks.
        brightness_range_alpha (tuple): Range of alpha for brightness adjustment (default (0.6, 1.4)).
        brightness_range_beta (tuple): Range of beta for brightness adjustment (default (-20, 20)).
        max_blur (int): Maximum blur to apply (default 3).
        max_noise (int): Maximum noise to apply (default 20).
        free_rotation (bool): Flag to allow free rotation (default False).
        rgb (bool): Flag indicating if images are in RGB format (default True).

    Returns:
        None: Augmented images and masks are saved in the specified output directory.
    '''

    images_path_in = os.path.join(main_input_path, 'images')
    masks_path_in = os.path.join(main_input_path, 'masks')
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)


    for imagename in tqdm(os.listdir(images_path_in)):  

        image = cv2.imread(os.path.join(images_path_in, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        mask_gray = cv2.imread(os.path.join(masks_path_in, imagename), 0)

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))

        mask_array = np.array(mask_gray).reshape(-1, h, w, 1)
        preparedMasks_one_hot = np.stack([(mask_array == cls).astype(np.int32) for cls in range(num_classes)], axis=-1).squeeze(axis=-2)

        
        with contextlib.redirect_stdout(io.StringIO()):
            pred_iou = evalnet_model.predict([prepared_image, preparedMasks_one_hot])

        threshold_step = (max_threshold - min_threshold) / 5

        if pred_iou > max_threshold:
            num_augs = 5
        elif pred_iou > min_threshold:
            num_augs = 1 + int((pred_iou - min_threshold) / threshold_step)
        else:
            num_augs = 1
        
        num_augs = min(num_augs, 5)

        for j in range(num_augs):
            aug_image, aug_mask = augment_image_and_mask(image, mask_gray, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation=free_rotation)

            cv2.imwrite(os.path.join(images_path_out, f'{imagename[:-4]}___{j}.png'), aug_image)
            cv2.imwrite(os.path.join(masks_path_out, f'{imagename[:-4]}___{j}.png'), aug_mask)





def create_augment_images_and_masks_with_evalnet_ensemble_hela(evalnets, h, w, c, min_threshold, max_threshold, main_input_path, main_output_path, brightness_range_alpha=(0.6, 1.4), brightness_range_beta=(-20, 20), max_blur=3, max_noise=20, free_rotation=True):
    '''
    Create and augment images and masks for the Hela dataset using an ensemble of EvalNet models.
    The number of augmentations is based on the average IoU predicted by the EvalNet ensemble.

    Args:
        evalnets (list): List of EvalNet models for ensemble prediction.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        min_threshold (float): Minimum IoU threshold for augmentation.
        max_threshold (float): Maximum IoU threshold for augmentation.
        main_input_path (str): Directory containing the original images and masks.
        main_output_path (str): Output directory for augmented images and masks.
        brightness_range_alpha (tuple): Range of alpha for brightness adjustment (default (0.6, 1.4)).
        brightness_range_beta (tuple): Range of beta for brightness adjustment (default (-20, 20)).
        max_blur (int): Maximum blur to apply (default 3).
        max_noise (int): Maximum noise to apply (default 20).
        free_rotation (bool): Flag to allow free rotation (default True).

    Returns:
        None: Augmented images and masks are saved in the specified output directory.
    '''
    
    brightfield_path_in = os.path.join(main_input_path, 'brightfield')
    alive_path_in = os.path.join(main_input_path, 'alive')
    dead_path_in = os.path.join(main_input_path, 'dead')
    pos_path_in = os.path.join(main_input_path, 'mod_position')

    brightfield_path_out = os.path.join(main_output_path, 'brightfield')
    alive_path_out = os.path.join(main_output_path, 'alive')
    dead_path_out = os.path.join(main_output_path, 'dead')
    pos_path_out = os.path.join(main_output_path, 'mod_position')
    
    os.makedirs(brightfield_path_out, exist_ok=True)
    os.makedirs(alive_path_out, exist_ok=True)
    os.makedirs(dead_path_out, exist_ok=True)
    os.makedirs(pos_path_out, exist_ok=True)

    for imagename in tqdm(os.listdir(brightfield_path_in)):  
    
        stacked_masks = []

        bf_image = cv2.imread(os.path.join(brightfield_path_in, imagename), 0)
        alive_mask = cv2.imread(os.path.join(alive_path_in, imagename), 0) / 255.0
        dead_mask = cv2.imread(os.path.join(dead_path_in, imagename), 0) / 255.0
        pos_mask = cv2.imread(os.path.join(pos_path_in, imagename), 0) / 255.0
        stacked_masks = np.stack((alive_mask, dead_mask, pos_mask), axis=-1)

        prepared_image = (np.array((bf_image).reshape(-1, h, w, c), dtype=np.uint8))
        prepared_masks = np.array(stacked_masks).reshape(-1, h, w, 3)

        pred_ious_lists = []
        pred_detection_lists = []

        for model in evalnets:
            with contextlib.redirect_stdout(io.StringIO()):
                predictions = model.predict([prepared_image, prepared_masks])
                pred_ious = predictions[0]
                pred_detection = predictions[1]

                pred_ious_lists.append(pred_ious)
                pred_detection_lists.append(pred_detection)


        pred_ious_lists_stacked = np.stack(pred_ious_lists, axis=0)
        pred_detection_lists_stacked = np.stack(pred_detection_lists, axis=0)


        mean_ious_per_class = np.mean(pred_ious_lists_stacked, axis=0).squeeze(0)
        mean_detection_per_class = np.mean(pred_detection_lists_stacked, axis=0).squeeze(0)

        valid_iou_values = []
        
        # Iterate over each class
        for class_idx in range(mean_ious_per_class.shape[0]):
            if mean_detection_per_class[class_idx] >= 0.5:
                valid_iou_values.append(mean_ious_per_class[class_idx])
        
        mIoU = 0

        if valid_iou_values:
            mIoU = sum(valid_iou_values) / len(valid_iou_values)

        threshold_step = (max_threshold - min_threshold) / 5

        if mIoU > max_threshold:
            num_augs = 5
        elif mIoU > min_threshold:
            num_augs = 1 + int((mIoU - min_threshold) / threshold_step)
        else:
            num_augs = 1
        
        num_augs = min(num_augs, 5)

        for j in range(num_augs):
        
            masks = [alive_mask, dead_mask, pos_mask]

            aug_image, aug_masks = augment_image_and_masks(bf_image, masks, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation=free_rotation)

            cv2.imwrite(os.path.join(brightfield_path_out, f'{imagename[:-4]}___{j}.png'), aug_image)
            cv2.imwrite(os.path.join(alive_path_out, f'{imagename[:-4]}___{j}.png'), (aug_masks[0] >= 0.5).astype(np.int)*255)
            cv2.imwrite(os.path.join(dead_path_out, f'{imagename[:-4]}___{j}.png'), (aug_masks[1] >= 0.5).astype(np.int)*255)
            cv2.imwrite(os.path.join(pos_path_out, f'{imagename[:-4]}___{j}.png'), (aug_masks[2] >= 0.5).astype(np.int)*255)




def create_augment_images_and_masks_with_evalnet_ensemble_multiclass(evalnets, h, w, c, num_classes, min_threshold, max_threshold, main_input_path, main_output_path, brightness_range_alpha=(0.6, 1.4), brightness_range_beta=(-20, 20), max_blur=3, max_noise=20, free_rotation=False, rgb=True):
    '''
    Augments images and masks based on the performance of an ensemble of EvalNet models for multiclass segmentation tasks.
    The number of augmentations is determined by the mIoU from the ensemble predictions.

    Args:
        evalnets (list): List of EvalNet models for ensemble prediction.
        h (int): Height of the images.
        w (int): Width of the images.
        c (int): Number of channels in the images.
        num_classes (int): Number of classes for segmentation.
        min_threshold (float): Minimum IoU threshold for determining the number of augmentations.
        max_threshold (float): Maximum IoU threshold for determining the number of augmentations.
        main_input_path (str): Directory containing the original images and masks.
        main_output_path (str): Output directory for augmented images and masks.
        brightness_range_alpha (tuple): Range of alpha for brightness adjustment (default (0.6, 1.4)).
        brightness_range_beta (tuple): Range of beta for brightness adjustment (default (-20, 20)).
        max_blur (int): Maximum blur to apply (default 3).
        max_noise (int): Maximum noise to apply (default 20).
        free_rotation (bool): Flag to allow free rotation (default False).
        rgb (bool): Flag to convert images to RGB (default True).

    Returns:
        None: Augmented images and masks are saved in the specified output directory.
    '''
    
    images_path_in = os.path.join(main_input_path, 'images')
    masks_path_in = os.path.join(main_input_path, 'masks')
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)


    for imagename in tqdm(os.listdir(images_path_in)):  

        image = cv2.imread(os.path.join(images_path_in, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        mask_gray = cv2.imread(os.path.join(masks_path_in, imagename), 0)

        prepared_image = (np.array((input_image).reshape(-1, h, w, c), dtype=np.uint8))

        mask_array = np.array(mask_gray).reshape(-1, h, w, 1)
        preparedMasks_one_hot = np.stack([(mask_array == cls).astype(np.int32) for cls in range(num_classes)], axis=-1).squeeze(axis=-2)

        pred_ious_lists = []
        detected_classes_lists = []
        
        for model in evalnets:
            with contextlib.redirect_stdout(io.StringIO()):
                predictions = model.predict([prepared_image, preparedMasks_one_hot])
                pred_ious = predictions[0]
                detected_classes = predictions[1]
        
                pred_ious_lists.append(pred_ious)
                detected_classes_lists.append(detected_classes)
        
        
        pred_ious_lists_stacked = np.stack(pred_ious_lists, axis=0)
        detected_classes_lists_stacked = np.stack(detected_classes_lists, axis=0)
        
        
        mIoU_list = []
        mean_ious_per_mask = np.mean(pred_ious_lists_stacked, axis=0)
        mean_detected_classes = np.mean(detected_classes_lists_stacked, axis=0)
         
        # Iterate over each mask
        for i in range(mean_ious_per_mask.shape[0]):
            valid_iou_values = []
        
            # Iterate over each class
            for class_idx in range(mean_ious_per_mask.shape[1]):
                if class_idx > 0:
                    if mean_detected_classes[i, class_idx] >= 0.5:
                        valid_iou_values.append(mean_ious_per_mask[i, class_idx])
        
            # Compute mean IoU for the current mask
            if valid_iou_values:  # Check if the list is not empty
                mIoU = sum(valid_iou_values) / len(valid_iou_values)
                mIoU_list.append(mIoU)
            else:
                mIoU_list.append(0.0)  # If no class passed the conf check, append 0

        best_miou_index = np.argmax(mIoU_list)
        best_miou = mIoU_list[best_miou_index]

        threshold_step = (max_threshold - min_threshold) / 5

        if best_miou > max_threshold:
            num_augs = 5
        elif best_miou > min_threshold:
            num_augs = 1 + int((best_miou - min_threshold) / threshold_step)
        else:
            num_augs = 1
        
        num_augs = min(num_augs, 5)

        for j in range(num_augs):
            aug_image, aug_mask = augment_image_and_mask(image, mask_gray, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation=free_rotation)

            cv2.imwrite(os.path.join(images_path_out, f'{imagename[:-4]}___{j}.png'), aug_image)
            cv2.imwrite(os.path.join(masks_path_out, f'{imagename[:-4]}___{j}.png'), aug_mask)




def create_augment_images_and_masks_with_gt(main_gt_input_path, min_threshold, max_threshold, main_input_path, main_output_path, brightness_range_alpha=(0.6, 1.4), brightness_range_beta=(-20, 20), max_blur=3, max_noise=20, free_rotation=False, rgb=True):
    '''
    Augments images and masks based on ground truth masks, simulating a perfect EvalNet. 
    The number of augmentations is determined by the mean IoU between the predicted masks and ground truth masks.

    Args:
        main_gt_input_path (str): Directory containing the ground truth masks.
        min_threshold (float): Minimum IoU threshold for determining the number of augmentations.
        max_threshold (float): Maximum IoU threshold for determining the number of augmentations.
        main_input_path (str): Directory containing the original images, masks, and inconsistency masks.
        main_output_path (str): Output directory for augmented images and masks.
        brightness_range_alpha (tuple): Range of alpha for brightness adjustment (default (0.6, 1.4)).
        brightness_range_beta (tuple): Range of beta for brightness adjustment (default (-20, 20)).
        max_blur (int): Maximum blur to apply (default 3).
        max_noise (int): Maximum noise to apply (default 20).
        free_rotation (bool): Flag to allow free rotation (default False).
        rgb (bool): Flag to convert images to RGB (default True).

    Returns:
        None: Augmented images and masks are saved in the specified output directory.
    '''
    
    im_path_in = os.path.join(main_input_path, 'im')
    images_path_in = os.path.join(main_input_path, 'images')
    masks_path_in = os.path.join(main_input_path, 'masks')
    images_path_out = os.path.join(main_output_path, 'images')
    masks_path_out = os.path.join(main_output_path, 'masks')
    
    os.makedirs(images_path_out, exist_ok=True)
    os.makedirs(masks_path_out, exist_ok=True)


    for imagename in tqdm(os.listdir(images_path_in)):  

        image = cv2.imread(os.path.join(images_path_in, imagename))
        if rgb == True:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_image = image

        im_gray = cv2.imread(os.path.join(im_path_in, imagename), 0)
        mask_gray = cv2.imread(os.path.join(masks_path_in, imagename), 0)

        gt_mask_gray = cv2.imread(os.path.join(main_gt_input_path, imagename), 0)

        gt_mask_gray[im_gray>0] = 0

        miou = get_IoU_multi_unique(mask_gray, gt_mask_gray)

        threshold_step = (max_threshold - min_threshold) / 5

        if miou > max_threshold:
            num_augs = 5
        elif miou > min_threshold:
            num_augs = 1 + int((miou - min_threshold) / threshold_step)
        else:
            num_augs = 1
        
        num_augs = min(num_augs, 5)

        for j in range(num_augs):
            aug_image, aug_mask = augment_image_and_mask(image, mask_gray, brightness_range_alpha, brightness_range_beta, max_blur, max_noise, free_rotation=free_rotation)

            cv2.imwrite(os.path.join(images_path_out, f'{imagename[:-4]}___{j}.png'), aug_image)
            cv2.imwrite(os.path.join(masks_path_out, f'{imagename[:-4]}___{j}.png'), aug_mask)





def convert_class_to_color_mask(class_mask, output_path, class_to_color_mapping):
    '''
    Convert a class mask to a color mask based on a given class-to-color mapping and save it.

    Args:
        class_mask (np.array): The input class mask where each pixel's value represents a class.
        output_path (str): Path where the color mask image will be saved.
        class_to_color_mapping (dict): Mapping from class values to colors.

    Returns:
        None: The function saves the color mask image to the specified output path.
    '''
    
    # Create a 3-channel output image initialized with zeros
    color_mask = np.zeros(list(class_mask.shape) + [3], dtype=np.uint8)
    
    # For each class, find the pixels of that class in the class mask and set their color in the output image
    for color, class_value in class_to_color_mapping.items():
        color_mask[class_mask == class_value] = color
    
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    # Save the color mask image
    cv2.imwrite(output_path, color_mask)





def get_im_prediction_depth_map(models, prepared_image, threshold_multiplier=2):
   
    pred_maps = []

    # Collect depth map predictions from each model
    for model in models:
        with contextlib.redirect_stdout(io.StringIO()):  # Suppress any print outputs
            pred_map = model.predict([prepared_image])
            pred_maps.append(pred_map)

    # Stack the depth maps along a new axis for easy computation
    stacked_maps = np.stack(pred_maps, axis=-1)
    
    # Compute standard deviation along the new axis (across all depth maps) for each pixel position
    std_devs = np.std(stacked_maps, axis=-1)
    
    # Compute the threshold for significant error based on the mean standard deviation
    threshold = threshold_multiplier * np.mean(std_devs)
    
    # Create the mask with significant errors marked as 1
    inconsistency_mask = (std_devs > threshold).astype(int)
    
    return inconsistency_mask



def get_pos_contours(img, erode_kernel=3):
    '''
    Extract position contours from an image.

    Args:
        img (np.array): The input image, either grayscale or BGR.
        erode_kernel (int, optional): Size of the erosion kernel. A value greater than 0 applies erosion to the image. Defaults to 3.

    Returns:
        list: A list of tuples representing the (x, y) coordinates of the contour centers.
    '''
    
    assert len(img.shape) in [2, 3], "Invalid image dimensions."

    grayimg = img
    if len(img.shape) == 3 and img.shape[2] > 1:
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if erode_kernel > 0:
        kernel = np.ones((erode_kernel, erode_kernel), 'uint8')
        grayimg = cv2.convertScaleAbs(grayimg)
        grayimg = cv2.erode(grayimg, kernel, iterations=1)

    _, thresh = cv2.threshold(grayimg, 10, 255, 0)
    thresh = thresh.astype('uint8')
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    pos = []
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

        if M["m00"] != 0:  # avoid division by zero
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"]) + 1
            cY = int(M["m01"] / M["m00"]) + 1
            pos.append((cX, cY))

    return pos


def get_min_dist(xy, positions):
    '''
    Calculate the minimum distance between a given point and a list of other points.

    Args:
        xy (tuple): A tuple (x, y) representing the coordinates of the point.
        positions (list of tuples): A list of tuples where each tuple represents the coordinates of a point.

    Returns:
        float: The minimum distance between the given point and the points in the list. Returns 0 if the list is empty or only contains the given point.
    '''
    
    # Convert input to NumPy arrays
    point = np.array(xy)
    points_array = np.array(positions)

    # Compute distances
    distances = np.linalg.norm(points_array - point, axis=1)

    # Exclude zero distance (i.e., distance to itself)
    distances = distances[distances > 0]

    if distances.size == 0:
        print("No other points or positions list is empty.")
        min_dist = 0

    else:
        # Find minimum distance
        min_dist = np.min(distances)

    return min_dist



def mod_pos_size(gray_img, max_pos_circle_size=8, min_pos_circle_size=3):
    '''
    Modify the size of position circles in a grayscale image based on the distance between them.

    Args:
        gray_img (np.array): The input grayscale image.
        max_pos_circle_size (int, optional): Maximum size of the position circles. Defaults to 8.
        min_pos_circle_size (int, optional): Minimum size of the position circles. Defaults to 3.

    Returns:
        np.array: The modified image with adjusted position circle sizes.
    '''
    
    positions = get_pos_contours(gray_img)

    h,w = gray_img.shape
    
    out_img = np.zeros((h,w), np.uint8)
    
    for pos in positions:
        try:
            min_dist = get_min_dist(pos, positions)
            circle_size = int(min_dist // 4)
    
            if circle_size > max_pos_circle_size:
                circle_size = max_pos_circle_size
    
            if circle_size < min_pos_circle_size:
                circle_size = min_pos_circle_size
    
            cv2.circle(out_img, (pos[0], pos[1]), circle_size, (255), -1) 
        except Exception as e:
            print(e)

    out_img = cv2.blur(out_img, (2,2))
    out_img[out_img<254]=0

    return out_img





def get_cell_count(positions, img_alive, img_dead, measuring_range=3):
    '''
    Count the number of alive and dead cells in an image based on positions and alive/dead masks.

    Args:
        positions (list of tuples): A list of tuples representing the (x, y) coordinates of cell positions.
        img_alive (np.array): Binary or grayscale image representing alive cells.
        img_dead (np.array): Binary or grayscale image representing dead cells.
        measuring_range (int, optional): The range around each position to check for alive or dead cells. Defaults to 3.

    Returns:
        tuple: A tuple (alive_count, dead_count, unclear_count) representing the counts of alive, dead, and unclear cells.
    '''

    alive_count = 0
    dead_count = 0
    unclear_count = 0
    img_h = 0
    img_w = 0

    if len(img_alive.shape) == 3:
        img_h, img_w, img_c = img_alive.shape

        if img_c > 1:
            gray_img_alive = cv2.cvtColor(img_alive, cv2.COLOR_BGR2GRAY)
    else:
        img_h, img_w = img_alive.shape
        gray_img_alive = img_alive


    if len(img_dead.shape) == 3:
        img_h, img_w, img_c = img_dead.shape

        if img_c > 1:
            gray_img_dead = cv2.cvtColor(img_dead, cv2.COLOR_BGR2GRAY)
    else:
        img_h, img_w = img_dead.shape
        gray_img_dead = img_dead


    ret, binary_img_alive = cv2.threshold(gray_img_alive, 10, 255, cv2.THRESH_BINARY)
    ret, binary_img_dead = cv2.threshold(gray_img_dead, 10, 255, cv2.THRESH_BINARY)

    for pos in positions:
        x = pos[0]
        y = pos[1]

        if x-measuring_range <= 0:
            x += measuring_range

        if x+measuring_range > img_w:
            x = img_w-measuring_range

        if y-measuring_range < 0:
            y += measuring_range

        if y+measuring_range > img_h:
            y = img_h-measuring_range

        area_alive = binary_img_alive[y-measuring_range:y+measuring_range ,  x-measuring_range:x+measuring_range]
        area_dead = binary_img_dead[y-measuring_range:y+measuring_range ,  x-measuring_range:x+measuring_range]

        if np.sum(area_alive) > np.sum(area_dead):
            alive_count += 1
            #print(f"{x}  {y}                original: {pos[0]}  {pos[1]}")

        if np.sum(area_dead) > np.sum(area_alive):
            dead_count += 1

        if np.sum(area_dead) == np.sum(area_alive):
            unclear_count += 1

    return alive_count, dead_count, unclear_count




def load_images_from_dir(dir_path):
    '''
    Load all images from a directory as grayscale images.

    Args:
        dir_path (str): Path to the directory containing images.

    Returns:
        tuple: A tuple containing two elements. The first element is a NumPy array of images, and the second element is a list of filenames.
    '''
    
    filenames = os.listdir(dir_path)
    images = [cv2.imread(os.path.join(dir_path, fname), cv2.IMREAD_GRAYSCALE) for fname in filenames]
    return np.array(images), filenames


def normalize_and_threshold(masks):
    '''
    Normalize and apply thresholding to a set of masks to create binary masks.

    Args:
        masks (np.array): Array of masks to be normalized and thresholded.

    Returns:
        np.array: The resulting binary masks after normalization and thresholding.
    '''
    
    masks_normalized = masks.astype(np.float32) / 255.0
    binary_masks = np.where(masks_normalized > 0.5, 1, 0)
    return binary_masks
    

def load_images_and_masks_hela(main_dir):
    '''
    Load brightfield images and corresponding masks from the specified directory.

    Parameters:
    main_dir (str): The main directory containing the image and mask subdirectories.

    Returns:
    tuple: A tuple containing four numpy arrays:
        - bf_imgs: Array of brightfield images.
        - binary_masks_alive: Binary masks of alive cells.
        - binary_masks_dead: Binary masks of dead cells.
        - binary_masks_pos: Binary masks of mod_position.
    '''
    
    bf_img_dir = os.path.join(main_dir, 'brightfield')
    alive_mask_dir = os.path.join(main_dir, 'alive')
    dead_mask_dir = os.path.join(main_dir, 'dead')
    pos_mask_dir = os.path.join(main_dir, 'mod_position')

    bf_imgs, bf_filenames = load_images_from_dir(bf_img_dir)
    alive_masks, alive_filenames = load_images_from_dir(alive_mask_dir)
    dead_masks, dead_filenames = load_images_from_dir(dead_mask_dir)
    pos_masks, pos_filenames = load_images_from_dir(pos_mask_dir)

    # Optional: Check if the number of files in each directory matches
    if not (len(bf_filenames) == len(alive_filenames) == len(dead_filenames) == len(pos_filenames)):
        raise ValueError("Mismatch in number of files across directories")

    binary_masks_alive = normalize_and_threshold(alive_masks)
    binary_masks_dead = normalize_and_threshold(dead_masks)
    binary_masks_pos = normalize_and_threshold(pos_masks)

    return bf_imgs, binary_masks_alive, binary_masks_dead, binary_masks_pos


def load_unlabeled_images_hela(unlabeled_images_dir):
    images, imagenames = load_images_from_dir(unlabeled_images_dir)

    return images














######################################################################################################################################################################################################################################################
###  Functions for saving and plotting training history 
######################################################################################################################################################################################################################################################




def save_history_to_csv(history, save_path):
    """
    Saves a training history dictionary to a CSV file.
    
    Args:
        history (dict): The history object returned from a training function.
        save_path (str): The full path to save the CSV file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create a pandas DataFrame from the history dictionary
    df = pd.DataFrame(history)
    
    # Add an 'epoch' column
    df['epoch'] = range(1, len(df) + 1)
    
    # Reorder columns to make 'epoch' the first column
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Training history saved to {save_path}")


def plot_fixmatch_history(history, save_path):
    """Generates and saves a dual-axis plot for FixMatch training dynamics."""
    epochs = range(1, len(history['val_loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss', color=color)
    ax1.plot(epochs, history['val_loss'], color=color, marker='o', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Pseudo-Label Utilization', color=color)
    ax2.plot(epochs, history['pseudo_label_utilization'], color=color, marker='x', linestyle='--', label='Pseudo-Label Utilization')
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    align_y_axes_zero(ax1, ax2)

    fig.suptitle('FixMatch Training Dynamics', fontsize=16)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_fpl_history(history, save_path):
    """Generates and saves a dual-axis plot for FPL training dynamics."""
    epochs = range(1, len(history['val_loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss', color=color)
    ax1.plot(epochs, history['val_loss'], color=color, marker='o', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Fuzzy Positive Ratio', color=color) # Specific label
    ax2.plot(epochs, history['fuzzy_positive_ratio'], color=color, marker='x', linestyle='--', label='Fuzzy Positive Ratio') # Specific label
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.suptitle('FPL Training Dynamics', fontsize=16)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"FPL diagnostic plot saved to {save_path}")



def align_y_axes_zero(ax1, ax2):
    """
    Adjusts the limits of two y-axes (ax1, ax2) to align their zero points.
    """
    y1_min, y1_max = ax1.get_ylim()

    y2_min, y2_max = ax2.get_ylim()

    if y1_min < 0 and y1_max > 0:
        ratio = y1_max / abs(y1_min)
        if y2_min == 0:
             ax2.set_ylim(-y2_max / ratio, y2_max)
    elif y2_min < 0 and y2_max > 0:
        ratio = y2_max / abs(y2_min)

        if y1_min == 0:
            ax1.set_ylim(-y1_max / ratio, y1_max)



def plot_aggregated_history(df_agg, save_path):
    """
    Generates a dual-axis plot with confidence intervals and aligned zero-axes.
    """
    epochs = df_agg['epoch']
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Validation Loss
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Validation Loss', color=color)
    ax1.plot(epochs, df_agg['mean_val_loss'], color=color, marker='o', label='Mean Validation Loss')
    ax1.fill_between(epochs, df_agg['mean_val_loss'] - df_agg['std_val_loss'], df_agg['mean_val_loss'] + df_agg['std_val_loss'], color=color, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot Pseudo-Label Utilization
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean Pseudo-Label Utilization', color=color) 
    ax2.plot(epochs, df_agg['mean_pseudo_label_utilization'], color=color, marker='x', linestyle='--', label='Mean Pseudo-Label Utilization') 
    ax2.fill_between(epochs, df_agg['mean_pseudo_label_utilization'] - df_agg['std_pseudo_label_utilization'], df_agg['mean_pseudo_label_utilization'] + df_agg['std_pseudo_label_utilization'], color=color, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    
    align_y_axes_zero(ax1, ax2)

    # Final Touches
    fig.suptitle('Aggregated FixMatch Training Dynamics (Mean ± 1σ)', fontsize=16)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Aggregated diagnostic plot saved to {save_path}")
    plt.close()










##############################################################################################################################################################################################
###                         augmentation functions
##############################################################################################################################################################################################


def apply_geometric_augmentations(image, mask):
    """
    Applies ONLY geometric transformations (flips and rotations) to an image and its mask.
    Returns the transformed image and mask.
    """
    # Horizontal Flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # Vertical Flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    return image, mask

def apply_weak_photometric_augmentations(image):
    """
    Applies ONLY weak photometric augmentations to an image.
    This is used for generating the pseudo-label.
    """
    # Convert to numpy for cv2 functions
    img_np = image.numpy()

    # Brightness and Contrast
    brightness_alpha  = np.random.uniform(0.9, 1.1)
    brightness_beta  = np.random.uniform(-5, 5)
    img_np = cv2.convertScaleAbs(img_np, alpha=brightness_alpha, beta=brightness_beta)

    # Noise and Blur
    img_np = add_noise_and_blur(img_np, max_blur=0, max_noise=5)
    
    return tf.convert_to_tensor(img_np, dtype=tf.float32)


def apply_strong_photometric_augmentations(image):
    """
    Applies strong photometric and noise/blur augmentations.
    """
    # Convert to numpy for cv2 functions
    img_np = image.numpy()

    # Brightness and Contrast
    brightness_alpha  = np.random.uniform(0.8, 1.2)
    brightness_beta  = np.random.uniform(-12, 12)
    img_np = cv2.convertScaleAbs(img_np, alpha=brightness_alpha, beta=brightness_beta)

    # Noise and Blur
    img_np = add_noise_and_blur(img_np, max_blur=1, max_noise=20)
    
    return tf.convert_to_tensor(img_np, dtype=tf.float32)


















##############################################################################################################################################################################################
###                         General utility functions
##############################################################################################################################################################################################






def _build_optimizer_weights(optimizer, var_list):
    """
    Eagerly create optimizer slot variables for the given var_list,
    without doing a dummy apply_gradients (so it's safe with LossScaleOptimizer).

    Works for both plain optimizers and mixed-precision LossScaleOptimizer wrappers.
    """
    # If wrapped (mixed precision), unwrap to the inner optimizer for slot creation.
    inner = getattr(optimizer, "_optimizer", None)
    target = inner if inner is not None else optimizer

    # TF 2.10+ private API (present in most 2.x releases used with Keras optimizers)
    if hasattr(target, "_create_all_weights"):
        target._create_all_weights(var_list)
    # Older/fallback API
    elif hasattr(target, "build"):
        target.build(var_list)
    # Otherwise: do nothing; slots will be created lazily on first real apply_gradients.

    # Touch iterations so the variable exists
    _ = optimizer.iterations


def _ensure_rank4(x: tf.Tensor) -> tf.Tensor:
    """Force masks to [B,H,W,1] if they arrive as [B,H,W]."""
    if x is None:
        return None
    return x if tf.rank(x) == 4 else tf.expand_dims(x, -1)


def _clone_model(model: tf.keras.Model) -> tf.keras.Model:
    clone = tf.keras.models.clone_model(model)
    _ = clone(tf.zeros_like(model.inputs[0] if isinstance(model.inputs, (list,tuple)) else model.input))
    clone.set_weights(model.get_weights())
    return clone


def _ema_update(teacher_model: tf.keras.Model, student_model: tf.keras.Model, momentum: float):
    """
    Plain-Python EMA update.
    Safe to call inside @tf.function-compiled steps.
    - teacher_model is frozen (trainable=False) elsewhere.
    - momentum is a Python float (e.g., 0.996).
    """
    tvars = teacher_model.trainable_variables
    svars = student_model.trainable_variables
    for tw, sw in zip(tvars, svars):
        # Keep dtypes consistent; use teacher var dtype
        m = tf.convert_to_tensor(momentum, dtype=tw.dtype)
        tw.assign(m * tw + (1.0 - m) * tf.cast(sw, tw.dtype))




##############################################################################################################################################################################################
###                         FixMatch
##############################################################################################################################################################################################


def random_hflip_batch(x: tf.Tensor, y: tf.Tensor | None = None) -> Tuple[tf.Tensor, tf.Tensor | None]:
    """
    Vectorized random horizontal flip for a batch.
    x: [B,H,W,C], y: [B,H,W] or [B,H,W,1] or None
    """
    flip = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) > 0.5
    x_f = tf.where(flip, tf.reverse(x, axis=[2]), x)
    if y is None:
        return x_f, None
    y4 = _ensure_rank4(y)
    y_f = tf.where(flip, tf.reverse(y4, axis=[2]), y4)
    # Squeeze back if we added a channel for masks
    if tf.rank(y) == 3:
        y_f = tf.squeeze(y_f, -1)
    return x_f, y_f

def strong_photometric_batch(x: tf.Tensor, dataset_name: str = 'ISIC_2018') -> tf.Tensor:
    """
    Brightness/contrast, Gaussian noise, Gaussian blur.
    x: float32 in [0,255], shape [B,H,W,C]
    """
    if dataset_name == 'ISIC_2018':
        max_blur, max_noise, alpha_dev, beta_dev = 3, 25, 0.5, 25.0
    elif dataset_name == 'HeLa':
        max_blur, max_noise, alpha_dev, beta_dev = 3, 25, 0.3, 15.0
    elif dataset_name == 'SUIM':
        max_blur, max_noise, alpha_dev, beta_dev = 3, 25, 0.5, 25.0
    elif dataset_name == 'Cityscapes':
        max_blur, max_noise, alpha_dev, beta_dev = 1, 25, 0.3, 15.0
    else:
        max_blur, max_noise, alpha_dev, beta_dev = 1, 20, 0.3, 15.0

    x = tf.cast(x, tf.float32)

    # Contrast (alpha)
    alpha = 1.0 + tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -alpha_dev, alpha_dev, dtype=x.dtype)
    x = tf.clip_by_value(x * alpha, 0., 255.)

    # Brightness (beta)
    beta = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -beta_dev, beta_dev, dtype=x.dtype)
    x = tf.clip_by_value(x + beta, 0., 255.)

    # Gaussian noise (stddev in pixel scale since x ∈ [0,255])
    if max_noise > 0:
        noise = tf.random.normal(tf.shape(x), stddev=float(max_noise), dtype=x.dtype)
        x = tf.clip_by_value(x + noise, 0., 255.)

    # Gaussian blur (per-channel) using TFA
    if max_blur > 0:
        k = int(max_blur) * 2 + 1  # odd kernel size (e.g., 7 when max_blur=3)
        x = tfa.image.gaussian_filter2d(x, filter_shape=(k, k))

    return x

def build_batch_unlabeled(x_unlab_raw: tf.Tensor, dataset_name: str = 'ISIC_2018') -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Create weak/strong pair with synchronized geometry.
    x_unlab_raw: [B,H,W,C] float32
    """
    x_unlab_weak, _ = random_hflip_batch(x_unlab_raw, None)
    x_unlab_strong = strong_photometric_batch(x_unlab_weak, dataset_name)
    return x_unlab_weak, x_unlab_strong

def build_batch_labeled(x_lab_raw: tf.Tensor, y_lab_raw: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Shared geometric augments for labeled data (horizontal flip).
    x_lab_raw: [B,H,W,C], y_lab_raw: [B,H,W] (int) or [B,H,W,1]
    """
    return random_hflip_batch(x_lab_raw, y_lab_raw)


@tf.function(reduce_retracing=True)
def _train_step(
    model,
    optimizer,
    sup_loss_fn,
    unsup_loss_fn,
    x_lab_raw, y_lab_raw, x_unlab_raw,
    confidence_threshold: float,
    unlabeled_loss_weight: float
):
    # Augment
    x_unlab_weak, x_unlab_strong = build_batch_unlabeled(x_unlab_raw, 'ISIC_2018')
    x_lab_aug, y_lab_aug = build_batch_labeled(x_lab_raw, y_lab_raw)

    b_lab = tf.shape(x_lab_aug)[0]
    b_unlab = tf.shape(x_unlab_weak)[0]
    x_combined = tf.concat([x_lab_aug, x_unlab_weak, x_unlab_strong], axis=0)

    with tf.GradientTape() as tape:
        logits_all = tf.cast(model(x_combined, training=True), tf.float32)
        logits_lab = logits_all[:b_lab]
        logits_unlab_weak = logits_all[b_lab:b_lab + b_unlab]
        logits_unlab_strong = logits_all[b_lab + b_unlab:]

        # Supervised loss
        sup_loss = sup_loss_fn(y_lab_aug, logits_lab)

        # Unsupervised FixMatch loss (Eq. 4-style normalization)
        probs_unlab_weak = tf.stop_gradient(tf.nn.softmax(logits_unlab_weak, axis=-1))
        pseudo = tf.argmax(probs_unlab_weak, axis=-1)            # [B,H,W]
        conf   = tf.reduce_max(probs_unlab_weak, axis=-1)        # [B,H,W]
        mask   = tf.cast(conf >= confidence_threshold, tf.float32)   # [B,H,W]

        pixel_unsup = unsup_loss_fn(pseudo, logits_unlab_strong)     # [B,H,W]

        # NEW: average over ALL unlabeled pixels (mask zeros-out the low-confidence ones)
        total_pixels = tf.cast(tf.reduce_prod(tf.shape(mask)), tf.float32)  # B*H*W
        unsup_loss = tf.math.divide_no_nan(tf.reduce_sum(pixel_unsup * mask), total_pixels)

        total_loss = sup_loss + unlabeled_loss_weight * unsup_loss

        # Mixed precision friendly (works whether optimizer is wrapped or not)
        if hasattr(optimizer, "get_scaled_loss"):
            scaled = optimizer.get_scaled_loss(total_loss)
        else:
            scaled = total_loss

    grads = tape.gradient(scaled, model.trainable_variables)
    if hasattr(optimizer, "get_unscaled_gradients"):
        grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Keep this for pseudo-label utilization logging
    num_conf = tf.reduce_sum(mask)

    return total_loss, sup_loss, unsup_loss, num_conf


def train_isic_2018_fixmatch(
    model,
    filepath_h5,
    h, w, c,
    confidence_threshold,
    unlabeled_loss_weight,
    epochs,
    batch_size,
    optimizer
):
    """
    FixMatch loop for ISIC 2018 with Eq.(4)-style u-loss normalization
    and a local validation helper to avoid NameError.
    """
    import numpy as np, tensorflow as tf
    import paths
    from functions import load_labeled_data_single_mask, load_unlabeled_images

    # -------- Load data once --------
    labeled_images_np, labeled_masks_np = load_labeled_data_single_mask(
        paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR, paths.ISIC_2018_TRAIN_LABELED_MASKS_DIR
    )
    unlabeled_images_np = load_unlabeled_images(paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR)
    val_images_np, val_masks_np = load_labeled_data_single_mask(
        paths.ISIC_2018_VAL_IMAGES_DIR, paths.ISIC_2018_VAL_MASKS_DIR
    )

    # Convert to tensors once
    labeled_images   = tf.convert_to_tensor(labeled_images_np,   dtype=tf.float32)  # [N,H,W,C]
    labeled_masks    = tf.convert_to_tensor(labeled_masks_np,    dtype=tf.int32)    # [N,H,W]
    unlabeled_images = tf.convert_to_tensor(unlabeled_images_np, dtype=tf.float32)  # [M,H,W,C]
    val_images       = tf.convert_to_tensor(val_images_np,       dtype=tf.float32)  # [V,H,W,C]
    val_masks        = tf.convert_to_tensor(val_masks_np,        dtype=tf.int32)    # [V,H,W]

    # -------- Losses --------
    sup_loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    unsup_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    # --- Build model & optimizer outside tf.function ---
    _ = model(tf.zeros([1, h, w, c], dtype=tf.float32), training=False)
    _build_optimizer_weights(optimizer, model.trainable_variables)

    # -------- Local helpers (compiled) --------
    @tf.function(reduce_retracing=True)
    def _val_step_fn(x, y):
        logits = tf.cast(model(x, training=False), tf.float32)
        return sup_loss_fn(y, logits)

    # -------- Loop bookkeeping --------
    history = {'loss': [], 'sup_loss': [], 'unsup_loss': [], 'val_loss': [], 'pseudo_label_utilization': []}
    best_val = np.inf
    N_l = tf.shape(labeled_images)[0]
    N_u = tf.shape(unlabeled_images)[0]
    num_labeled_batches   = tf.cast(tf.math.ceil(tf.cast(N_l, tf.float32) / batch_size), tf.int32)
    num_unlabeled_batches = tf.cast(tf.math.ceil(tf.cast(N_u, tf.float32) / batch_size), tf.int32)
    steps_per_epoch = int(max(int(num_labeled_batches.numpy()), int(num_unlabeled_batches.numpy())))

    # -------- Training --------
    for epoch in range(epochs):
        # Shuffle indices on device
        l_idx_all = tf.random.shuffle(tf.range(N_l))
        u_idx_all = tf.random.shuffle(tf.range(N_u))

        # Running sums for epoch averages
        sum_loss, sum_sup, sum_unsup, sum_ratio = 0.0, 0.0, 0.0, 0.0

        for step in range(steps_per_epoch):
            # Batch index windows (wrap using modulo across dataset)
            l_start = (step % int(num_labeled_batches.numpy())) * batch_size
            u_start = (step % int(num_unlabeled_batches.numpy())) * batch_size

            l_idx = l_idx_all[l_start : l_start + batch_size]
            u_idx = u_idx_all[u_start : u_start + batch_size]

            # Gather batches directly as tensors (no Python lists)
            x_lab_raw = tf.gather(labeled_images, l_idx)
            y_lab_raw = tf.gather(labeled_masks,  l_idx)
            x_unlab_raw = tf.gather(unlabeled_images, u_idx)

            # Uses module-level `_train_step` 
            total_loss, sup_loss, unsup_loss, num_conf = _train_step(
                model, optimizer, sup_loss_fn, unsup_loss_fn,
                x_lab_raw, y_lab_raw, x_unlab_raw,
                float(confidence_threshold), float(unlabeled_loss_weight)
            )

            # Accumulate running sums
            sum_loss  += float(total_loss.numpy())
            sum_sup   += float(sup_loss.numpy())
            sum_unsup += float(unsup_loss.numpy())

            # ratio = confident pixels / total pixels
            total_pixels = np.prod(x_unlab_raw.shape[:3])  # B*H*W
            ratio = float(num_conf.numpy()) / float(total_pixels) if total_pixels > 0 else 0.0
            sum_ratio += ratio

            # Occasional log
            if (step + 1) % 20 == 0 or (step + 1) == steps_per_epoch:
                avg_l = sum_loss / (step + 1)
                avg_s = sum_sup  / (step + 1)
                avg_u = sum_unsup/ (step + 1)
                avg_r = sum_ratio / (step + 1)
                print(f"Epoch {epoch+1}/{epochs} - step {step+1}/{steps_per_epoch} "
                      f"loss {avg_l:.4f} | sup {avg_s:.4f} | unsup {avg_u:.4f} | pseudo_util {avg_r:.4f}")

        # -------- Validation (batched tensors) --------
        V = tf.shape(val_images)[0]
        val_steps = int(tf.math.ceil(tf.cast(V, tf.float32) / batch_size).numpy())
        v_losses = []
        for i in range(val_steps):
            s = i * batch_size
            e = tf.minimum(s + batch_size, V)
            x = val_images[s:e]
            y = val_masks[s:e]
            v = _val_step_fn(x, y)
            v_losses.append(float(v.numpy()))
        val_loss = float(np.mean(v_losses)) if v_losses else np.inf
        print(f"Validation Loss: {val_loss:.4f}")

        # Save epoch metrics
        epoch_avg_loss  = sum_loss  / steps_per_epoch
        epoch_avg_sup   = sum_sup   / steps_per_epoch
        epoch_avg_unsup = sum_unsup / steps_per_epoch
        epoch_avg_ratio = sum_ratio / steps_per_epoch
        history['loss'].append(epoch_avg_loss)
        history['sup_loss'].append(epoch_avg_sup)
        history['unsup_loss'].append(epoch_avg_unsup)
        history['val_loss'].append(val_loss)
        history['pseudo_label_utilization'].append(epoch_avg_ratio)

        # -------- Save best --------
        if val_loss < best_val:
            best_val = val_loss
            if filepath_h5.endswith(".h5"):
                model.save(filepath_h5, include_optimizer=False, save_format="h5")
            else:
                model.save(filepath_h5, include_optimizer=False)

    # Load best back
    loaded = tf.keras.models.load_model(filepath_h5, compile=False)
    return loaded, history




def train_hela_fixmatch(
    model,
    filepath_h5,
    modelname,
    h, w, c,
    # FixMatch hyperparams
    confidence_threshold,          # e.g. 0.8  -> lower=0.2, upper=0.8
    unlabeled_loss_weight,         # e.g. 1.0
    # General training hyperparams
    epochs,
    batch_size,
    optimizer,
    approach='fixmatch',
    channel_weights=(1.0, 1.0, 3.0),   # keep pos-channel=3 weighting
):
    """
    HeLa FixMatch with weighted BCE.
    This version averages the unsupervised loss over the entire unlabeled batch
    (pixels * channels), while masking out low-confidence pixels.
    """
    import os, math, numpy as np, tensorflow as tf, tensorflow_addons as tfa
    from tqdm import tqdm
    from functions import (
        load_images_and_masks_hela,
        load_unlabeled_images,
        benchmark_hela,
    )
    import paths

    @tf.function
    def random_hflip_batch(x, y=None):
        b = tf.shape(x)[0]
        flip = tf.less(tf.random.uniform([b, 1, 1, 1]), 0.5)
        x_f = tf.where(flip, tf.reverse(x, axis=[2]), x)  # width axis=2
        if y is not None:
            y_f = tf.where(flip, tf.reverse(y, axis=[2]), y)
            return x_f, y_f
        return x_f, None

    @tf.function
    def weak_gray_batch(x):
        x = tf.cast(x, tf.float32)  # [B,H,W,1], 0..255
        alpha = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.9, 1.1, dtype=x.dtype)
        beta  = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -10.0, 10.0, dtype=x.dtype)
        return tf.clip_by_value(x * alpha + beta, 0., 255.)

    @tf.function
    def strong_gray_batch(x):
        x = tf.cast(x, tf.float32)
        alpha = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.7, 1.3, dtype=x.dtype)
        beta  = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -25.0, 25.0, dtype=x.dtype)
        x = tf.clip_by_value(x * alpha + beta, 0., 255.)
        noise = tf.random.normal(tf.shape(x), stddev=10.0, dtype=x.dtype)
        x = tf.clip_by_value(x + noise, 0., 255.)
        x = tfa.image.gaussian_filter2d(x, filter_shape=(7, 7))
        return x

    def _weighted_bce_full(y_true, y_pred, w):
        """
        Channel-weighted BCE averaged over all pixels/channels.
        y_*: [B,H,W,3], probs in [0,1]
        w  : [1,1,1,3]
        """
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        per_ch = -(y_true * tf.math.log(y_pred) + (1. - y_true) * tf.math.log(1. - y_pred))  # [B,H,W,3]
        per_ch = per_ch * w
        return tf.reduce_mean(per_ch)

    def _weighted_bce_masked(y_true, y_pred, w, mask01):
        """
        Channel-weighted BCE where loss is computed only on confident pixels,
        but **normalized over the whole unlabeled batch** (B*H*W*3), matching FixMatch Eq. 4 style.
        mask01: [B,H,W,1] in {0,1}; broadcast to channels.
        """
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        per_ch = -(y_true * tf.math.log(y_pred) + (1. - y_true) * tf.math.log(1. - y_pred))  # [B,H,W,3]
        per_ch = per_ch * w
        masked = per_ch * mask01  # broadcast to 3 channels

        # NEW: denominator is all unlabeled pixels * channels (constant per batch)
        num_total = tf.cast(tf.size(mask01), tf.float32) * 3.0  # (B*H*W*1) * 3
        u_loss = tf.math.divide_no_nan(tf.reduce_sum(masked), num_total)

        num_conf = tf.reduce_sum(mask01)  # for utilization stats (counts pixels, not channels)
        return u_loss, num_conf

    # ----------------------------- Load data once -----------------------------
    l_imgs, lA, lD, lP = load_images_and_masks_hela(paths.HELA_TRAIN_LABELED_DIR)
    lab_masks_np = np.stack([lA, lD, lP], axis=-1).astype(np.float32)  # [N,H,W,3] in {0,1}
    u_imgs_np    = load_unlabeled_images(paths.HELA_TRAIN_UNLABELED_BRIGHTFIELD_DIR, read_as_grayscale=True)
    v_imgs, vA, vD, vP = load_images_and_masks_hela(paths.HELA_VAL_DIR)
    val_masks_np = np.stack([vA, vD, vP], axis=-1).astype(np.float32)

    lab_imgs  = tf.expand_dims(tf.convert_to_tensor(l_imgs,       tf.float32), axis=-1)  # [N,H,W,1]
    unl_imgs  = tf.expand_dims(tf.convert_to_tensor(u_imgs_np,    tf.float32), axis=-1)  # [M,H,W,1]
    val_imgs  = tf.expand_dims(tf.convert_to_tensor(v_imgs,       tf.float32), axis=-1)  # [V,H,W,1]
    lab_masks = tf.convert_to_tensor(lab_masks_np, tf.float32)                              # [N,H,W,3]
    val_masks = tf.convert_to_tensor(val_masks_np, tf.float32)                              # [V,H,W,3]

    # Channel weights tensor
    w_ch = tf.reshape(tf.constant(channel_weights, dtype=tf.float32), [1,1,1,3])

    # ----------------------------- Build & optimizer slots -----------------------------
    _ = model(tf.zeros([1, h, w, c], tf.float32), training=False)  # build variables
    _build_optimizer_weights(optimizer, model.trainable_variables)

    lower = float(1.0 - confidence_threshold)
    upper = float(confidence_threshold)

    # ----------------------------- Compiled steps -----------------------------
    @tf.function(reduce_retracing=True)
    def _train_step(x_lab_raw, y_lab_raw, x_unlab_raw, lambda_u: float):
        x_lab_aug, y_lab_aug = random_hflip_batch(x_lab_raw, y_lab_raw)
        x_unl_flip, _        = random_hflip_batch(x_unlab_raw, None)

        x_lw = weak_gray_batch(x_lab_aug)
        x_uw = weak_gray_batch(x_unl_flip)
        x_us = strong_gray_batch(x_uw)

        with tf.GradientTape() as tape:
            # Supervised: weighted BCE
            pred_lab = tf.cast(model(x_lw, training=True), tf.float32)     # [B,H,W,3], sigmoid probs
            l_loss   = _weighted_bce_full(y_lab_aug, pred_lab, w_ch)

            # Teacher on weak unlabeled
            probs_w  = tf.cast(model(x_uw, training=False), tf.float32)    # [B,H,W,3]
            hard_lbl = tf.cast(probs_w > 0.5, tf.float32)

            # Confidence mask: pixel is confident if ALL channels confident
            is_conf_ch   = (probs_w < lower) | (probs_w > upper)           # [B,H,W,3] bool
            conf_mask_px = tf.cast(tf.reduce_all(is_conf_ch, axis=-1, keepdims=True), tf.float32)  # [B,H,W,1]

            # Student on strong unlabeled
            probs_s = tf.cast(model(x_us, training=True), tf.float32)

            # Unlabeled loss: **normalized over all pixels*channels**
            u_loss, num_conf = _weighted_bce_masked(hard_lbl, probs_s, w_ch, conf_mask_px)

            total  = l_loss + tf.cast(lambda_u, tf.float32) * u_loss
            scaled = optimizer.get_scaled_loss(total) if hasattr(optimizer, "get_scaled_loss") else total

        grads = tape.gradient(scaled, model.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return total, l_loss, u_loss, num_conf

    @tf.function(reduce_retracing=True)
    def _val_step(x, y):
        preds = tf.cast(model(x, training=False), tf.float32)
        return _weighted_bce_full(y, preds, w_ch)

    # ----------------------------- Loop -----------------------------
    history = {'val_loss': [], 'pseudo_label_utilization': []}
    best_val = float('inf')

    Nl = int(lab_imgs.shape[0])
    Nu = int(unl_imgs.shape[0])
    nbl = math.ceil(Nl / batch_size)
    nbu = math.ceil(Nu / batch_size)
    steps_per_epoch = max(nbl, nbu)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        idxL = np.random.permutation(Nl)
        idxU = np.random.permutation(Nu)

        sum_total = sum_sup = sum_uns = 0.0
        conf_px, total_px = 0.0, 0.0

        with tqdm(total=steps_per_epoch, desc="Training") as pbar:
            for step in range(steps_per_epoch):
                sL = (step % nbl) * batch_size
                sU = (step % nbu) * batch_size
                bL = idxL[sL:sL + batch_size]
                bU = idxU[sU:sU + batch_size]

                xL = tf.gather(lab_imgs,  bL)  # [B,H,W,1]
                yL = tf.gather(lab_masks, bL)  # [B,H,W,3]
                xU = tf.gather(unl_imgs,  bU)  # [B,H,W,1]

                total, l_loss, u_loss, num_conf = _train_step(
                    xL, yL, xU, float(unlabeled_loss_weight)
                )

                sum_total += float(total.numpy())
                sum_sup   += float(l_loss.numpy())
                sum_uns   += float(u_loss.numpy())
                conf_px   += float(num_conf.numpy())
                total_px  += float(np.prod(xU.shape[:3]))  # B*H*W

                if (step + 1) % 20 == 0 or (step + 1) == steps_per_epoch:
                    avg_l = sum_total / (step + 1)
                    avg_s = sum_sup   / (step + 1)
                    avg_u = sum_uns   / (step + 1)
                    avg_r = (conf_px / max(total_px, 1.0)) if total_px > 0 else 0.0
                    pbar.set_postfix(loss=f"{avg_l:.4f}",
                                     sup=f"{avg_s:.4f}",
                                     unsup=f"{avg_u:.4f}",
                                     pseudo_util=f"{avg_r:.4f}")
                pbar.update(1)

        # Validation
        V = int(val_imgs.shape[0])
        vsteps = math.ceil(V / batch_size)
        v_losses = []
        for i in range(vsteps):
            s = i * batch_size
            e = min(s + batch_size, V)
            v_losses.append(float(_val_step(val_imgs[s:e], val_masks[s:e]).numpy()))
        val_loss = float(np.mean(v_losses)) if v_losses else float('inf')

        pseudo_util = conf_px / max(total_px, 1.0)
        history['val_loss'].append(val_loss)
        history['pseudo_label_utilization'].append(pseudo_util)
        print(f"Validation Loss: {val_loss:.4f} | pseudo_util {pseudo_util:.2%}")

        if val_loss < best_val:
            best_val = val_loss
            if filepath_h5.endswith(".h5"):
                model.save(filepath_h5, include_optimizer=False, save_format="h5")
            else:
                model.save(filepath_h5, include_optimizer=False)

    # ----------------------------- Benchmarking (unchanged) -----------------------------
    best_model = tf.keras.models.load_model(filepath_h5, compile=False)

    val_pred_dir       = os.path.join(paths.HELA_BASE_DIR, 'val_predictions', approach, modelname)
    test_pred_dir      = os.path.join(paths.HELA_BASE_DIR, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)

    mIoU_val, mIoU_ad_val, mcce_val = benchmark_hela(best_model, paths.HELA_VAL_DIR,  val_pred_dir,  h, w, c)
    mIoU_test, mIoU_ad_test, mcce_test = benchmark_hela(best_model, paths.HELA_TEST_DIR, test_pred_dir, h, w, c)
    mIoU_unlabeled, mIoU_ad_unlabeled, mcce_unlabeled = benchmark_hela(best_model, paths.HELA_TRAIN_UNLABELED_DIR, unlabeled_pred_dir, h, w, c)

    return (mIoU_val, mIoU_test, mIoU_unlabeled,
            mIoU_ad_val, mIoU_ad_test, mIoU_ad_unlabeled,
            mcce_val, mcce_test, mcce_unlabeled, history)



def train_multiclass_fixmatch(
    model,
    filepath_h5,
    modelname,
    h, w, c, num_classes, class_to_color_mapping,
    # FixMatch specific hyperparams
    confidence_threshold,
    unlabeled_loss_weight,
    # General training hyperparams
    epochs,
    batch_size,
    optimizer,
    # -------- dataset-specific I/O paths --------
    train_labeled_images_dir,
    train_labeled_masks_dir,
    train_unlabeled_images_dir,
    val_images_dir,
    val_masks_dir,
    test_images_dir,
    test_masks_dir,
    train_unlabeled_masks_dir,     # used only for benchmarking/pred dumps (if available)
    dataset_base_dir,              # base folder to save predictions under
    # Augmentation dataset key for brightness/noise/blur presets ("SUIM", "Cityscapes", ...)
    dataset_name_for_aug='SUIM',
    approach='fixmatch',
):
    """
    Fast FixMatch for multi-class segmentation with Eq.4-style u-loss normalization:
    the masked unsupervised loss is averaged over ALL unlabeled pixels.
    """
    import os, numpy as np, tensorflow as tf
    from functions import (
        load_labeled_data_multiclass_mask,
        load_unlabeled_images,
        benchmark_multiclass,
    )

    # -------- Load once --------
    lab_imgs_np, lab_onehot_np = load_labeled_data_multiclass_mask(
        train_labeled_images_dir, train_labeled_masks_dir, num_classes
    )
    unl_imgs_np = load_unlabeled_images(train_unlabeled_images_dir)
    val_imgs_np, val_onehot_np = load_labeled_data_multiclass_mask(
        val_images_dir, val_masks_dir, num_classes
    )

    # Convert to class IDs and tensors
    lab_masks_np = np.argmax(lab_onehot_np, axis=-1).astype(np.int32)        # [N,H,W]
    val_masks_np = np.argmax(val_onehot_np, axis=-1).astype(np.int32)        # [V,H,W]

    lab_imgs = tf.convert_to_tensor(lab_imgs_np, dtype=tf.float32)           # [N,H,W,C]
    lab_masks= tf.convert_to_tensor(lab_masks_np, dtype=tf.int32)            # [N,H,W]
    unl_imgs = tf.convert_to_tensor(unl_imgs_np, dtype=tf.float32)           # [M,H,W,C]
    val_imgs = tf.convert_to_tensor(val_imgs_np, dtype=tf.float32)           # [V,H,W,C]
    val_masks= tf.convert_to_tensor(val_masks_np, dtype=tf.int32)            # [V,H,W]

    # -------- Losses (model outputs probabilities) --------
    sup_loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    unsup_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )

    # --- Build model & optimizer outside tf.function ---
    _ = model(tf.zeros([1, h, w, c], dtype=tf.float32), training=False)
    _build_optimizer_weights(optimizer, model.trainable_variables)

    # -------- Compiled steps --------
    @tf.function(reduce_retracing=True)
    def _train_step(x_lab_raw, y_lab_raw, x_unlab_raw, tau: float, lambda_u: float):
        # Labeled: geom flip
        x_lab_aug, y_lab_aug = random_hflip_batch(x_lab_raw, y_lab_raw)  # [B,H,W,C], [B,H,W]

        # Unlabeled: weak then strong from weak
        x_unlab_weak, _ = random_hflip_batch(x_unlab_raw, None)
        x_unlab_strong  = strong_photometric_batch(x_unlab_weak, dataset_name=dataset_name_for_aug)

        with tf.GradientTape() as tape:
            # Supervised
            probs_lab = tf.cast(model(x_lab_aug, training=True), tf.float32)  # [B,H,W,K]
            l_loss    = sup_loss_fn(y_lab_aug, probs_lab)

            # Pseudo labels from weak
            probs_weak = tf.cast(model(x_unlab_weak, training=False), tf.float32)  # [B,H,W,K]
            pseudo     = tf.argmax(probs_weak, axis=-1)                            # [B,H,W]
            conf       = tf.reduce_max(probs_weak, axis=-1)                        # [B,H,W]
            mask       = tf.cast(conf >= tau, tf.float32)                          # [B,H,W]

            # Consistency on strong
            probs_strong = tf.cast(model(x_unlab_strong, training=True), tf.float32)   # [B,H,W,K]
            pixel_unsup  = unsup_loss_fn(pseudo, probs_strong)                         # [B,H,W]

            # NEW: average over ALL unlabeled pixels (FixMatch Eq.4 behavior)
            total_pixels = tf.cast(tf.reduce_prod(tf.shape(mask)), tf.float32)         # B*H*W
            u_loss = tf.math.divide_no_nan(tf.reduce_sum(pixel_unsup * mask), total_pixels)

            total_loss = l_loss + lambda_u * u_loss
            scaled = optimizer.get_scaled_loss(total_loss) if hasattr(optimizer, "get_scaled_loss") else total_loss

        grads = tape.gradient(scaled, model.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        num_confident = tf.reduce_sum(mask)  # keep for utilization metric
        return total_loss, l_loss, u_loss, num_confident

    @tf.function(reduce_retracing=True)
    def _val_step(x, y):
        probs = tf.cast(model(x, training=False), tf.float32)
        return sup_loss_fn(y, probs)

    # -------- Loop setup --------
    history = {'val_loss': [], 'pseudo_label_utilization': []}
    best_val = float('inf')
    Nl = tf.shape(lab_imgs)[0]
    Nu = tf.shape(unl_imgs)[0]
    nbl = int(tf.math.ceil(tf.cast(Nl, tf.float32) / batch_size).numpy())
    nbu = int(tf.math.ceil(tf.cast(Nu, tf.float32) / batch_size).numpy())
    steps_per_epoch = max(nbl, nbu)
    tau = float(confidence_threshold)

    # -------- Train --------
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        l_idx_all = tf.random.shuffle(tf.range(Nl))
        u_idx_all = tf.random.shuffle(tf.range(Nu))

        sum_total, sum_sup, sum_uns = 0.0, 0.0, 0.0
        conf_px, total_px = 0.0, 0.0           # per-epoch utilization
        sum_ratio = 0.0                        # running-average pseudo_util print

        for step in range(steps_per_epoch):
            l_start = (step % nbl) * batch_size
            u_start = (step % nbu) * batch_size
            l_idx = l_idx_all[l_start:l_start+batch_size]
            u_idx = u_idx_all[u_start:u_start+batch_size]

            x_lab = tf.gather(lab_imgs,  l_idx)
            y_lab = tf.gather(lab_masks, l_idx)
            x_unl = tf.gather(unl_imgs,  u_idx)

            total, l_loss, u_loss, num_conf = _train_step(
                x_lab, y_lab, x_unl, tau, float(unlabeled_loss_weight)
            )

            sum_total += float(total.numpy())
            sum_sup   += float(l_loss.numpy())
            sum_uns   += float(u_loss.numpy())

            # Step-wise pseudo utilization = confident pixels / (B*H*W)
            total_pixels = float(np.prod(x_unl.shape[:3]))  # B*H*W
            step_pseudo_util = (float(num_conf.numpy()) / max(total_pixels, 1.0)) if total_pixels > 0 else 0.0
            sum_ratio += step_pseudo_util

            # For per-epoch utilization
            conf_px  += float(num_conf.numpy())
            total_px += total_pixels

            if (step+1) % 20 == 0 or (step+1) == steps_per_epoch:
                print(f"  step {step+1}/{steps_per_epoch} | "
                      f"loss {sum_total/(step+1):.4f} | sup {sum_sup/(step+1):.4f} | unsup {sum_uns/(step+1):.4f} | "
                      f"pseudo_util {sum_ratio/(step+1):.4f}")

        # ---- Validation ----
        V = tf.shape(val_imgs)[0]
        vsteps = int(tf.math.ceil(tf.cast(V, tf.float32) / batch_size).numpy())
        v_losses = []
        for i in range(vsteps):
            s = i * batch_size
            e = tf.minimum(s + batch_size, V)
            v_losses.append(float(_val_step(val_imgs[s:e], val_masks[s:e]).numpy()))
        val_loss = float(np.mean(v_losses)) if v_losses else float('inf')

        # Per-epoch pseudo label utilization (stored in history)
        conf_ratio = conf_px / max(total_px, 1.0)
        history['val_loss'].append(val_loss)
        history['pseudo_label_utilization'].append(conf_ratio)
        print(f"Validation Loss: {val_loss:.4f} | pseudo_util {conf_ratio:.2%}")

        if val_loss < best_val:
            best_val = val_loss
            if filepath_h5.endswith(".h5"):
                model.save(filepath_h5, include_optimizer=False, save_format="h5")
            else:
                model.save(filepath_h5, include_optimizer=False)

    print("Training finished. Loading best multi-class model…")
    best_model = tf.keras.models.load_model(filepath_h5, compile=False)

    # Use the model directly for benchmarking
    inference_model = best_model

    # ---- Benchmarks (paths are now parameters) ----
    val_pred_dir       = os.path.join(dataset_base_dir, 'val_predictions', approach, modelname)
    test_pred_dir      = os.path.join(dataset_base_dir, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(dataset_base_dir, 'train_unlabeled_predictions', approach, modelname)

    mPA_val,  mIoU_val  = benchmark_multiclass(inference_model, val_images_dir,  val_masks_dir,  val_pred_dir,  h, w, c, class_to_color_mapping, print_results=False)
    mPA_test, mIoU_test = benchmark_multiclass(inference_model, test_images_dir, test_masks_dir, test_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    mPA_unlabeled, mIoU_unlabeled = benchmark_multiclass(inference_model, train_unlabeled_images_dir, train_unlabeled_masks_dir, unlabeled_pred_dir, h, w, c, class_to_color_mapping, print_results=False)

    return mPA_val, mPA_test, mPA_unlabeled, mIoU_val, mIoU_test, mIoU_unlabeled, history







##############################################################################################################################################################################################
###                         FuzzyPositiveLearning (FPL)
##############################################################################################################################################################################################


def save_fpl_history_to_csv(history, save_path):
    """Saves the FPL training history to a CSV file."""
    df = pd.DataFrame(history)
    df['epoch'] = range(1, len(df) + 1)
    # Select only the key FPL metrics
    df = df[['epoch', 'val_loss', 'fuzzy_positive_ratio']]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"FPL history saved to {save_path}")


@tf.function(reduce_retracing=True)
def _compute_wus_from_probs(probs, fpmask, A=50.0, eps=1e-6):
    """
    Computes the Unsupervised Weight Signal (wus) for FPL from teacher probabilities.
    Args:
      probs: [B,H,W,C] softmax probabilities from the weak view.
      fpmask: [B,H,W,C] {0,1} fuzzy positive mask.
      A: A scaling constant for the log function.
    Returns:
      wus: [B,H,W] pixel-wise weights for the unsupervised loss.
    """
    pos_sum  = tf.reduce_sum(probs * fpmask, axis=-1)                  # Σ_{i∈Y} p_i
    K        = tf.maximum(tf.reduce_sum(fpmask, axis=-1), 1.0)         # K per-pixel
    pos_mean = pos_sum / K                                             # (Σ p_i)/K
    neg_max  = tf.reduce_max(probs * (1.0 - fpmask) + (-1e9)*fpmask, axis=-1)

    # Delta in Eq. (11): pos_mean - max_neg
    delta = pos_mean - neg_max

    # Domain-safe lower bound so 1 + A*delta > 0
    lower = (-1.0 + 1e-4) / float(A)
    delta = tf.maximum(delta, lower)

    num = tf.math.log1p(A * delta)                                     # log(1 + A*(pos_mean - neg_max))
    den = tf.math.log1p(A * pos_mean + eps)                            # log(1 + A*pos_mean)

    wus = tf.math.divide_no_nan(num, den)                              # ratio in Eq. (11)
    return tf.clip_by_value(wus, 0.0, 1.0)


def fuzzy_positive_regularization_loss_lse_weighted(logits, fpmask, wus):
    """
    Computes the weighted Fuzzy Positive Regularization loss using LogSumExp for stability.
    Args:
      logits: [B,H,W,C] logits from the strong view.
      fpmask: [B,H,W,C] {0,1} fuzzy positive mask.
      wus: [B,H,W] pixel-wise weights.
    Returns:
      Scalar loss value.
    """
    z = tf.clip_by_value(tf.cast(logits, tf.float32), -30.0, 30.0)
    pos = tf.cast(fpmask, tf.float32); neg = 1.0 - pos
    NEG_BIG = tf.constant(-1e30, tf.float32)
    lse_pos = tf.reduce_logsumexp(tf.where(pos>0.5, -z, NEG_BIG), axis=-1)
    lse_neg = tf.reduce_logsumexp(tf.where(neg>0.5,  z, NEG_BIG), axis=-1)
    loss_pix = tf.nn.softplus(lse_pos + lse_neg)
    return tf.math.divide_no_nan(tf.reduce_sum(loss_pix * wus), tf.reduce_sum(wus))


@tf.function(reduce_retracing=True)
def weak_photometric_batch(x: tf.Tensor, dataset_name: tf.Tensor = None):
    """Mild brightness/contrast. x float32 in [0,255]."""
    x = tf.cast(x, tf.float32)
    alpha = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.95, 1.05, dtype=x.dtype)
    beta  = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -10.0, 10.0, dtype=x.dtype)
    return tf.clip_by_value(x * alpha + beta, 0., 255.)


@tf.function(reduce_retracing=True)
def weak_gray_batch(x: tf.Tensor):
    """Weak photometric augmentations for grayscale images."""
    x = tf.cast(x, tf.float32)
    alpha = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.9, 1.1, dtype=x.dtype)
    beta  = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -10.0, 10.0, dtype=x.dtype)
    return tf.clip_by_value(x * alpha + beta, 0., 255.)


@tf.function(reduce_retracing=True)
def strong_gray_batch(x: tf.Tensor):
    """Strong photometric augmentations for grayscale images."""
    x = tf.cast(x, tf.float32)
    alpha = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.7, 1.3, dtype=x.dtype)
    beta  = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -25.0, 25.0, dtype=x.dtype)
    x = tf.clip_by_value(x * alpha + beta, 0., 255.)
    x = tf.clip_by_value(x + tf.random.normal(tf.shape(x), stddev=10.0, dtype=x.dtype), 0., 255.)
    x = tfa.image.gaussian_filter2d(x, filter_shape=(7, 7))
    return x


@tf.function(reduce_retracing=True)
def fuzzy_positive_assignment(predictions: tf.Tensor, t_threshold: float):
    """
    Assigns fuzzy positive labels based on cumulative probability.
    Picks the smallest `n` classes whose cumulative probability >= T, then sets the
    top `k=max(n-1, 1)` classes as positive.
    Args:
      predictions: [B,H,W,C] softmax probabilities.
      t_threshold: float scalar T in (0,1].
    Returns:
      mask: [B,H,W,C] float32 mask with {0,1} indicating positive classes.
    """
    C = tf.shape(predictions)[-1]
    sorted_probs, sorted_indices = tf.nn.top_k(predictions, k=C)
    cum = tf.cumsum(sorted_probs, axis=-1)
    
    first_ge = tf.argmax(tf.cast(cum >= t_threshold, tf.int32), axis=-1, output_type=tf.int32)
    n = first_ge + 1
    k = tf.maximum(n - 1, 1)

    pos = tf.range(C, dtype=tf.int32)[tf.newaxis, tf.newaxis, tf.newaxis, :]
    mask_sorted = tf.cast(pos < tf.expand_dims(k, -1), tf.float32)

    inv = tf.argsort(sorted_indices, axis=-1)
    mask_unsorted = tf.gather(mask_sorted, inv, batch_dims=3)
    return mask_unsorted


@tf.function(reduce_retracing=True)
def fuzzy_positive_regularization_loss_lse(logits: tf.Tensor,
                                           fuzzy_positive_mask: tf.Tensor):
    """
    Stable Fuzzy Positive Regularization loss for softmax heads using LogSumExp.
    """
    z = tf.clip_by_value(tf.cast(logits, tf.float32), -30.0, 30.0)
    pos = tf.cast(fuzzy_positive_mask, tf.float32)
    neg = 1.0 - pos
    NEG_BIG = tf.constant(-1e30, tf.float32)

    lse_pos = tf.reduce_logsumexp(tf.where(pos > 0.5, -z, NEG_BIG), axis=-1)
    lse_neg = tf.reduce_logsumexp(tf.where(neg > 0.5,  z, NEG_BIG), axis=-1)

    loss_pix = tf.nn.softplus(lse_pos + lse_neg)
    return tf.reduce_mean(loss_pix)


def train_isic_2018_fpl(
    model,
    base_model_to_save,
    filepath_h5,
    modelname,
    h, w, c,
    t_threshold,
    unlabeled_loss_weight,
    epochs,
    batch_size,
    optimizer,
    approach='fpl'
):
    """Faithful FPL training loop for ISIC 2018 (binary softmax)."""
    from tqdm import tqdm
    import paths
    from functions import load_labeled_data_single_mask, load_unlabeled_images, benchmark_ISIC2018

    lab_imgs_np, lab_mask_np = load_labeled_data_single_mask(paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR,
                                                             paths.ISIC_2018_TRAIN_LABELED_MASKS_DIR)
    unl_imgs_np = load_unlabeled_images(paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR)
    val_imgs_np, val_mask_np = load_labeled_data_single_mask(paths.ISIC_2018_VAL_IMAGES_DIR,
                                                             paths.ISIC_2018_VAL_MASKS_DIR)

    lab_imgs  = tf.convert_to_tensor(lab_imgs_np,  tf.float32)
    lab_masks = tf.convert_to_tensor(lab_mask_np,  tf.int32)
    unl_imgs  = tf.convert_to_tensor(unl_imgs_np,  tf.float32)
    val_imgs  = tf.convert_to_tensor(val_imgs_np,  tf.float32)
    val_masks = tf.convert_to_tensor(val_mask_np,  tf.int32)

    sup_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    _ = model(tf.zeros([1, h, w, c], tf.float32), training=False)
    _build_optimizer_weights(optimizer, model.trainable_variables)

    @tf.function(reduce_retracing=True)
    def _train_step(xL, yL, xU, T: float, lambda_u: float):
        xL, yL = random_hflip_batch(xL, yL)
        xU, _  = random_hflip_batch(xU, None)
        xLw = weak_photometric_batch(xL)
        xUw = weak_photometric_batch(xU)
        xUs = strong_photometric_batch(xUw, dataset_name='ISIC_2018')

        with tf.GradientTape() as tape:
            l_logits = tf.cast(model(xLw, training=True), tf.float32)
            l_loss   = sup_loss_fn(yL, l_logits)

            w_logits = tf.cast(model(xUw, training=False), tf.float32)
            w_probs  = tf.nn.softmax(w_logits, axis=-1)
            fpmask   = fuzzy_positive_assignment(w_probs, T)
            wus      = _compute_wus_from_probs(w_probs, fpmask, A=50.0)

            s_logits = tf.cast(model(xUs, training=True), tf.float32)
            u_loss   = fuzzy_positive_regularization_loss_lse_weighted(s_logits, fpmask, wus)

            total = l_loss + float(lambda_u) * u_loss
            scaled = optimizer.get_scaled_loss(total) if hasattr(optimizer, "get_scaled_loss") else total

        grads = tape.gradient(scaled, model.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        fuzzy_ratio = tf.reduce_mean(fpmask)
        return total, l_loss, u_loss, fuzzy_ratio

    @tf.function(reduce_retracing=True)
    def _val_step(x, y):
        logits = tf.cast(model(x, training=False), tf.float32)
        return sup_loss_fn(y, logits)

    history = {'val_loss': [], 'fuzzy_positive_ratio': []}
    best_val = float('inf')
    Nl, Nu = int(lab_imgs.shape[0]), int(unl_imgs.shape[0])
    nbl, nbu = math.ceil(Nl / batch_size), math.ceil(Nu / batch_size)
    steps = max(nbl, nbu)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        idxL, idxU = np.random.permutation(Nl), np.random.permutation(Nu)
        sum_total = sum_sup = sum_uns = 0.0
        ratios = []
        with tqdm(total=steps, desc="Training") as pbar:
            for step in range(steps):
                sL = (step % nbl) * batch_size
                sU = (step % nbu) * batch_size
                bL, bU = idxL[sL:sL+batch_size], idxU[sU:sU+batch_size]

                xL, yL = tf.gather(lab_imgs, bL), tf.gather(lab_masks, bL)
                xU = tf.gather(unl_imgs, bU)

                total, l, u, fr = _train_step(xL, yL, xU, float(t_threshold), float(unlabeled_loss_weight))
                sum_total += float(total); sum_sup += float(l); sum_uns += float(u)
                ratios.append(float(fr))
                if (step+1) % 20 == 0 or (step+1) == steps:
                    pbar.set_postfix(loss=f"{sum_total/(step+1):.4f}", fuzzy_ratio=f"{np.mean(ratios):.4f}")
                pbar.update(1)

        V, vsteps = int(val_imgs.shape[0]), math.ceil(int(val_imgs.shape[0]) / batch_size)
        v_losses = [float(_val_step(val_imgs[i*batch_size:min((i+1)*batch_size, V)], val_masks[i*batch_size:min((i+1)*batch_size, V)])) for i in range(vsteps)]
        vloss = float(np.mean(v_losses)) if v_losses else float('inf')
        
        history['val_loss'].append(vloss)
        history['fuzzy_positive_ratio'].append(float(np.mean(ratios)) if ratios else 0.0)
        print(f"Validation Loss: {vloss:.4f} | Fuzzy Positive Ratio: {history['fuzzy_positive_ratio'][-1]:.2%}")

        if vloss < best_val:
            best_val = vloss
            base_model_to_save.set_weights(model.get_weights())
            base_model_to_save.save(filepath_h5, include_optimizer=False)

    best_model = tf.keras.models.load_model(filepath_h5, compile=False)
    probs = tf.nn.softmax(tf.cast(best_model.output, tf.float32), axis=-1)
    inference_model = tf.keras.Model(inputs=best_model.inputs, outputs=probs[..., 1:2])

    val_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'val_predictions', approach, modelname)
    test_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)

    mIoU_val, dice_val = benchmark_ISIC2018(inference_model, paths.ISIC_2018_VAL_IMAGES_DIR, paths.ISIC_2018_VAL_MASKS_DIR, val_pred_dir, h, w, c)
    mIoU_tst, dice_tst = benchmark_ISIC2018(inference_model, paths.ISIC_2018_TEST_IMAGES_DIR, paths.ISIC_2018_TEST_MASKS_DIR, test_pred_dir, h, w, c)
    mIoU_unl, dice_unl = benchmark_ISIC2018(inference_model, paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR, paths.ISIC_2018_TRAIN_UNLABELED_MASKS_DIR, unlabeled_pred_dir, h, w, c)

    return mIoU_val, mIoU_tst, mIoU_unl, dice_val, dice_tst, dice_unl, history


def train_hela_fpl(
    model,
    filepath_h5,
    modelname,
    h, w, c,
    confidence_threshold,
    unlabeled_loss_weight,
    epochs,
    batch_size,
    optimizer,
    approach='fpl',
    channel_weights=(1.0, 1.0, 3.0),
):
    """HeLa FPL variant using banded fuzzy targets and weighted MSE loss."""
    from tqdm import tqdm
    import paths
    from functions import load_images_and_masks_hela, load_unlabeled_images, benchmark_hela

    l_imgs, lA, lD, lP = load_images_and_masks_hela(paths.HELA_TRAIN_LABELED_DIR)
    lab_masks_np = np.stack([lA, lD, lP], axis=-1).astype(np.float32)
    u_imgs_np = load_unlabeled_images(paths.HELA_TRAIN_UNLABELED_BRIGHTFIELD_DIR, read_as_grayscale=True)
    v_imgs, vA, vD, vP = load_images_and_masks_hela(paths.HELA_VAL_DIR)
    val_masks_np = np.stack([vA, vD, vP], axis=-1).astype(np.float32)

    lab_imgs  = tf.expand_dims(tf.convert_to_tensor(l_imgs, tf.float32), -1)
    unl_imgs  = tf.expand_dims(tf.convert_to_tensor(u_imgs_np, tf.float32), -1)
    val_imgs  = tf.expand_dims(tf.convert_to_tensor(v_imgs, tf.float32), -1)
    lab_masks = tf.convert_to_tensor(lab_masks_np, tf.float32)
    val_masks = tf.convert_to_tensor(val_masks_np, tf.float32)

    w_ch = tf.reshape(tf.constant(channel_weights, tf.float32), [1,1,1,3])
    def weighted_mse(y_true, y_pred):
        return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred) * w_ch, axis=-1))

    lower, upper = float(1.0 - confidence_threshold), float(confidence_threshold)

    _ = model(tf.zeros([1, h, w, c], tf.float32), training=False)
    _build_optimizer_weights(optimizer, model.trainable_variables)

    @tf.function(reduce_retracing=True)
    def _train_step(xL, yL, xU, lambda_u: float):
        xL, yL = random_hflip_batch(xL, yL)
        xU, _  = random_hflip_batch(xU, None)

        xLw = weak_gray_batch(xL)
        xUw = weak_gray_batch(xU)
        xUs = strong_gray_batch(xUw)

        with tf.GradientTape() as tape:
            pL = tf.cast(model(xLw, training=True), tf.float32)
            l_loss = weighted_mse(yL, pL)

            t_probs = tf.cast(model(xUw, training=False), tf.float32)
            conf_mask = tf.cast((t_probs < lower) | (t_probs > upper), tf.float32)
            hard = tf.cast(t_probs > 0.5, tf.float32)
            fuzzy_targets = conf_mask * hard + (1.0 - conf_mask) * t_probs

            pS = tf.cast(model(xUs, training=True), tf.float32)
            se_u = tf.square(tf.stop_gradient(fuzzy_targets) - pS) * w_ch
            denom = tf.reduce_sum(conf_mask * w_ch) + 1e-7
            u_loss = tf.reduce_sum(se_u * conf_mask) / denom

            total = l_loss + float(lambda_u) * u_loss
            scaled = optimizer.get_scaled_loss(total) if hasattr(optimizer, "get_scaled_loss") else total

        grads = tape.gradient(scaled, model.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total, l_loss, u_loss, tf.reduce_mean(conf_mask)

    @tf.function(reduce_retracing=True)
    def _val_step(x, y):
        return weighted_mse(y, tf.cast(model(x, training=False), tf.float32))

    history = {'val_loss': [], 'fuzzy_positive_ratio': []}
    best_val = float('inf')
    Nl, Nu = int(lab_imgs.shape[0]), int(unl_imgs.shape[0])
    nbl, nbu = math.ceil(Nl / batch_size), math.ceil(Nu / batch_size)
    steps = max(nbl, nbu)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        idxL, idxU = np.random.permutation(Nl), np.random.permutation(Nu)
        sum_total = sum_sup = sum_uns = 0.0
        ratios = []
        with tqdm(total=steps, desc="Training") as pbar:
            for step in range(steps):
                sL, sU = (step % nbl) * batch_size, (step % nbu) * batch_size
                bL, bU = idxL[sL:sL+batch_size], idxU[sU:sU+batch_size]
                xL, yL = tf.gather(lab_imgs, bL), tf.gather(lab_masks, bL)
                xU = tf.gather(unl_imgs, bU)

                total, l, u, fr = _train_step(xL, yL, xU, float(unlabeled_loss_weight))
                sum_total += float(total); sum_sup += float(l); sum_uns += float(u)
                ratios.append(float(fr))
                if (step+1) % 20 == 0 or (step+1) == steps:
                    pbar.set_postfix(loss=f"{sum_total/(step+1):.4f}", fuzzy_ratio=f"{np.mean(ratios):.4f}")
                pbar.update(1)

        V, vsteps = int(val_imgs.shape[0]), math.ceil(int(val_imgs.shape[0]) / batch_size)
        v_losses = [float(_val_step(val_imgs[i*batch_size:min((i+1)*batch_size, V)], val_masks[i*batch_size:min((i+1)*batch_size, V)])) for i in range(vsteps)]
        vloss = float(np.mean(v_losses)) if v_losses else float('inf')
        
        history['val_loss'].append(vloss)
        history['fuzzy_positive_ratio'].append(float(np.mean(ratios)) if ratios else 0.0)
        print(f"Validation Loss: {vloss:.4f} | Fuzzy Positive Ratio: {history['fuzzy_positive_ratio'][-1]:.2%}")
        
        if vloss < best_val:
            best_val = vloss
            model.save(filepath_h5, include_optimizer=False)

    best_model = tf.keras.models.load_model(filepath_h5, compile=False)
    val_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'val_predictions', approach, modelname)
    test_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)

    mIoU_val, mIoU_ad_val, mcce_val = benchmark_hela(best_model, paths.HELA_VAL_DIR, val_pred_dir, h, w, c)
    mIoU_tst, mIoU_ad_tst, mcce_tst = benchmark_hela(best_model, paths.HELA_TEST_DIR, test_pred_dir, h, w, c)
    mIoU_unl, mIoU_ad_unl, mcce_unl = benchmark_hela(best_model, paths.HELA_TRAIN_UNLABELED_DIR, unlabeled_pred_dir, h, w, c)

    return (mIoU_val, mIoU_tst, mIoU_unl, mIoU_ad_val, mIoU_ad_tst, mIoU_ad_unl, mcce_val, mcce_tst, mcce_unl, history)


def train_multiclass_fpl(
    model,
    filepath_h5,
    modelname,
    h, w, c, num_classes, class_to_color_mapping,
    t_threshold,
    unlabeled_loss_weight,
    epochs,
    batch_size,
    optimizer,
    train_labeled_images_dir,
    train_labeled_masks_dir,
    train_unlabeled_images_dir,
    val_images_dir,
    val_masks_dir,
    test_images_dir,
    test_masks_dir,
    train_unlabeled_masks_dir,
    dataset_base_dir,
    dataset_name_for_aug='SUIM',
    approach='fpl'
):
    """Faithful FPL training loop for multi-class segmentation."""
    from tqdm import tqdm
    from functions import load_labeled_data_multiclass_mask, load_unlabeled_images, benchmark_multiclass

    lab_imgs_np, lab_1hot_np = load_labeled_data_multiclass_mask(train_labeled_images_dir, train_labeled_masks_dir, num_classes)
    unl_imgs_np = load_unlabeled_images(train_unlabeled_images_dir)
    val_imgs_np, val_1hot_np = load_labeled_data_multiclass_mask(val_images_dir, val_masks_dir, num_classes)

    lab_masks_np = np.argmax(lab_1hot_np, axis=-1).astype(np.int32)
    val_masks_np = np.argmax(val_1hot_np, axis=-1).astype(np.int32)

    lab_imgs, lab_masks = tf.convert_to_tensor(lab_imgs_np, tf.float32), tf.convert_to_tensor(lab_masks_np, tf.int32)
    unl_imgs = tf.convert_to_tensor(unl_imgs_np, tf.float32)
    val_imgs, val_masks = tf.convert_to_tensor(val_imgs_np, tf.float32), tf.convert_to_tensor(val_masks_np, tf.int32)

    sup_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    _ = model(tf.zeros([1, h, w, c], tf.float32), training=False)
    _build_optimizer_weights(optimizer, model.trainable_variables)

    @tf.function(reduce_retracing=True)
    def _train_step(xL, yL, xU, T: float, lambda_u: float):
        xL, yL = random_hflip_batch(xL, yL)
        xU, _  = random_hflip_batch(xU, None)
        xLw = weak_photometric_batch(xL)
        xUw = weak_photometric_batch(xU)
        xUs = strong_photometric_batch(xUw, dataset_name=dataset_name_for_aug)

        with tf.GradientTape() as tape:
            l_logits = tf.cast(model(xLw, training=True), tf.float32)
            l_loss   = sup_loss_fn(yL, l_logits)
        
            w_logits = tf.cast(model(xUw, training=False), tf.float32)
            w_probs  = tf.nn.softmax(w_logits, axis=-1)
            fpmask   = fuzzy_positive_assignment(w_probs, T)
            wus      = _compute_wus_from_probs(w_probs, fpmask, A=50.0)
        
            s_logits = tf.cast(model(xUs, training=True), tf.float32)
            u_loss   = fuzzy_positive_regularization_loss_lse_weighted(s_logits, fpmask, wus)
        
            total = l_loss + float(lambda_u) * u_loss
            scaled = optimizer.get_scaled_loss(total) if hasattr(optimizer, "get_scaled_loss") else total

        grads = tape.gradient(scaled, model.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total, l_loss, u_loss, tf.reduce_mean(fpmask)

    @tf.function(reduce_retracing=True)
    def _val_step(x, y):
        return sup_loss_fn(y, tf.cast(model(x, training=False), tf.float32))

    history = {'val_loss': [], 'fuzzy_positive_ratio': []}
    best_val = float('inf')
    Nl, Nu = int(lab_imgs.shape[0]), int(unl_imgs.shape[0])
    nbl, nbu = math.ceil(Nl / batch_size), math.ceil(Nu / batch_size)
    steps = max(nbl, nbu)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        idxL, idxU = np.random.permutation(Nl), np.random.permutation(Nu)
        sum_total = sum_sup = sum_uns = 0.0
        ratios = []
        with tqdm(total=steps, desc="Training") as pbar:
            for step in range(steps):
                sL, sU = (step % nbl) * batch_size, (step % nbu) * batch_size
                bL, bU = idxL[sL:sL+batch_size], idxU[sU:sU+batch_size]
                xL, yL = tf.gather(lab_imgs, bL), tf.gather(lab_masks, bL)
                xU = tf.gather(unl_imgs, bU)

                total, l, u, fr = _train_step(xL, yL, xU, float(t_threshold), float(unlabeled_loss_weight))
                sum_total += float(total); sum_sup += float(l); sum_uns += float(u)
                ratios.append(float(fr))
                if (step+1) % 20 == 0 or (step+1) == steps:
                    pbar.set_postfix(loss=f"{sum_total/(step+1):.4f}", fuzzy_ratio=f"{np.mean(ratios):.4f}")
                pbar.update(1)

        V, vsteps = int(val_imgs.shape[0]), math.ceil(int(val_imgs.shape[0]) / batch_size)
        v_losses = [float(_val_step(val_imgs[i*batch_size:min((i+1)*batch_size, V)], val_masks[i*batch_size:min((i+1)*batch_size, V)])) for i in range(vsteps)]
        vloss = float(np.mean(v_losses)) if v_losses else float('inf')

        history['val_loss'].append(vloss)
        history['fuzzy_positive_ratio'].append(float(np.mean(ratios)) if ratios else 0.0)
        print(f"Validation Loss: {vloss:.4f} | Fuzzy Positive Ratio: {history['fuzzy_positive_ratio'][-1]:.2%}")
        
        if vloss < best_val:
            best_val = vloss
            model.save(filepath_h5, include_optimizer=False)

    best_model = tf.keras.models.load_model(filepath_h5, compile=False)
    val_pred_dir = os.path.join(dataset_base_dir, 'val_predictions', approach, modelname)
    test_pred_dir = os.path.join(dataset_base_dir, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(dataset_base_dir, 'train_unlabeled_predictions', approach, modelname)

    mPA_val, mIoU_val = benchmark_multiclass(best_model, val_images_dir, val_masks_dir, val_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    mPA_test, mIoU_test = benchmark_multiclass(best_model, test_images_dir, test_masks_dir, test_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    mPA_unl, mIoU_unl = benchmark_multiclass(best_model, train_unlabeled_images_dir, train_unlabeled_masks_dir, unlabeled_pred_dir, h, w, c, class_to_color_mapping, print_results=False)

    return mPA_val, mPA_test, mPA_unl, mIoU_val, mIoU_test, mIoU_unl, history




##############################################################################################################################################################################################
###                         CrossMatch
##############################################################################################################################################################################################



def _softmax(logits):
    return tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1)

def _categorical_dice_per_pixel(p, q, eps=1e-6):
    """Dice distance per pixel between two distributions p,q (probs)."""
    num = 2.0 * tf.reduce_sum(p * q, axis=-1)
    den = tf.reduce_sum(tf.square(p), axis=-1) + tf.reduce_sum(tf.square(q), axis=-1) + eps
    return 1.0 - (num / den)

_ce_dist = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none')

def _distill_loss(p_teacher, p_student, mode='dice'):
    """Return per-pixel loss map [B,H,W] between distributions."""
    pt = tf.clip_by_value(p_teacher, 1e-8, 1.0)
    ps = tf.clip_by_value(p_student, 1e-8, 1.0)
    if mode == 'dice':
        return _categorical_dice_per_pixel(pt, ps)
    elif mode == 'ce':
        return _ce_dist(pt, ps)
    elif mode == 'kl':
        # KL(teacher||student) = sum pt*(log pt - log ps)
        return tf.reduce_sum(pt * (tf.math.log(pt) - tf.math.log(ps)), axis=-1)
    else:
        raise ValueError("kd_mode must be one of {'dice','ce','kl'}")

def _masked_mean(map2d, mask):
    """
    Paper-faithful averaging: sum over masked pixels, then divide by
    the total number of pixels in the unlabeled batch (B*H*W), NOT by
    the number of confident pixels.
    """
    per = tf.cast(map2d, tf.float32) * tf.cast(mask, tf.float32)  # [B,H,W]
    Bu = tf.cast(tf.shape(per)[0], tf.float32)
    HW = tf.cast(tf.reduce_prod(tf.shape(per)[1:]), tf.float32)
    denom = Bu * HW + 1e-8
    return tf.reduce_sum(per) / denom



def _build_feature_and_tail_models(model: tf.keras.Model):
    """
    Split the model at the last conv-like classifier block feeding the final output.
    Returns:
      feature_model(x) -> feature tensor just BEFORE the classifier head.
      tail_model(f)    -> logits/probs after the classifier head.
    Works for most Functional UNet-style models. Falls back to None,None if not possible.
    """
    try:
        K = int(model.output_shape[-1])
    except Exception:
        return None, None

    last_idx = None
    for idx in range(len(model.layers) - 1, -1, -1):
        L = model.layers[idx]
        # Heuristics: last conv-like layer with filters == num_classes and 4D output
        if isinstance(L, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D, tf.keras.layers.Conv2DTranspose)):
            try:
                if getattr(L, "filters", None) == K and len(L.output_shape) == 4:
                    last_idx = idx
                    break
            except Exception:
                continue

    if last_idx is None:
        # fallback: take the layer whose output feeds the final layer
        try:
            last_idx = len(model.layers) - 2
            _ = model.layers[last_idx].output_shape  # sanity check
        except Exception:
            return None, None

    try:
        feat_tensor = model.layers[last_idx].input
        feature_model = tf.keras.Model(inputs=model.inputs, outputs=feat_tensor, name="crossmatch_features")

        # rebuild the tail from last_idx -> end using shared weights
        tail_in = tf.keras.Input(shape=feat_tensor.shape[1:], name="crossmatch_tail_in")
        x = tail_in
        for L in model.layers[last_idx:]:
            x = L(x)
        tail_model = tf.keras.Model(inputs=tail_in, outputs=x, name="crossmatch_classifier_tail")
        return feature_model, tail_model
    except Exception:
        return None, None


def _decoder_forward_with_feature_dropout(
    feature_model: tf.keras.Model,
    tail_model: tf.keras.Model,
    x: tf.Tensor,
    drop_rate: float,
    outputs_are_logits: bool = True,
    gaussian_std: float | None = None,
    deterministic: bool = True,
):
    """
    Forward pass that:
      1) extracts features with the main network (typically encoder + early decoder),
      2) applies a perturbation at the decoder input (dropout or gaussian),
      3) runs the classifier tail deterministically (no Dropout).
    Returns probabilities.
    """
    # deterministic=True ensures teacher path has no train-time stochasticity
    h = tf.cast(feature_model(x, training=not deterministic), tf.float32)

    if gaussian_std is not None and gaussian_std > 0.0:
        h = h + tf.random.normal(tf.shape(h), stddev=float(gaussian_std), dtype=h.dtype)
    elif drop_rate is not None and drop_rate > 0.0:
        h = tf.nn.dropout(h, rate=float(drop_rate))

    logits_or_probs = tail_model(h, training=False)  # tail kept deterministic
    if outputs_are_logits:
        return _softmax(logits_or_probs)
    else:
        return tf.cast(logits_or_probs, tf.float32)


def crossmatch_build_flows_faithful(
    model,
    x_unl_raw,
    dataset_name='ISIC_2018',
    feat_mode='dropout',         # 'dropout' or 'gaussian'
    feat_rate_weak=0.25,
    feat_rate_strong=0.75,
    # NEW: optional submodels to avoid rebuilding every step (recommended)
    feature_model: tf.keras.Model | None = None,
    classifier_tail: tf.keras.Model | None = None,
    outputs_are_logits: bool = True,
    deterministic_teacher: bool = True,
):
    """
    Build flows per paper, but with **feature-level** perturbations.
    Returns dict of probabilities:
      p_wn, p_ww, p_ws, p_sw, p_ss, p_s1n, p_s2n
    """
    # If caller didn't pass submodels, try to build them (works for Functional models).
    if feature_model is None or classifier_tail is None:
        feature_model, classifier_tail = _build_feature_and_tail_models(model)

    # Fallback (rare): if we failed to split the model, use a prob-space proxy (still returns same keys).
    fallback = (feature_model is None or classifier_tail is None)

    # weak & two strong image augs (geom sync via flip; strong = photometric)
    x_w, _  = random_hflip_batch(x_unl_raw, None)
    x_s1    = strong_photometric_batch(x_w, dataset_name=dataset_name)
    x_s2    = strong_photometric_batch(x_w, dataset_name=dataset_name)

    if not fallback:
        if feat_mode == 'dropout':
            p_wn  = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_w,  drop_rate=0.0,              outputs_are_logits=outputs_are_logits, deterministic=deterministic_teacher)
            p_ww  = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_w,  drop_rate=feat_rate_weak,   outputs_are_logits=outputs_are_logits, deterministic=True)
            p_ws  = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_w,  drop_rate=feat_rate_strong, outputs_are_logits=outputs_are_logits, deterministic=True)
            p_sw  = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_s1, drop_rate=feat_rate_weak,   outputs_are_logits=outputs_are_logits, deterministic=True)
            p_ss  = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_s1, drop_rate=feat_rate_strong, outputs_are_logits=outputs_are_logits, deterministic=True)
            p_s1n = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_s1, drop_rate=0.0,              outputs_are_logits=outputs_are_logits, deterministic=True)
            p_s2n = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_s2, drop_rate=0.0,              outputs_are_logits=outputs_are_logits, deterministic=True)
        else:  # gaussian feature noise
            p_wn  = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_w,  drop_rate=0.0, outputs_are_logits=outputs_are_logits, gaussian_std=0.0,               deterministic=deterministic_teacher)
            p_ww  = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_w,  drop_rate=0.0, outputs_are_logits=outputs_are_logits, gaussian_std=feat_rate_weak,    deterministic=True)
            p_ws  = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_w,  drop_rate=0.0, outputs_are_logits=outputs_are_logits, gaussian_std=feat_rate_strong,  deterministic=True)
            p_sw  = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_s1, drop_rate=0.0, outputs_are_logits=outputs_are_logits, gaussian_std=feat_rate_weak,    deterministic=True)
            p_ss  = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_s1, drop_rate=0.0, outputs_are_logits=outputs_are_logits, gaussian_std=feat_rate_strong,  deterministic=True)
            p_s1n = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_s1, drop_rate=0.0, outputs_are_logits=outputs_are_logits, gaussian_std=0.0,               deterministic=True)
            p_s2n = _decoder_forward_with_feature_dropout(feature_model, classifier_tail, x_s2, drop_rate=0.0, outputs_are_logits=outputs_are_logits, gaussian_std=0.0,               deterministic=True)
    else:
        # Fallback proxy: prob-space dropout + renorm (less faithful, but keeps code running)
        Pw  = _softmax(model(x_w,  training=False))
        Ps1 = _softmax(model(x_s1, training=False))
        Ps2 = _softmax(model(x_s2, training=False))
        def _renorm(P, rate):
            if rate <= 0.0:
                Q = tf.clip_by_value(P, 1e-8, 1.0)
            else:
                Q = tf.nn.dropout(P, rate=rate)
                Q = tf.clip_by_value(Q, 1e-8, 1.0)
            s = tf.reduce_sum(Q, axis=-1, keepdims=True)
            return Q / tf.maximum(s, 1e-8)
        p_wn, p_s1n, p_s2n = Pw, Ps1, Ps2
        p_ww = _renorm(Pw,  feat_rate_weak)
        p_ws = _renorm(Pw,  feat_rate_strong)
        p_sw = _renorm(Ps1, feat_rate_weak)
        p_ss = _renorm(Ps1, feat_rate_strong)

    return dict(p_wn=p_wn, p_ww=p_ww, p_ws=p_ws, p_sw=p_sw, p_ss=p_ss, p_s1n=p_s1n, p_s2n=p_s2n)


@tf.function(reduce_retracing=True)
def crossmatch_unlabeled_losses_faithful(flows, tau=0.95, eta=0.30, kd_mode='dice'):
    """
    L_u = L_ip + (1-eta) * L_tkd + eta * L_dkd
    Confidence gating by tau is applied to TKD/DKD terms.
    L_ip is averaged over the whole unlabeled batch (no tau-gating),
    and ALL masked averages divide by B*H*W (not by #confident).
    Returns: L_u, L_tkd, L_dkd, L_ip, tconf
    """
    p_wn  = flows['p_wn'];  p_ww = flows['p_ww'];  p_ws = flows['p_ws'];  p_sw = flows['p_sw'];  p_ss = flows['p_ss']
    p_s1n = flows['p_s1n']; p_s2n = flows['p_s2n']

    # Teacher confidence mask (for TKD) and assistant masks (for DKD)
    max_wn = tf.reduce_max(p_wn, axis=-1)                # [B,H,W]
    m_t    = tf.cast(max_wn >= tau, tf.float32)

    max_ww = tf.reduce_max(p_ww, axis=-1)
    max_ws = tf.reduce_max(p_ws, axis=-1)
    m_d_w  = tf.cast(max_ww >= tau, tf.float32)
    m_d_s  = tf.cast(max_ws >= tau, tf.float32)

    # ---------- Teacher KD: p_wn -> {p_ww, p_ws, p_sw, p_ss} ----------
    Ls = []
    for q in (p_ww, p_ws, p_sw, p_ss):
        per = _distill_loss(p_wn, q, mode=kd_mode)       # [B,H,W]
        Ls.append(_masked_mean(per, m_t))                # divide by B*H*W
    L_tkd = tf.add_n(Ls) / 4.0

    # ---------- Decoder KD: p_ww -> p_sw (m_d_w), p_ws -> p_ss (m_d_s) ----------
    d1 = _distill_loss(p_ww, p_sw, mode=kd_mode)
    d2 = _distill_loss(p_ws, p_ss, mode=kd_mode)
    L_d1 = _masked_mean(d1, m_d_w)                       # divide by B*H*W
    L_d2 = _masked_mean(d2, m_d_s)                       # divide by B*H*W
    L_dkd = 0.5 * (L_d1 + L_d2)

    # ---------- Image perturbation: p_wn -> {p_s1n, p_s2n} ----------
    # Paper doesn't apply tau-gating here. Average over entire batch.
    ones = tf.ones_like(max_wn, dtype=tf.float32)
    ip1 = _categorical_dice_per_pixel(p_wn, p_s1n)       # [B,H,W]
    ip2 = _categorical_dice_per_pixel(p_wn, p_s2n)       # [B,H,W]
    L_ip = 0.5 * (_masked_mean(ip1, ones) + _masked_mean(ip2, ones))

    tconf = tf.reduce_mean(max_wn)
    L_u = L_ip + (1.0 - eta) * L_tkd + eta * L_dkd
    return L_u, L_tkd, L_dkd, L_ip, tconf



def train_isic_2018_crossmatch(
    model,
    filepath_h5,
    modelname,
    h, w, c,
    # --- Refined Hyperparameters for Stability and Performance ---
    epochs=100,                     # SSL often needs longer training
    batch_size=8,
    initial_learning_rate=1e-4,     # A safe starting point
    confidence_threshold=0.95,      # tau: A robust value from FixMatch literature
    eta=0.30,                       # mix for decoder-KD
    kd_mode='dice',                 # 'dice' (paper), or 'ce'/'kl'
    final_unlabeled_loss_weight=1.0,# The final lambda_u after ramp-up
    rampup_epochs=20,               # CRITICAL: Ramp-up period for lambda_u (e.g., 20% of total epochs)
    feat_mode='dropout',
    feat_rate_weak=0.1,             # A more conservative start
    feat_rate_strong=0.3,           # A much more conservative start
    # general training
    optimizer=None,                 # Will be created internally if None
    outputs_are_logits=True,
    dataset_name_for_aug='ISIC_2018',
    approach='crossmatch'
):
    """
    Faithful CrossMatch for ISIC 2018 with stability improvements:
    - Unlabeled loss weight ramp-up.
    - Cosine decay learning rate scheduler.
    - Refined default hyperparameters.
    """
    from functions import load_labeled_data_single_mask, load_unlabeled_images, benchmark_ISIC2018
    import paths

    # ---- data ----
    lab_imgs_np, lab_masks_np = load_labeled_data_single_mask(
        paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR, paths.ISIC_2018_TRAIN_LABELED_MASKS_DIR
    )
    unl_imgs_np = load_unlabeled_images(paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR)
    val_imgs_np, val_masks_np = load_labeled_data_single_mask(
        paths.ISIC_2018_VAL_IMAGES_DIR, paths.ISIC_2018_VAL_MASKS_DIR
    )

    lab_imgs  = tf.convert_to_tensor(lab_imgs_np,  tf.float32)
    lab_masks = tf.convert_to_tensor(lab_masks_np, tf.int32)
    unl_imgs  = tf.convert_to_tensor(unl_imgs_np,  tf.float32)
    val_imgs  = tf.convert_to_tensor(val_imgs_np,  tf.float32)
    val_masks = tf.convert_to_tensor(val_masks_np, tf.int32)

    Nl = tf.shape(lab_imgs)[0]; Nu = tf.shape(unl_imgs)[0]
    nbl = int(tf.math.ceil(tf.cast(Nl, tf.float32) / batch_size).numpy())
    nbu = int(tf.math.ceil(tf.cast(Nu, tf.float32) / batch_size).numpy())
    steps_per_epoch = max(nbl, nbu)
    total_steps = steps_per_epoch * epochs

    # ---- Optimizer with Learning Rate Scheduler ----
    if optimizer is None:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=total_steps
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    # ---- supervised loss (probabilities) ----
    ce_sup = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    def sup_mix_loss(y_true_ids, probs):
        ce = ce_sup(y_true_ids, probs)
        y1h = tf.one_hot(tf.cast(y_true_ids, tf.int32), tf.shape(probs)[-1], dtype=probs.dtype)
        dice = tf.reduce_mean(_categorical_dice_per_pixel(y1h, probs))
        return 0.5 * (ce + dice)

    # ---- build slots ----
    _ = model(tf.zeros([1, h, w, c], dtype=tf.float32), training=False)
    _build_optimizer_weights(optimizer, model.trainable_variables)

    feat_model, tail_model = _build_feature_and_tail_models(model)

    # ---- steps ----
    @tf.function(reduce_retracing=True)
    def _train_step(x_lab_raw, y_lab_raw, x_unl_raw, tau: float, eta_val: float, lambda_u: float):
        x_lab_aug, y_lab_aug = random_hflip_batch(x_lab_raw, y_lab_raw)

        with tf.GradientTape() as tape:
            logits_lab = tf.cast(model(x_lab_aug, training=True), tf.float32)
            probs_lab = _softmax(logits_lab) if outputs_are_logits else tf.cast(logits_lab, tf.float32)
            l_loss = sup_mix_loss(y_lab_aug, probs_lab)

            flows = crossmatch_build_flows_faithful(
                model, x_unl_raw,
                dataset_name=dataset_name_for_aug,
                feat_mode=feat_mode, feat_rate_weak=feat_rate_weak, feat_rate_strong=feat_rate_strong,
                feature_model=feat_model, classifier_tail=tail_model,
                outputs_are_logits=outputs_are_logits, deterministic_teacher=True,
            )
            L_u, L_tkd, L_dkd, L_ip, tconf = crossmatch_unlabeled_losses_faithful(
                flows, tau=tau, eta=eta_val, kd_mode=kd_mode
            )

            total = l_loss + lambda_u * L_u
            scaled = optimizer.get_scaled_loss(total) if hasattr(optimizer, "get_scaled_loss") else total

        grads = tape.gradient(scaled, model.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total, l_loss, L_u, L_tkd, L_dkd, L_ip, tconf

    @tf.function(reduce_retracing=True)
    def _val_step(x, y):
        logits = tf.cast(model(x, training=False), tf.float32)
        probs = _softmax(logits) if outputs_are_logits else tf.cast(logits, tf.float32)
        return sup_mix_loss(y, probs)

    # ---- loop ----
    history = {'val_loss': [], 'tconf': [], 'L_u': [], 'L_tkd': [], 'L_dkd': [], 'L_ip': [], 'lr': [], 'lambda_u': []}
    best_val = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        l_idx_all = tf.random.shuffle(tf.range(Nl))
        u_idx_all = tf.random.shuffle(tf.range(Nu))

        # --- DYNAMICALLY COMPUTE RAMP-UP WEIGHT ---
        if epoch < rampup_epochs:
            current_ul_weight = final_unlabeled_loss_weight * (float(epoch + 1) / float(rampup_epochs))
        else:
            current_ul_weight = final_unlabeled_loss_weight
        
        #current_lr = optimizer._decayed_lr(tf.float32).numpy()
        actual_optimizer = optimizer._optimizer if hasattr(optimizer, '_optimizer') else optimizer
        current_lr = actual_optimizer._decayed_lr(tf.float32).numpy()

        print(f"Current LR: {current_lr:.6f}, Current Lambda_U: {current_ul_weight:.4f}")
        history['lr'].append(current_lr)
        history['lambda_u'].append(current_ul_weight)

        sum_total = sum_sup = sum_u = sum_tkd = sum_dkd = sum_ip = sum_conf = 0.0

        for step in range(steps_per_epoch):
            l_start = (step % nbl) * batch_size
            u_start = (step % nbu) * batch_size
            l_idx = l_idx_all[l_start:l_start+batch_size]
            u_idx = u_idx_all[u_start:u_start+batch_size]
            x_lab = tf.gather(lab_imgs,  l_idx)
            y_lab = tf.gather(lab_masks, l_idx)
            x_unl = tf.gather(unl_imgs,  u_idx)

            total, l_loss, L_u, L_tkd, L_dkd, L_ip, tconf = _train_step(
                x_lab, y_lab, x_unl,
                tf.constant(float(confidence_threshold), tf.float32),
                tf.constant(float(eta), tf.float32),
                tf.constant(float(current_ul_weight), tf.float32)
            )

            sum_total += float(total.numpy()); sum_sup += float(l_loss.numpy()); sum_u += float(L_u.numpy())
            sum_tkd += float(L_tkd.numpy()); sum_dkd += float(L_dkd.numpy()); sum_ip += float(L_ip.numpy())
            sum_conf += float(tconf.numpy())

            if (step + 1) % 20 == 0 or (step + 1) == steps_per_epoch:
                print(f"  step {step+1}/{steps_per_epoch} | "
                      f"loss {sum_total/(step+1):.4f} | sup {sum_sup/(step+1):.4f} | "
                      f"Lu {sum_u/(step+1):.4f} | tkd {sum_tkd/(step+1):.4f} | "
                      f"dkd {sum_dkd/(step+1):.4f} | ip {sum_ip/(step+1):.4f} | "
                      f"tconf {sum_conf/(step+1):.4f}", end='\r')

        # validation
        V = tf.shape(val_imgs)[0]
        vsteps = int(tf.math.ceil(tf.cast(V, tf.float32) / batch_size).numpy())
        v_losses = [_val_step(val_imgs[i*batch_size:tf.minimum((i+1)*batch_size, V)], 
                               val_masks[i*batch_size:tf.minimum((i+1)*batch_size, V)]).numpy() 
                    for i in range(vsteps)]
        val_loss = float(np.mean(v_losses)) if v_losses else float('inf')
        
        history['val_loss'].append(val_loss)
        history['tconf'].append(sum_conf/steps_per_epoch)
        history['L_u'].append(sum_u/steps_per_epoch)
        history['L_tkd'].append(sum_tkd/steps_per_epoch)
        history['L_dkd'].append(sum_dkd/steps_per_epoch)
        history['L_ip'].append(sum_ip/steps_per_epoch)

        print(f"\nValidation Loss: {val_loss:.4f} | tconf {history['tconf'][-1]:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            print(f"  New best validation loss. Saving model to {filepath_h5}")
            model.save(filepath_h5, include_optimizer=False)

    # ---- load best & benchmark ----
    print("\nTraining finished. Loading best model for final benchmarking...")
    best_model = tf.keras.models.load_model(filepath_h5, compile=False)
    inference_model = tf.keras.Model(inputs=best_model.inputs, outputs=tf.nn.softmax(best_model.output)[..., 1:2])

    val_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'val_predictions', approach, modelname)
    test_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)

    mIoU_val, dice_val = benchmark_ISIC2018(inference_model, paths.ISIC_2018_VAL_IMAGES_DIR, paths.ISIC_2018_VAL_MASKS_DIR, val_pred_dir, h, w, c)
    mIoU_tst, dice_tst = benchmark_ISIC2018(inference_model, paths.ISIC_2018_TEST_IMAGES_DIR, paths.ISIC_2018_TEST_MASKS_DIR, test_pred_dir, h, w, c)
    mIoU_unl, dice_unl = benchmark_ISIC2018(inference_model, paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR, paths.ISIC_2018_TRAIN_UNLABELED_MASKS_DIR, unlabeled_pred_dir, h, w, c)

    return mIoU_val, mIoU_tst, mIoU_unl, dice_val, dice_tst, dice_unl, history



def train_hela_crossmatch(
    model,
    filepath_h5,
    modelname,
    h, w, c,
    # --- Refined Hyperparameters for Stability and Performance ---
    epochs=150,                     # HeLa is harder, needs more epochs
    batch_size=8,
    initial_learning_rate=1e-4,
    confidence_threshold=0.98,      # HeLa needs a high tau due to sigmoid outputs
    eta=0.30,
    kd_mode='dice',
    final_unlabeled_loss_weight=1.5,# HeLa can benefit from a stronger unlabeled signal
    rampup_epochs=30,               # A longer ramp-up for a harder task
    feat_mode="dropout",
    feat_rate_strong=0.4,
    feat_rate_weak=0.1,
    # General
    optimizer=None,
    approach='crossmatch',
    channel_weights=(1.0, 1.0, 3.0),
):
    """
    Faithful CrossMatch for HeLa with stability improvements:
    - Unlabeled loss weight ramp-up.
    - Cosine decay learning rate scheduler.
    - Refined default hyperparameters for a multi-label sigmoid task.
    """
    # ---------- Augs ----------
    @tf.function
    def random_hflip_batch(x, y=None):
        b = tf.shape(x)[0]
        flip = tf.less(tf.random.uniform([b, 1, 1, 1]), 0.5)
        x_f = tf.where(flip, tf.reverse(x, axis=[2]), x)
        if y is not None:
            y4 = y if tf.rank(y) == 4 else tf.expand_dims(y, -1)
            y_f = tf.where(flip, tf.reverse(y4, axis=[2]), y4)
            if tf.rank(y) == 3: y_f = tf.squeeze(y_f, -1)
            return x_f, y_f
        return x_f, None

    @tf.function
    def weak_gray_batch(x):
        x = tf.cast(x, tf.float32)
        alpha = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.9, 1.1, dtype=x.dtype)
        beta  = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -10.0, 10.0, dtype=x.dtype)
        return tf.clip_by_value(x * alpha + beta, 0., 255.)

    @tf.function
    def strong_gray_batch(x):
        x = tf.cast(x, tf.float32)
        alpha = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.7, 1.3, dtype=x.dtype)
        beta  = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -25.0, 25.0, dtype=x.dtype)
        x = tf.clip_by_value(x * alpha + beta, 0., 255.)
        x = tf.clip_by_value(x + tf.random.normal(tf.shape(x), stddev=10.0, dtype=x.dtype), 0., 255.)
        x = tfa.image.gaussian_filter2d(x, filter_shape=(7, 7))
        return x

    # ---------- Channel weights ----------
    w_ch = tf.reshape(tf.constant(channel_weights, tf.float32), [1,1,1,3])

    # ---------- Supervised BCE (class-balanced, channel-weighted) ----------
    def bce_class_balanced(y_true, y_pred, ch_w):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        pos_frac = tf.reduce_mean(y_true, axis=[0,1,2], keepdims=True)
        pos_frac = tf.clip_by_value(pos_frac, 1e-4, 1. - 1e-4)
        alpha = (1. - pos_frac) / pos_frac
        alpha = tf.clip_by_value(alpha, 1.0, 50.0)
        loss_pos = - y_true * tf.math.log(y_pred) * alpha
        loss_neg = - (1. - y_true) * tf.math.log(1. - y_pred)
        per_ch = (loss_pos + loss_neg) * ch_w
        return tf.reduce_mean(per_ch)

    # ---------- Distances (per-channel maps) ----------
    def dice_map_sigmoid(pt, ps, eps=1e-6):
        num = 2.0 * pt * ps
        den = pt*pt + ps*ps + eps
        return 1.0 - (num / den)

    def distill_map(pt, ps, mode='dice'):
        if mode == 'dice': return dice_map_sigmoid(pt, ps)
        # Add other modes if needed
        raise ValueError("kd_mode must be 'dice'")

    # Masked, channel-weighted mean -> scalar
    def masked_weighted_mean(per_ch_map, mask_ch):
        num = tf.reduce_sum(per_ch_map * mask_ch * w_ch)
        den = tf.reduce_sum(mask_ch * w_ch) + 1e-8
        return num / den

    # ---------- Confidence masks (per channel) ----------
    def conf_mask_from_probs(probs, lower, upper):
        return tf.cast((probs < lower) | (probs > upper), tf.float32)

    # ---------- Data Loading ----------
    l_imgs, lA, lD, lP = load_images_and_masks_hela(paths.HELA_TRAIN_LABELED_DIR)
    lab_masks_np = np.stack([lA, lD, lP], axis=-1).astype(np.float32)
    u_imgs_np    = load_unlabeled_images(paths.HELA_TRAIN_UNLABELED_BRIGHTFIELD_DIR, read_as_grayscale=True)
    v_imgs, vA, vD, vP = load_images_and_masks_hela(paths.HELA_VAL_DIR)
    val_masks_np = np.stack([vA, vD, vP], axis=-1).astype(np.float32)

    lab_imgs  = tf.expand_dims(tf.convert_to_tensor(l_imgs,    tf.float32), -1)
    unl_imgs  = tf.expand_dims(tf.convert_to_tensor(u_imgs_np, tf.float32), -1)
    val_imgs  = tf.expand_dims(tf.convert_to_tensor(v_imgs,    tf.float32), -1)
    lab_masks = tf.convert_to_tensor(lab_masks_np, tf.float32)
    val_masks = tf.convert_to_tensor(val_masks_np, tf.float32)
    
    Nl = int(lab_imgs.shape[0]); Nu = int(unl_imgs.shape[0])
    nbl = math.ceil(Nl / batch_size); nbu = math.ceil(Nu / batch_size)
    steps_per_epoch = max(nbl, nbu)
    total_steps = steps_per_epoch * epochs

    # ---------- Optimizer with Scheduler ----------
    if optimizer is None:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=total_steps
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # ---------- Build & slots ----------
    _ = model(tf.zeros([1, h, w, c], tf.float32), training=False)
    _build_optimizer_weights(optimizer, model.trainable_variables)

    feat_model, tail_model = _build_feature_and_tail_models(model)
    if feat_model is None or tail_model is None:
        raise RuntimeError("CrossMatch HeLa: failed to split model into feature/tail.")

    lower = float(1.0 - confidence_threshold)
    upper = float(confidence_threshold)

    # ---------- Steps ----------
    @tf.function(reduce_retracing=True)
    def _train_step(x_lab_raw, y_lab_raw, x_unl_raw, tau_upper: float, tau_lower: float, eta_val: float, lambda_u: float):
        x_lab, y_lab = random_hflip_batch(x_lab_raw, y_lab_raw)
        x_w, _       = random_hflip_batch(x_unl_raw, None)

        x_lw = weak_gray_batch(x_lab)
        x_uw = weak_gray_batch(x_w)
        x_s1 = strong_gray_batch(x_uw)
        x_s2 = strong_gray_batch(x_uw)

        with tf.GradientTape() as tape:
            p_lab = tf.cast(model(x_lw, training=True), tf.float32)
            Lx = bce_class_balanced(y_lab, p_lab, w_ch)

            p_wn  = _decoder_forward_with_feature_dropout(feat_model, tail_model, x_uw, drop_rate=0.0, outputs_are_logits=False, deterministic=True)
            
            if feat_mode == 'dropout':
                p_ww  = _decoder_forward_with_feature_dropout(feat_model, tail_model, x_uw, drop_rate=feat_rate_weak,   outputs_are_logits=False, deterministic=True)
                p_ws  = _decoder_forward_with_feature_dropout(feat_model, tail_model, x_uw, drop_rate=feat_rate_strong, outputs_are_logits=False, deterministic=True)
                p_sw  = _decoder_forward_with_feature_dropout(feat_model, tail_model, x_s1, drop_rate=feat_rate_weak,   outputs_are_logits=False, deterministic=True)
                p_ss  = _decoder_forward_with_feature_dropout(feat_model, tail_model, x_s1, drop_rate=feat_rate_strong, outputs_are_logits=False, deterministic=True)
                p_s1n = _decoder_forward_with_feature_dropout(feat_model, tail_model, x_s1, drop_rate=0.0,              outputs_are_logits=False, deterministic=True)
                p_s2n = _decoder_forward_with_feature_dropout(feat_model, tail_model, x_s2, drop_rate=0.0,              outputs_are_logits=False, deterministic=True)
            else: # gaussian
                p_ww  = _decoder_forward_with_feature_dropout(feat_model, tail_model, x_uw, drop_rate=0.0, outputs_are_logits=False, gaussian_std=feat_rate_weak,   deterministic=True)
                p_ws  = _decoder_forward_with_feature_dropout(feat_model, tail_model, x_uw, drop_rate=0.0, outputs_are_logits=False, gaussian_std=feat_rate_strong, deterministic=True)
                p_sw  = _decoder_forward_with_feature_dropout(feat_model, tail_model, x_s1, drop_rate=0.0, outputs_are_logits=False, gaussian_std=feat_rate_weak,   deterministic=True)
                p_ss  = _decoder_forward_with_feature_dropout(feat_model, tail_model, x_s1, drop_rate=0.0, outputs_are_logits=False, gaussian_std=feat_rate_strong, deterministic=True)
                p_s1n = _decoder_forward_with_feature_dropout(feat_model, tail_model, x_s1, drop_rate=0.0, outputs_are_logits=False, gaussian_std=0.0,            deterministic=True)
                p_s2n = _decoder_forward_with_feature_dropout(feat_model, tail_model, x_s2, drop_rate=0.0, outputs_are_logits=False, gaussian_std=0.0,            deterministic=True)

            m_t  = conf_mask_from_probs(p_wn, tau_lower, tau_upper)
            m_dw = conf_mask_from_probs(p_ww, tau_lower, tau_upper)
            m_ds = conf_mask_from_probs(p_ws, tau_lower, tau_upper)

            tkd = 0.0
            for q in (p_ww, p_ws, p_sw, p_ss):
                tkd += masked_weighted_mean(distill_map(p_wn, q, kd_mode), m_t)
            L_tkd = tkd / 4.0

            d1 = masked_weighted_mean(distill_map(p_ww, p_sw, kd_mode), m_dw)
            d2 = masked_weighted_mean(distill_map(p_ws, p_ss, kd_mode), m_ds)
            L_dkd = 0.5 * (d1 + d2)

            # NOTE: Paper uses Dice for L_ip, not the general kd_mode.
            ip1 = masked_weighted_mean(dice_map_sigmoid(p_wn, p_s1n), tf.ones_like(m_t))
            ip2 = masked_weighted_mean(dice_map_sigmoid(p_wn, p_s2n), tf.ones_like(m_t))
            L_ip = 0.5 * (ip1 + ip2)

            L_u = L_ip + (1.0 - eta_val) * L_tkd + eta_val * L_dkd
            total = Lx + lambda_u * L_u
            scaled = optimizer.get_scaled_loss(total) if hasattr(optimizer, "get_scaled_loss") else total

        grads = tape.gradient(scaled, model.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"): grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        tconf = tf.reduce_mean(tf.cast((p_wn < tau_lower) | (p_wn > tau_upper), tf.float32))
        return total, Lx, L_u, L_tkd, L_dkd, L_ip, tconf

    @tf.function(reduce_retracing=True)
    def _val_step(x, y):
        p = tf.cast(model(x, training=False), tf.float32)
        return bce_class_balanced(y, p, w_ch)

    # ---------- Loop ----------
    history = {'val_loss': [], 'tconf': [], 'L_u': [], 'L_tkd': [], 'L_dkd': [], 'L_ip': [], 'lr': [], 'lambda_u': []}
    best_val = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        idxL = np.random.permutation(Nl)
        idxU = np.random.permutation(Nu)

        if epoch < rampup_epochs:
            current_ul_weight = final_unlabeled_loss_weight * (float(epoch + 1) / float(rampup_epochs))
        else:
            current_ul_weight = final_unlabeled_loss_weight
        
        #current_lr = optimizer._decayed_lr(tf.float32).numpy()
        actual_optimizer = optimizer._optimizer if hasattr(optimizer, '_optimizer') else optimizer
        current_lr = actual_optimizer._decayed_lr(tf.float32).numpy()

        print(f"Current LR: {current_lr:.6f}, Current Lambda_U: {current_ul_weight:.4f}")
        history['lr'].append(current_lr)
        history['lambda_u'].append(current_ul_weight)

        sum_total = sum_sup = sum_u = sum_tkd = sum_dkd = sum_ip = sum_conf = 0.0

        with tqdm(total=steps_per_epoch, desc="Training") as pbar:
            for step in range(steps_per_epoch):
                sL = (step % nbl) * batch_size; sU = (step % nbu) * batch_size
                bL = idxL[sL:sL+batch_size];   bU = idxU[sU:sU+batch_size]
                xL = tf.gather(lab_imgs,  bL); yL = tf.gather(lab_masks, bL)
                xU = tf.gather(unl_imgs,  bU)

                total, Lx, L_u, L_tkd, L_dkd, L_ip, tconf = _train_step(
                    xL, yL, xU,
                    tf.constant(upper, tf.float32),
                    tf.constant(lower, tf.float32),
                    tf.constant(float(eta), tf.float32),
                    tf.constant(float(current_ul_weight), tf.float32)
                )

                sum_total += float(total.numpy()); sum_sup += float(Lx.numpy()); sum_u += float(L_u.numpy())
                sum_tkd += float(L_tkd.numpy());  sum_dkd += float(L_dkd.numpy()); sum_ip += float(L_ip.numpy())
                sum_conf += float(tconf.numpy())
                
                pbar.set_postfix(loss=f"{sum_total/(step+1):.4f}", Lu=f"{sum_u/(step+1):.4f}", tconf=f"{sum_conf/(step+1):.4f}")
                pbar.update(1)

        V = int(val_imgs.shape[0]); vsteps = math.ceil(V / batch_size)
        v_losses = [_val_step(val_imgs[i*batch_size:min((i+1)*batch_size, V)], 
                               val_masks[i*batch_size:min((i+1)*batch_size, V)]).numpy() 
                    for i in range(vsteps)]
        vloss = float(np.mean(v_losses)) if v_losses else float('inf')
        
        history['val_loss'].append(vloss); history['tconf'].append(sum_conf/steps_per_epoch)
        history['L_u'].append(sum_u/steps_per_epoch); history['L_tkd'].append(sum_tkd/steps_per_epoch)
        history['L_dkd'].append(sum_dkd/steps_per_epoch); history['L_ip'].append(sum_ip/steps_per_epoch)
        print(f"\nValidation Loss: {vloss:.4f} | tconf {history['tconf'][-1]:.4f}")

        if vloss < best_val:
            best_val = vloss
            print(f"  New best validation loss. Saving model to {filepath_h5}")
            model.save(filepath_h5, include_optimizer=False)

    # ---------- Benchmark ----------
    print("\nTraining finished. Loading best model for final benchmarking...")
    best_model = tf.keras.models.load_model(filepath_h5, compile=False)

    val_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'val_predictions', approach, modelname)
    test_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)

    mIoU_val, mIoU_ad_val, mcce_val = benchmark_hela(best_model, paths.HELA_VAL_DIR, val_pred_dir, h, w, c)
    mIoU_tst, mIoU_ad_tst, mcce_tst = benchmark_hela(best_model, paths.HELA_TEST_DIR, test_pred_dir, h, w, c)
    mIoU_unl, mIoU_ad_unl, mcce_unl = benchmark_hela(best_model, paths.HELA_TRAIN_UNLABELED_DIR, unlabeled_pred_dir, h, w, c)

    return (mIoU_val, mIoU_tst, mIoU_unl, mIoU_ad_val, mIoU_ad_tst, mIoU_ad_unl, mcce_val, mcce_tst, mcce_unl, history)




def train_multiclass_crossmatch(
    model,
    filepath_h5,
    modelname,
    h, w, c, num_classes, class_to_color_mapping,
    # --- Refined Hyperparameters for Stability and Performance ---
    epochs=150,
    batch_size=8,
    initial_learning_rate=1e-4,
    confidence_threshold=0.90,      # A lower tau for harder multi-class tasks
    eta=0.30,
    kd_mode='dice',
    final_unlabeled_loss_weight=1.0,
    rampup_epochs=20,               # Ramp-up is critical here
    feat_mode='dropout',
    feat_rate_weak=0.1,
    feat_rate_strong=0.3,           # Conservative dropout for a harder task
    # Training
    optimizer=None,
    # dataset IO
    train_labeled_images_dir=None,
    train_labeled_masks_dir=None,
    train_unlabeled_images_dir=None,
    val_images_dir=None,
    val_masks_dir=None,
    test_images_dir=None,
    test_masks_dir=None,
    train_unlabeled_masks_dir=None,
    dataset_base_dir=None,
    dataset_name_for_aug='SUIM',
    approach='crossmatch'
):
    """
    Faithful CrossMatch for multi-class tasks with stability improvements:
    - Unlabeled loss weight ramp-up.
    - Cosine decay learning rate scheduler.
    - Refined default hyperparameters suitable for harder datasets like SUIM.
    """
    # ---- data ----
    lab_imgs_np, lab_onehot_np = load_labeled_data_multiclass_mask(train_labeled_images_dir, train_labeled_masks_dir, num_classes)
    unl_imgs_np = load_unlabeled_images(train_unlabeled_images_dir)
    val_imgs_np, val_onehot_np = load_labeled_data_multiclass_mask(val_images_dir, val_masks_dir, num_classes)

    lab_masks_np = np.argmax(lab_onehot_np, axis=-1).astype(np.int32)
    val_masks_np = np.argmax(val_onehot_np, axis=-1).astype(np.int32)

    lab_imgs  = tf.convert_to_tensor(lab_imgs_np,  tf.float32)
    lab_masks = tf.convert_to_tensor(lab_masks_np, tf.int32)
    unl_imgs  = tf.convert_to_tensor(unl_imgs_np,  tf.float32)
    val_imgs  = tf.convert_to_tensor(val_imgs_np,  tf.float32)
    val_masks = tf.convert_to_tensor(val_masks_np, tf.int32)

    Nl = tf.shape(lab_imgs)[0]; Nu = tf.shape(unl_imgs)[0]
    nbl = int(tf.math.ceil(tf.cast(Nl, tf.float32) / batch_size).numpy())
    nbu = int(tf.math.ceil(tf.cast(Nu, tf.float32) / batch_size).numpy())
    steps_per_epoch = max(nbl, nbu)
    total_steps = steps_per_epoch * epochs

    # ---- Optimizer with Scheduler ----
    if optimizer is None:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=total_steps
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # ---- supervised CE+Dice on probs ----
    ce_sup = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    def sup_mix_loss(y_ids, probs):
        ce = ce_sup(y_ids, probs)
        y1h = tf.one_hot(tf.cast(y_ids, tf.int32), tf.shape(probs)[-1], dtype=probs.dtype)
        # Using _categorical_dice_per_pixel for consistency with unlabeled loss
        dice = tf.reduce_mean(_categorical_dice_per_pixel(y1h, probs))
        return 0.5 * (ce + dice)

    # ---- build + slots ----
    _ = model(tf.zeros([1, h, w, c], tf.float32), training=False)
    _build_optimizer_weights(optimizer, model.trainable_variables)
    feat_model, tail_model = _build_feature_and_tail_models(model)

    @tf.function(reduce_retracing=True)
    def _train_step(x_lab_raw, y_lab_raw, x_unl_raw, tau: tf.Tensor, eta_val: tf.Tensor, lambda_u: tf.Tensor):
        x_lab_aug, y_lab_aug = random_hflip_batch(x_lab_raw, y_lab_raw)
        with tf.GradientTape() as tape:
            logits_lab = tf.cast(model(x_lab_aug, training=True), tf.float32)
            p_lab = _softmax(logits_lab)
            l_loss = sup_mix_loss(y_lab_aug, p_lab)

            flows = crossmatch_build_flows_faithful(
                model, x_unl_raw, dataset_name=dataset_name_for_aug,
                feat_mode=feat_mode, feat_rate_weak=feat_rate_weak, feat_rate_strong=feat_rate_strong,
                feature_model=feat_model, classifier_tail=tail_model,
                outputs_are_logits=True, deterministic_teacher=True
            )
            L_u, L_tkd, L_dkd, L_ip, tconf = crossmatch_unlabeled_losses_faithful(
                flows, tau=tau, eta=eta_val, kd_mode=kd_mode
            )

            total = l_loss + lambda_u * L_u
            scaled = optimizer.get_scaled_loss(total) if hasattr(optimizer, "get_scaled_loss") else total
        grads = tape.gradient(scaled, model.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"): grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total, l_loss, L_u, L_tkd, L_dkd, L_ip, tconf

    @tf.function(reduce_retracing=True)
    def _val_step(x, y):
        logits = tf.cast(model(x, training=False), tf.float32)
        p = _softmax(logits)
        return sup_mix_loss(y, p)

    # ---- loop ----
    history = {'val_loss': [], 'tconf': [], 'L_u': [], 'L_tkd': [], 'L_dkd': [], 'L_ip': [], 'lr': [], 'lambda_u': []}
    best_val = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        l_idx_all = tf.random.shuffle(tf.range(Nl))
        u_idx_all = tf.random.shuffle(tf.range(Nu))

        if epoch < rampup_epochs:
            current_ul_weight = final_unlabeled_loss_weight * (float(epoch + 1) / float(rampup_epochs))
        else:
            current_ul_weight = final_unlabeled_loss_weight
            
        #current_lr = optimizer._decayed_lr(tf.float32).numpy()
        actual_optimizer = optimizer._optimizer if hasattr(optimizer, '_optimizer') else optimizer
        current_lr = actual_optimizer._decayed_lr(tf.float32).numpy()

        print(f"Current LR: {current_lr:.6f}, Current Lambda_U: {current_ul_weight:.4f}")
        history['lr'].append(current_lr)
        history['lambda_u'].append(current_ul_weight)

        sum_total = sum_sup = sum_u = sum_tkd = sum_dkd = sum_ip = sum_conf = 0.0

        for step in range(steps_per_epoch):
            l_start = (step % nbl) * batch_size; u_start = (step % nbu) * batch_size
            l_idx = l_idx_all[l_start:l_start+batch_size]; u_idx = u_idx_all[u_start:u_start+batch_size]
            x_lab = tf.gather(lab_imgs,  l_idx); y_lab = tf.gather(lab_masks, l_idx)
            x_unl = tf.gather(unl_imgs,  u_idx)

            total, l_loss, L_u, L_tkd, L_dkd, L_ip, tconf = _train_step(
                x_lab, y_lab, x_unl,
                tf.constant(float(confidence_threshold), tf.float32),
                tf.constant(float(eta), tf.float32),
                tf.constant(float(current_ul_weight), tf.float32)
            )

            sum_total += float(total.numpy()); sum_sup += float(l_loss.numpy()); sum_u += float(L_u.numpy())
            sum_tkd += float(L_tkd.numpy()); sum_dkd += float(L_dkd.numpy()); sum_ip += float(L_ip.numpy())
            sum_conf += float(tconf.numpy())

            if (step+1) % 50 == 0 or (step+1) == steps_per_epoch: # Log less frequently for long epochs
                 print(f"  step {step+1}/{steps_per_epoch} | "
                      f"loss {sum_total/(step+1):.4f} | sup {sum_sup/(step+1):.4f} | "
                      f"Lu {sum_u/(step+1):.4f} | tconf {sum_conf/(step+1):.4f}", end='\r')

        V = tf.shape(val_imgs)[0]
        vsteps = int(tf.math.ceil(tf.cast(V, tf.float32) / batch_size).numpy())
        v_losses = [_val_step(val_imgs[i*batch_size:tf.minimum((i+1)*batch_size, V)], 
                               val_masks[i*batch_size:tf.minimum((i+1)*batch_size, V)]).numpy() 
                    for i in range(vsteps)]
        val_loss = float(np.mean(v_losses)) if v_losses else float('inf')

        history['val_loss'].append(val_loss); history['tconf'].append(sum_conf/steps_per_epoch)
        history['L_u'].append(sum_u/steps_per_epoch); history['L_tkd'].append(sum_tkd/steps_per_epoch)
        history['L_dkd'].append(sum_dkd/steps_per_epoch); history['L_ip'].append(sum_ip/steps_per_epoch)
        print(f"\nValidation Loss: {val_loss:.4f} | tconf {history['tconf'][-1]:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            print(f"  New best validation loss. Saving model to {filepath_h5}")
            model.save(filepath_h5, include_optimizer=False)

    print("\nTraining finished. Loading best model for final benchmarking...")
    best_model = tf.keras.models.load_model(filepath_h5, compile=False)

    val_pred_dir = os.path.join(dataset_base_dir, 'val_predictions', approach, modelname)
    test_pred_dir = os.path.join(dataset_base_dir, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(dataset_base_dir, 'train_unlabeled_predictions', approach, modelname)

    mPA_val,  mIoU_val  = benchmark_multiclass(best_model, val_images_dir, val_masks_dir, val_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    mPA_test, mIoU_test = benchmark_multiclass(best_model, test_images_dir, test_masks_dir, test_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    mPA_unl,  mIoU_unl  = benchmark_multiclass(best_model, train_unlabeled_images_dir, train_unlabeled_masks_dir, unlabeled_pred_dir, h, w, c, class_to_color_mapping, print_results=False)

    return (mPA_val, mPA_test, mPA_unl, mIoU_val, mIoU_test, mIoU_unl, history)


##############################################################################################################################################################################################
###                         iMAS
##############################################################################################################################################################################################


# -------------------------------
# Hardness via class-weighted symmetric IoU
# -------------------------------
@tf.function(reduce_retracing=True)
def _one_hot_argmax(p: tf.Tensor) -> tf.Tensor:
    # p: [B,H,W,K] probabilities -> one-hot maps [B,H,W,K]
    ids = tf.argmax(p, axis=-1, output_type=tf.int32)
    K = tf.shape(p)[-1]
    return tf.one_hot(ids, K, dtype=p.dtype)

@tf.function(reduce_retracing=True)
def _class_weighted_iou(z1_probs: tf.Tensor, z2_probs: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    # Non-commutative by weighting with class frequencies from z1's predictions.
    # Returns scalar per image: [B]
    z1 = _one_hot_argmax(z1_probs)
    z2 = _one_hot_argmax(z2_probs)
    inter = tf.reduce_sum(z1 * z2, axis=[1,2])             # [B,K]
    union = tf.reduce_sum(z1 + z2 - z1*z2, axis=[1,2])     # [B,K]
    iou_per_class = inter / (union + eps)                   # [B,K]
    freq = tf.reduce_sum(z1, axis=[1,2])                   # [B,K]
    w = freq / (tf.reduce_sum(freq, axis=-1, keepdims=True) + eps)
    wiou = tf.reduce_sum(w * iou_per_class, axis=-1)       # [B]
    return wiou

@tf.function(reduce_retracing=True)
def _confidence_ratio(p: tf.Tensor, tau: float) -> tf.Tensor:
    # Per-image ratio of confident pixels using teacher/student max prob threshold.
    # p: [B,H,W,K]
    conf = tf.reduce_max(p, axis=-1)                       # [B,H,W]
    m = tf.cast(conf >= tau, tf.float32)
    num = tf.reduce_sum(m, axis=[1,2])
    den = tf.cast(tf.shape(p)[1] * tf.shape(p)[2], tf.float32)
    return num / tf.maximum(den, 1.0)                      # [B]

@tf.function(reduce_retracing=True)
def compute_hardness_gamma(ps_w: tf.Tensor, pt_w: tf.Tensor, tau: float) -> tf.Tensor:
    # ps_w, pt_w: student/teacher probs on weakly-augmented unlabeled batch
    # γ = 1 - [ (ρ_s/2) * wIoU(ps, pt) + (ρ_t/2) * wIoU(pt, ps) ]  in [0,1]
    wiou_sp = _class_weighted_iou(ps_w, pt_w)              # [B]
    wiou_ts = _class_weighted_iou(pt_w, ps_w)              # [B]
    rho_s = _confidence_ratio(ps_w, tau)                   # [B]
    rho_t = _confidence_ratio(pt_w, tau)                   # [B]
    score = 0.5 * (rho_s * wiou_sp + rho_t * wiou_ts)
    gamma = 1.0 - score
    return tf.clip_by_value(gamma, 0.0, 1.0)               # [B]


# -------------------------------
# Model-adaptive strong augmentations
# -------------------------------
@tf.function(reduce_retracing=True)
def adaptive_intensity_mix(x_weak: tf.Tensor, dataset_name: str, gamma: tf.Tensor) -> tf.Tensor:
    # A_Is(u) = γ * Strong(u) + (1-γ) * Weak(u), per-instance γ (broadcast to H,W,C)
    x_strong = strong_photometric_batch(x_weak, dataset_name=dataset_name)
    g = tf.reshape(gamma, [-1,1,1,1])
    return tf.cast(g * x_strong + (1.0 - g) * x_weak, tf.float32)

@tf.function(reduce_retracing=True)
def _cutmix_mask_rect(B: tf.Tensor, H: tf.Tensor, W: tf.Tensor) -> tf.Tensor:
    # Simple single-rectangle CutMix mask per image. Shape: [B,H,W,1], values {0,1}
    # Sample rectangle size from Beta to vary coverage.
    lam = tf.clip_by_value(tf.random.uniform([B], 0.3, 0.7), 0.0, 1.0)  # keep diverse
    cut_rat = tf.sqrt(1.0 - lam)
    cut_w = tf.cast(cut_rat * tf.cast(W, tf.float32), tf.int32)
    cut_h = tf.cast(cut_rat * tf.cast(H, tf.float32), tf.int32)
    cx = tf.random.uniform([B], 0, W, dtype=tf.int32)
    cy = tf.random.uniform([B], 0, H, dtype=tf.int32)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, W)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, H)
    x2 = tf.clip_by_value(cx + (cut_w - cut_w // 2), 0, W)
    y2 = tf.clip_by_value(cy + (cut_h - cut_h // 2), 0, H)

    def draw_box(i):
        yy = tf.range(H)[:,None]
        xx = tf.range(W)[None,:]
        in_box = tf.cast((yy >= y1[i]) & (yy < y2[i]) & (xx >= x1[i]) & (xx < x2[i]), tf.float32)  # [H,W]
        return tf.expand_dims(in_box, -1)

    mask = tf.map_fn(lambda i: draw_box(i), tf.range(B), fn_output_signature=tf.float32)
    return mask  # [B,H,W,1]

@tf.function(reduce_retracing=True)
def model_adaptive_cutmix(x_weak: tf.Tensor, pt_w: tf.Tensor, gamma: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    # Hard–easy pairing by sorting gamma. Trigger probability = mean(gamma).
    # Returns (x_cm, pt_cm) where images & teacher soft labels have been CutMixed accordingly.
    B = tf.shape(x_weak)[0]
    H = tf.shape(x_weak)[1]
    W = tf.shape(x_weak)[2]

    # Sort indices ascending/descending by hardness
    asc = tf.argsort(gamma, direction='ASCENDING')
    dsc = tf.argsort(gamma, direction='DESCENDING')
    # Pair element-wise
    src_idx = tf.range(B)
    pair_a = tf.gather(asc, src_idx)
    pair_b = tf.gather(dsc, src_idx)

    # Build masks (one rectangle per image)
    m = _cutmix_mask_rect(B, H, W)  # [B,H,W,1]

    x_a = tf.gather(x_weak, pair_a)
    x_b = tf.gather(x_weak, pair_b)
    pt_a = tf.gather(pt_w,   pair_a)
    pt_b = tf.gather(pt_w,   pair_b)

    x_cm  = m * x_b  + (1.0 - m) * x_a
    pt_cm = m * pt_b + (1.0 - m) * pt_a

    # Trigger with probability = mean hardness
    p = tf.reduce_mean(gamma)
    do_apply = tf.less(tf.random.uniform([]), p)  # True with prob p
    x_out  = tf.where(do_apply, x_cm, x_weak)
    pt_out = tf.where(do_apply, pt_cm, pt_w)
    return tf.cast(x_out, tf.float32), tf.cast(pt_out, tf.float32)


# -------------------------------
# Losses
# -------------------------------
_ce_sup_logits = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
_ce_unsup_logits_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')

@tf.function(reduce_retracing=True)
def supervised_loss_logits(y_true_ids: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    return _ce_sup_logits(y_true_ids, logits)


@tf.function(reduce_retracing=True)
def unsup_ce_from_teacher_probs(student_logits, teacher_probs, teacher_conf_mask):       #unsup_ce_from_teacher_probs_LITERALLY_FAITHFUL
    per_pix = _ce_unsup_logits_none(teacher_probs, student_logits)
    per_pix = per_pix * tf.squeeze(teacher_conf_mask, -1)
    
    # Sum over confident pixels
    s = tf.reduce_sum(per_pix, axis=[1,2])  # [B]
    
    # Divide by total pixels H*W
    HW = tf.cast(tf.shape(per_pix)[1] * tf.shape(per_pix)[2], tf.float32)
    return s / HW # [B]

@tf.function(reduce_retracing=True)
def teacher_conf_mask_from_probs(teacher_probs: tf.Tensor, tau: float) -> tf.Tensor:
    maxp = tf.reduce_max(teacher_probs, axis=-1, keepdims=True)      # [B,H,W,1]
    return tf.cast(maxp >= tau, tf.float32)


# -----------------------------------------------------------------------------
# iMAS — ISIC (binary softmax logits)
# -----------------------------------------------------------------------------
def train_isic_imas(
    model: tf.keras.Model,                 # student (logits, 2 channels)
    filepath_h5: str,
    modelname: str,
    h: int, w: int, c: int,
    # iMAS hypers
    tau: float = 0.95,                    # confidence threshold
    lambda_u: float = 3.0,                # unsupervised weight
    ema_momentum: float = 0.996,          # EMA for teacher
    # general
    epochs: int = 50,
    batch_size: int = 8,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
    dataset_name_for_aug: str = 'ISIC_2018',
    approach: str = 'imas'
):
    """Faithful iMAS trainer for ISIC (2-class softmax). Uses CE for both Lx and Lu.
    Unlabeled loss combines two parallel branches (A_Is and A_Cs) with instance-specific weighting (1-γ_i).
    """
    from functions import load_labeled_data_single_mask, load_unlabeled_images, benchmark_ISIC2018
    import paths

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(1e-4)

    # Data
    lab_imgs_np, lab_masks_np = load_labeled_data_single_mask(paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR, paths.ISIC_2018_TRAIN_LABELED_MASKS_DIR)
    unl_imgs_np = load_unlabeled_images(paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR)
    val_imgs_np, val_masks_np = load_labeled_data_single_mask(paths.ISIC_2018_VAL_IMAGES_DIR, paths.ISIC_2018_VAL_MASKS_DIR)

    lab_imgs  = tf.convert_to_tensor(lab_imgs_np,  tf.float32)
    lab_masks = tf.convert_to_tensor(lab_masks_np, tf.int32)
    unl_imgs  = tf.convert_to_tensor(unl_imgs_np,  tf.float32)
    val_imgs  = tf.convert_to_tensor(val_imgs_np,  tf.float32)
    val_masks = tf.convert_to_tensor(val_masks_np, tf.int32)

    # Build + teacher clone
    _ = model(tf.zeros([1, h, w, c], tf.float32), training=False)
    _build_optimizer_weights(optimizer, model.trainable_variables)
    teacher = _clone_model(model)
    teacher.trainable = False

    # Steps
    @tf.function(reduce_retracing=True)
    def _forward_probs(mdl, x, training: bool):
        logits = tf.cast(mdl(x, training=training), tf.float32)
        return tf.nn.softmax(logits, axis=-1), logits

    @tf.function(reduce_retracing=True)
    def _train_step(xL, yL, xU, tau_f: float, lambda_u_f: float, ema_m: float):
        # Weak views (sync flip) and photometric
        xL, yL = random_hflip_batch(xL, yL)
        xU, _  = random_hflip_batch(xU, None)
        xLw = weak_photometric_batch(xL)
        xUw = weak_photometric_batch(xU)
    
        with tf.GradientTape() as tape:
            # Supervised CE on logits
            p_lab, logits_lab = _forward_probs(model, xLw, training=True)
            Lx = supervised_loss_logits(yL, logits_lab)
    
            # Teacher/Student on weak unlabeled
            pt_w, _ = _forward_probs(teacher, xUw, training=False)
            pt_w = tf.stop_gradient(pt_w)
    
            ps_w, _ = _forward_probs(model, xUw, training=True)
            gamma = compute_hardness_gamma(ps_w, pt_w, tau_f)
            gamma = tf.stop_gradient(gamma)  # no grad through hardness
    
            # Branch 1: adaptive intensity strong -> CE to teacher pt_w with confidence mask
            xAIs = adaptive_intensity_mix(xUw, dataset_name_for_aug, gamma)
            pt_mask = tf.stop_gradient(teacher_conf_mask_from_probs(pt_w, tau_f))  # [B,H,W,1]
            _, logits_Is = _forward_probs(model, xAIs, training=True)
            Lu_Is = unsup_ce_from_teacher_probs(logits_Is, pt_w, pt_mask)          # [B]
    
            # Branch 2: model-adaptive CutMix on weak -> CE to mixed teacher pt_cm
            xACs, pt_cm = model_adaptive_cutmix(xUw, pt_w, gamma)
            pt_cm = tf.stop_gradient(pt_cm)
            pt_cm_mask = tf.stop_gradient(teacher_conf_mask_from_probs(pt_cm, tau_f))
            _, logits_Cs = _forward_probs(model, xACs, training=True)
            Lu_Cs = unsup_ce_from_teacher_probs(logits_Cs, pt_cm, pt_cm_mask)      # [B]
    
            # Instance-specific weighting (1 - γ_i) and combine branches
            weights = 1.0 - gamma                                                  # [B]
            Lu_per = 0.5 * (Lu_Is + Lu_Cs) * weights
            Lu = tf.reduce_mean(Lu_per)
    
            total = Lx + lambda_u_f * Lu
            scaled = optimizer.get_scaled_loss(total) if hasattr(optimizer, 'get_scaled_loss') else total
    
        grads = tape.gradient(scaled, model.trainable_variables)
        if hasattr(optimizer, 'get_unscaled_gradients'):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    
        # EMA update on variables (works in/eager graph)
        #_ema_update(teacher, model, tf.constant(ema_m, tf.float32))
        #_ema_update(teacher.trainable_variables, model.trainable_variables, tf.constant(ema_m, tf.float32))
        _ema_update(teacher, model, float(ema_m))
    
        # Logging
        return total, Lx, Lu, tf.reduce_mean(gamma)



    @tf.function(reduce_retracing=True)
    def _val_step(x, y):
        _, logits = _forward_probs(model, x, training=False)
        return supervised_loss_logits(y, logits)

    # Loop
    history = {'val_loss': [], 'hardness': []}
    best_val = float('inf')

    Nl = int(lab_imgs.shape[0]); Nu = int(unl_imgs.shape[0])
    nbl = math.ceil(Nl / batch_size); nbu = math.ceil(Nu / batch_size)
    steps = max(nbl, nbu)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        idxL = np.random.permutation(Nl)
        idxU = np.random.permutation(Nu)
        sum_tot = sum_sup = sum_uns = sum_hard = 0.0

        for step in range(steps):
            sL = (step % nbl) * batch_size; sU = (step % nbu) * batch_size
            bL = idxL[sL:sL+batch_size];     bU = idxU[sU:sU+batch_size]
            xL = tf.gather(lab_imgs,  bL)
            yL = tf.gather(lab_masks, bL)
            xU = tf.gather(unl_imgs,  bU)

            total, Lx, Lu, hard = _train_step(xL, yL, xU, float(tau), float(lambda_u), float(ema_momentum))
            sum_tot  += float(total.numpy())
            sum_sup  += float(Lx.numpy())
            sum_uns  += float(Lu.numpy())
            sum_hard += float(hard.numpy())

            if (step+1) % 20 == 0 or (step+1) == steps:
                print(f"  step {step+1}/{steps} | loss {sum_tot/(step+1):.4f} | sup {sum_sup/(step+1):.4f} | Lu {sum_uns/(step+1):.4f} | avg_hard {sum_hard/(step+1):.4f}")

        # Validation
        V = int(val_imgs.shape[0]); vsteps = math.ceil(V / batch_size)
        v_losses = []
        for i in range(vsteps):
            s = i * batch_size; e = min(s + batch_size, V)
            v_losses.append(float(_val_step(val_imgs[s:e], val_masks[s:e]).numpy()))
        vloss = float(np.mean(v_losses)) if v_losses else float('inf')
        history['val_loss'].append(vloss); history['hardness'].append(sum_hard/steps)
        print(f"Validation Loss: {vloss:.4f} | avg_hard {history['hardness'][-1]:.4f}")

        if vloss < best_val:
            best_val = vloss
            # Save student; for inference we expose lesion prob via softmax later
            if filepath_h5.endswith('.h5'):
                model.save(filepath_h5, include_optimizer=False, save_format='h5')
            else:
                model.save(filepath_h5, include_optimizer=False)

    # Inference head: softmax -> lesion prob [:,:, :,1]
    best_model = tf.keras.models.load_model(filepath_h5, compile=False)
    inputs = best_model.inputs
    logits = best_model.outputs[0] if isinstance(best_model.outputs, (list,tuple)) else best_model.output
    sm = tf.keras.layers.Softmax(axis=-1, name='infer_softmax')(tf.cast(logits, tf.float32))
    lesion = tf.keras.layers.Lambda(lambda z: z[...,1:2], name='lesion_prob')(sm)
    inference_model = tf.keras.Model(inputs=inputs, outputs=lesion)

    val_pred_dir       = os.path.join(paths.ISIC_2018_BASE_DIR, 'val_predictions', approach, modelname)
    test_pred_dir      = os.path.join(paths.ISIC_2018_BASE_DIR, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)

    mIoU_val, dice_val = benchmark_ISIC2018(inference_model, paths.ISIC_2018_VAL_IMAGES_DIR,  paths.ISIC_2018_VAL_MASKS_DIR,  val_pred_dir,  h, w, c)
    mIoU_tst, dice_tst = benchmark_ISIC2018(inference_model, paths.ISIC_2018_TEST_IMAGES_DIR, paths.ISIC_2018_TEST_MASKS_DIR, test_pred_dir, h, w, c)
    mIoU_unl, dice_unl = benchmark_ISIC2018(inference_model, paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR, paths.ISIC_2018_TRAIN_UNLABELED_MASKS_DIR, unlabeled_pred_dir, h, w, c)

    return mIoU_val, mIoU_tst, mIoU_unl, dice_val, dice_tst, dice_unl, history



def train_hela_imas(
    model,
    filepath_h5,
    modelname,
    h, w, c,
    # iMAS-style hypers
    confidence_threshold=0.8,     # -> band: lower=0.2, upper=0.8
    unlabeled_loss_weight=1.0,    # lambda_u
    ema_momentum=0.996,           # EMA teacher momentum
    # General
    epochs=50,
    batch_size=8,
    optimizer=None,
    approach='imas',
    channel_weights=(1.0, 1.0, 3.0),  # keep pos-channel heavier if you want
):
    """
    iMAS-style trainer for HeLa (3 independent sigmoids).
    - Supervised: channel-weighted BCE
    - Unsupervised: channel-weighted BCE to teacher soft labels on two adaptive strong branches
      (A_Is intensity mix, A_Cs CutMix), with teacher confidence gating and per-instance weighting (1-γ_i).
    """
    import os, math, numpy as np, tensorflow as tf, tensorflow_addons as tfa
    from functions import (
        load_images_and_masks_hela, load_unlabeled_images, benchmark_hela,
    )
    import paths

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(1e-4)

    # ----------------- Utils -----------------
    @tf.function(reduce_retracing=True)
    def random_hflip_batch(x, y=None):
        b = tf.shape(x)[0]
        flip = tf.less(tf.random.uniform([b, 1, 1, 1]), 0.5)
        x_f = tf.where(flip, tf.reverse(x, axis=[2]), x)
        if y is not None:
            y_f = tf.where(flip, tf.reverse(y, axis=[2]), y)
            return x_f, y_f
        return x_f, None

    @tf.function(reduce_retracing=True)
    def weak_gray_batch(x):
        x = tf.cast(x, tf.float32)
        alpha = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.9, 1.1, dtype=x.dtype)
        beta  = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -10.0, 10.0, dtype=x.dtype)
        return tf.clip_by_value(x * alpha + beta, 0., 255.)

    @tf.function(reduce_retracing=True)
    def strong_gray_batch(x):
        x = tf.cast(x, tf.float32)
        alpha = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.7, 1.3, dtype=x.dtype)
        beta  = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -25.0, 25.0, dtype=x.dtype)
        x = tf.clip_by_value(x * alpha + beta, 0., 255.)
        x = tf.clip_by_value(x + tf.random.normal(tf.shape(x), stddev=10.0, dtype=x.dtype), 0., 255.)
        x = tfa.image.gaussian_filter2d(x, filter_shape=(7, 7))
        return x

    def _clone_model(m):
        clone = tf.keras.models.clone_model(m)
        _ = clone(tf.zeros([1, h, w, c], tf.float32))
        clone.set_weights(m.get_weights())
        return clone

    #@tf.function(reduce_retracing=True)
    #def _ema_update(teacher_vars, student_vars, momentum: tf.Tensor):
    #    for tw, sw in zip(teacher_vars, student_vars):
    #        tw.assign(momentum * tw + (1.0 - momentum) * sw)

    # channel weights
    w_ch = tf.reshape(tf.constant(channel_weights, tf.float32), [1,1,1,3])

    # BCE (supervised) averaged over all pixels/channels
    def weighted_bce_full(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), eps, 1. - eps)
        per_ch = -(y_true * tf.math.log(y_pred) + (1. - y_true) * tf.math.log(1. - y_pred))  # [B,H,W,3]
        per_ch = per_ch * w_ch
        return tf.reduce_mean(per_ch)

    # BCE with soft teacher targets, masked by teacher confidence, returns per-instance vector [B]
    def weighted_bce_soft_targets_masked(t_probs, s_probs, mask01):
        eps = 1e-7
        s_probs = tf.clip_by_value(tf.cast(s_probs, tf.float32), eps, 1. - eps)
        t_probs = tf.clip_by_value(tf.cast(t_probs, tf.float32), eps, 1. - eps)
        per_ch = -(t_probs * tf.math.log(s_probs) + (1. - t_probs) * tf.math.log(1. - s_probs))  # [B,H,W,3]
        per_ch = per_ch * w_ch
        masked = per_ch * mask01  # broadcast to 3 channels
        # per-instance normalization by (#confident_pixels * 3)
        num_conf = tf.reduce_sum(mask01, axis=[1,2,3])  # [B]
        num_conf = tf.maximum(num_conf, 1e-6)
        sums = tf.reduce_sum(masked, axis=[1,2,3])      # [B]
        return sums / num_conf                           # [B]

    # Teacher confidence mask per channel using band thresholds
    @tf.function(reduce_retracing=True)
    def teacher_conf_mask_from_probs(teacher_probs, lower, upper):
        conf = tf.cast((teacher_probs < lower) | (teacher_probs > upper), tf.float32)  # [B,H,W,3]
        return conf

    # Hardness for multilabel (3 sigmoids): class-weighted symmetric IoU on binarized predictions
    def compute_hardness_gamma_sigmoid(ps_w, pt_w, lower, upper):
        # binarize at 0.5 for IoU; class frequency from teacher maps
        s_bin = tf.cast(ps_w > 0.5, tf.float32)   # [B,H,W,3]
        t_bin = tf.cast(pt_w > 0.5, tf.float32)

        # per-channel IoU
        inter_st = tf.reduce_sum(s_bin * t_bin, axis=[1,2])                           # [B,3]
        union_st = tf.reduce_sum(s_bin + t_bin - s_bin*t_bin, axis=[1,2]) + 1e-6      # [B,3]
        iou_st = inter_st / union_st                                                  # [B,3]

        inter_ts = inter_st  # symmetric
        union_ts = union_st
        iou_ts = inter_ts / union_ts                                                  # [B,3]

        # class weights from reference map (teacher for st, student for ts)
        freq_t = tf.reduce_sum(t_bin, axis=[1,2]) + 1e-6                               # [B,3]
        w_t = freq_t / tf.reduce_sum(freq_t, axis=-1, keepdims=True)                  # [B,3]

        freq_s = tf.reduce_sum(s_bin, axis=[1,2]) + 1e-6
        w_s = freq_s / tf.reduce_sum(freq_s, axis=-1, keepdims=True)                  # [B,3]

        wiou_sp = tf.reduce_sum(w_s * iou_st, axis=-1)                                # [B]
        wiou_ts = tf.reduce_sum(w_t * iou_ts, axis=-1)                                # [B]

        # confidence ratios per image (mean over pixels&channels)
        # consider a pixel/channel confident if p<lower or p>upper
        def _rho(p):
            conf = tf.cast((p < lower) | (p > upper), tf.float32)    # [B,H,W,3]
            return tf.reduce_mean(conf, axis=[1,2,3])                # [B]
        rho_s = _rho(ps_w)
        rho_t = _rho(pt_w)

        score = 0.5 * (rho_s * wiou_sp + rho_t * wiou_ts)            # [B]
        gamma = 1.0 - score
        return tf.clip_by_value(gamma, 0.0, 1.0)                      # [B]

    # Adaptive intensity mix A_Is(u) = γ*Strong(u) + (1-γ)*Weak(u)
    @tf.function(reduce_retracing=True)
    def adaptive_intensity_mix(x_weak, gamma):
        x_strong = strong_gray_batch(x_weak)
        g = tf.reshape(gamma, [-1,1,1,1])
        return tf.cast(g * x_strong + (1.0 - g) * x_weak, tf.float32)

    # CutMix utils
    def _cutmix_mask_rect(B, H, W):
        lam = tf.clip_by_value(tf.random.uniform([B], 0.3, 0.7), 0.0, 1.0)
        cut_rat = tf.sqrt(1.0 - lam)
        cut_w = tf.cast(cut_rat * tf.cast(W, tf.float32), tf.int32)
        cut_h = tf.cast(cut_rat * tf.cast(H, tf.float32), tf.int32)
        cx = tf.random.uniform([B], 0, W, dtype=tf.int32)
        cy = tf.random.uniform([B], 0, H, dtype=tf.int32)
        x1 = tf.clip_by_value(cx - cut_w // 2, 0, W)
        y1 = tf.clip_by_value(cy - cut_h // 2, 0, H)
        x2 = tf.clip_by_value(cx + (cut_w - cut_w // 2), 0, W)
        y2 = tf.clip_by_value(cy + (cut_h - cut_h // 2), 0, H)

        def draw_box(i):
            yy = tf.range(H)[:,None]
            xx = tf.range(W)[None,:]
            in_box = tf.cast((yy >= y1[i]) & (yy < y2[i]) & (xx >= x1[i]) & (xx < x2[i]), tf.float32)  # [H,W]
            return tf.expand_dims(in_box, -1)
        return tf.map_fn(lambda i: draw_box(i), tf.range(B), fn_output_signature=tf.float32)  # [B,H,W,1]

    def model_adaptive_cutmix(x_weak, pt_w, gamma):
        B = tf.shape(x_weak)[0]; H = tf.shape(x_weak)[1]; W = tf.shape(x_weak)[2]
        asc = tf.argsort(gamma, direction='ASCENDING')
        dsc = tf.argsort(gamma, direction='DESCENDING')
        src_idx = tf.range(B)
        pair_a = tf.gather(asc, src_idx)
        pair_b = tf.gather(dsc, src_idx)
        m = _cutmix_mask_rect(B, H, W)  # [B,H,W,1]
        x_a = tf.gather(x_weak, pair_a); x_b = tf.gather(x_weak, pair_b)
        pt_a = tf.gather(pt_w,   pair_a); pt_b = tf.gather(pt_w,   pair_b)
        x_cm  = m * x_b  + (1.0 - m) * x_a
        pt_cm = m * pt_b + (1.0 - m) * pt_a
        p = tf.reduce_mean(gamma)
        do_apply = tf.less(tf.random.uniform([]), p)
        x_out  = tf.where(do_apply, x_cm, x_weak)
        pt_out = tf.where(do_apply, pt_cm, pt_w)
        return tf.cast(x_out, tf.float32), tf.cast(pt_out, tf.float32)

    # ----------------- Data -----------------
    l_imgs, lA, lD, lP = load_images_and_masks_hela(paths.HELA_TRAIN_LABELED_DIR)
    lab_masks_np = np.stack([lA, lD, lP], axis=-1).astype(np.float32)  # [N,H,W,3] in {0,1}
    u_imgs_np    = load_unlabeled_images(paths.HELA_TRAIN_UNLABELED_BRIGHTFIELD_DIR, read_as_grayscale=True)
    v_imgs, vA, vD, vP = load_images_and_masks_hela(paths.HELA_VAL_DIR)
    val_masks_np = np.stack([vA, vD, vP], axis=-1).astype(np.float32)

    lab_imgs  = tf.expand_dims(tf.convert_to_tensor(l_imgs,    tf.float32), -1)  # [N,H,W,1]
    unl_imgs  = tf.expand_dims(tf.convert_to_tensor(u_imgs_np, tf.float32), -1)  # [M,H,W,1]
    val_imgs  = tf.expand_dims(tf.convert_to_tensor(v_imgs,    tf.float32), -1)  # [V,H,W,1]
    lab_masks = tf.convert_to_tensor(lab_masks_np, tf.float32)                    # [N,H,W,3]
    val_masks = tf.convert_to_tensor(val_masks_np, tf.float32)                    # [V,H,W,3]

    # ----------------- Build & EMA teacher -----------------
    _ = model(tf.zeros([1, h, w, c], tf.float32), training=False)
    _build_optimizer_weights(optimizer, model.trainable_variables)
    teacher = _clone_model(model)
    teacher.trainable = False

    lower = float(1.0 - confidence_threshold)
    upper = float(confidence_threshold)

    @tf.function(reduce_retracing=True)
    def _forward_probs(mdl, x, training):
        # model outputs sigmoid probabilities for 3 channels
        p = tf.cast(mdl(x, training=training), tf.float32)
        return p


    @tf.function(reduce_retracing=True)
    def _train_step(xL, yL, xU, lambda_u: float, ema_m: float):
        # weak views
        xL, yL = random_hflip_batch(xL, yL)
        xU, _  = random_hflip_batch(xU, None)
        xLw = weak_gray_batch(xL)
        xUw = weak_gray_batch(xU)
    
        with tf.GradientTape() as tape:
            # supervised (channel-weighted BCE)
            p_lab = _forward_probs(model, xLw, training=True)
            Lx = weighted_bce_full(yL, p_lab)
    
            # teacher & student on weak
            pt_w = _forward_probs(teacher, xUw, training=False)  # [B,H,W,3]
            pt_w = tf.stop_gradient(pt_w)
    
            ps_w = _forward_probs(model, xUw, training=True)
            gamma = compute_hardness_gamma_sigmoid(ps_w, pt_w, lower, upper)  # [B]
            gamma = tf.stop_gradient(gamma)
    
            # A_Is: adaptive intensity mix
            x_Is = adaptive_intensity_mix(xUw, gamma)
            pt_mask = tf.stop_gradient(teacher_conf_mask_from_probs(pt_w, lower, upper))  # [B,H,W,3]
            p_Is = _forward_probs(model, x_Is, training=True)
            Lu_Is = weighted_bce_soft_targets_masked(pt_w, p_Is, pt_mask)                 # [B]
    
            # A_Cs: model-adaptive CutMix (with mixed teacher soft targets)
            x_Cs, pt_cm = model_adaptive_cutmix(xUw, pt_w, gamma)
            pt_cm = tf.stop_gradient(pt_cm)
            pt_cm_mask = tf.stop_gradient(teacher_conf_mask_from_probs(pt_cm, lower, upper))
            p_Cs = _forward_probs(model, x_Cs, training=True)
            Lu_Cs = weighted_bce_soft_targets_masked(pt_cm, p_Cs, pt_cm_mask)             # [B]
    
            # instance weighting (1 - γ_i)
            weights = 1.0 - gamma                                                         # [B]
            Lu_per = 0.5 * (Lu_Is + Lu_Cs) * weights
            Lu = tf.reduce_mean(Lu_per)
    
            total = Lx + tf.cast(lambda_u, tf.float32) * Lu
            scaled = optimizer.get_scaled_loss(total) if hasattr(optimizer, "get_scaled_loss") else total
    
        grads = tape.gradient(scaled, model.trainable_variables)
        if hasattr(optimizer, "get_unscaled_gradients"):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
        # EMA update
        #_ema_update(teacher, model, tf.constant(ema_m, tf.float32))
        #_ema_update(teacher.trainable_variables, model.trainable_variables, tf.constant(ema_m, tf.float32))
        _ema_update(teacher, model, float(ema_m))
    
        return total, Lx, Lu, tf.reduce_mean(gamma)

    @tf.function(reduce_retracing=True)
    def _val_step(x, y):
        p = _forward_probs(model, x, training=False)
        return weighted_bce_full(y, p)

    # ----------------- Loop -----------------
    history = {'val_loss': [], 'avg_hardness': []}
    best_val = float('inf')

    Nl = int(lab_imgs.shape[0]); Nu = int(unl_imgs.shape[0])
    nbl = math.ceil(Nl / batch_size); nbu = math.ceil(Nu / batch_size)
    steps = max(nbl, nbu)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        idxL = np.random.permutation(Nl)
        idxU = np.random.permutation(Nu)
        sum_total = sum_sup = sum_uns = sum_hard = 0.0

        for step in range(steps):
            sL = (step % nbl) * batch_size; sU = (step % nbu) * batch_size
            bL = idxL[sL:sL+batch_size];   bU = idxU[sU:sU+batch_size]
            xL = tf.gather(lab_imgs,  bL)
            yL = tf.gather(lab_masks, bL)
            xU = tf.gather(unl_imgs,  bU)

            total, Lx, Lu, hard = _train_step(xL, yL, xU, float(unlabeled_loss_weight), float(ema_momentum))
            sum_total += float(total.numpy())
            sum_sup   += float(Lx.numpy())
            sum_uns   += float(Lu.numpy())
            sum_hard  += float(hard.numpy())

            if (step+1) % 20 == 0 or (step+1) == steps:
                print(f"  step {step+1}/{steps} | loss {sum_total/(step+1):.4f} | sup {sum_sup/(step+1):.4f} | "
                      f"Lu {sum_uns/(step+1):.4f} | avg_hard {sum_hard/(step+1):.4f}")

        # validation
        V = int(val_imgs.shape[0]); vsteps = math.ceil(V / batch_size)
        v_losses = []
        for i in range(vsteps):
            s = i * batch_size; e = min(s + batch_size, V)
            v_losses.append(float(_val_step(val_imgs[s:e], val_masks[s:e]).numpy()))
        vloss = float(np.mean(v_losses)) if v_losses else float('inf')
        history['val_loss'].append(vloss); history['avg_hardness'].append(sum_hard/steps)
        print(f"Validation Loss: {vloss:.4f} | avg_hard {history['avg_hardness'][-1]:.4f}")

        if vloss < best_val:
            best_val = vloss
            model.save(filepath_h5, include_optimizer=False)

    # ------------- benchmarking -------------
    best_model = tf.keras.models.load_model(filepath_h5, compile=False)

    val_pred_dir       = os.path.join(paths.HELA_BASE_DIR, 'val_predictions', approach, modelname)
    test_pred_dir      = os.path.join(paths.HELA_BASE_DIR, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)

    mIoU_val, mIoU_ad_val, mcce_val = benchmark_hela(best_model, paths.HELA_VAL_DIR,  val_pred_dir,  h, w, c)
    mIoU_tst, mIoU_ad_tst, mcce_tst = benchmark_hela(best_model, paths.HELA_TEST_DIR, test_pred_dir, h, w, c)
    mIoU_unl, mIoU_ad_unl, mcce_unl = benchmark_hela(best_model, paths.HELA_TRAIN_UNLABELED_DIR, unlabeled_pred_dir, h, w, c)

    return (mIoU_val, mIoU_tst, mIoU_unl,
            mIoU_ad_val, mIoU_ad_tst, mIoU_ad_unl,
            mcce_val, mcce_tst, mcce_unl, history)



# -----------------------------------------------------------------------------
# iMAS — Multiclass softmax (Cityscapes/SUIM-ready)
# -----------------------------------------------------------------------------
def train_multiclass_imas(
    model: tf.keras.Model,
    filepath_h5: str,
    modelname: str,
    h: int, w: int, c: int,
    num_classes: int,
    class_to_color_mapping: dict,
    # iMAS hypers
    tau: float = 0.7,                      # lower for harder datasets (Cityscapes in paper)
    lambda_u: float = 3.0,
    ema_momentum: float = 0.996,
    # general
    epochs: int = 240,
    batch_size: int = 16,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
    # dataset I/O (explicit)
    train_labeled_images_dir: str = None,
    train_labeled_masks_dir: str = None,
    train_unlabeled_images_dir: str = None,
    val_images_dir: str = None,
    val_masks_dir: str = None,
    test_images_dir: str = None,
    test_masks_dir: str = None,
    train_unlabeled_masks_dir: str | None = None,   # optional for pred-dumps
    dataset_base_dir: str | None = None,
    dataset_name_for_aug: str = 'Cityscapes',
    approach: str = 'imas'
):
    """Faithful iMAS for K-class softmax. Matches the paper's teacher–student, hardness, adaptive aug, CutMix, and instance-weighted Lu.
    Uses CE on logits for both supervised and unsupervised (teacher provides soft targets).
    """
    from functions import load_labeled_data_multiclass_mask, load_unlabeled_images, benchmark_multiclass

    if optimizer is None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    # Data
    lab_imgs_np, lab_1hot_np = load_labeled_data_multiclass_mask(train_labeled_images_dir, train_labeled_masks_dir, num_classes)
    unl_imgs_np = load_unlabeled_images(train_unlabeled_images_dir)
    val_imgs_np, val_1hot_np = load_labeled_data_multiclass_mask(val_images_dir, val_masks_dir, num_classes)

    lab_ids_np = np.argmax(lab_1hot_np, axis=-1).astype(np.int32)
    val_ids_np = np.argmax(val_1hot_np, axis=-1).astype(np.int32)

    lab_imgs  = tf.convert_to_tensor(lab_imgs_np,  tf.float32)
    lab_ids   = tf.convert_to_tensor(lab_ids_np,   tf.int32)
    unl_imgs  = tf.convert_to_tensor(unl_imgs_np,  tf.float32)
    val_imgs  = tf.convert_to_tensor(val_imgs_np,  tf.float32)
    val_ids   = tf.convert_to_tensor(val_ids_np,   tf.int32)

    # Build + teacher clone
    _ = model(tf.zeros([1, h, w, c], tf.float32), training=False)
    _build_optimizer_weights(optimizer, model.trainable_variables)
    teacher = _clone_model(model)
    teacher.trainable = False

    @tf.function(reduce_retracing=True)
    def _forward_probs(mdl, x, training: bool):
        logits = tf.cast(mdl(x, training=training), tf.float32)
        return tf.nn.softmax(logits, axis=-1), logits

    @tf.function(reduce_retracing=True)
    def _train_step(xL, yL_ids, xU, tau_f: float, lambda_u_f: float, ema_m: float):
        # Weak views + mild jitter
        xL, yL_ids = random_hflip_batch(xL, yL_ids)
        xU, _      = random_hflip_batch(xU, None)
        xLw = weak_photometric_batch(xL)
        xUw = weak_photometric_batch(xU)
    
        with tf.GradientTape() as tape:
            # Supervised CE on logits
            p_lab, logits_lab = _forward_probs(model, xLw, training=True)
            Lx = supervised_loss_logits(yL_ids, logits_lab)
    
            # Teacher/Student weak
            pt_w, _ = _forward_probs(teacher, xUw, training=False)
            pt_w = tf.stop_gradient(pt_w)
    
            ps_w, _ = _forward_probs(model, xUw, training=True)
            gamma = compute_hardness_gamma(ps_w, pt_w, tau_f)
            gamma = tf.stop_gradient(gamma)
    
            # Adaptive intensity branch
            xAIs = adaptive_intensity_mix(xUw, dataset_name_for_aug, gamma)
            pt_mask = tf.stop_gradient(teacher_conf_mask_from_probs(pt_w, tau_f))
            _, logits_Is = _forward_probs(model, xAIs, training=True)
            Lu_Is = unsup_ce_from_teacher_probs(logits_Is, pt_w, pt_mask)  # [B]
    
            # Adaptive CutMix branch
            xACs, pt_cm = model_adaptive_cutmix(xUw, pt_w, gamma)
            pt_cm = tf.stop_gradient(pt_cm)
            pt_cm_mask = tf.stop_gradient(teacher_conf_mask_from_probs(pt_cm, tau_f))
            _, logits_Cs = _forward_probs(model, xACs, training=True)
            Lu_Cs = unsup_ce_from_teacher_probs(logits_Cs, pt_cm, pt_cm_mask)  # [B]
    
            weights = 1.0 - gamma
            Lu_per = 0.5 * (Lu_Is + Lu_Cs) * weights
            Lu = tf.reduce_mean(Lu_per)
    
            total = Lx + lambda_u_f * Lu
            scaled = optimizer.get_scaled_loss(total) if hasattr(optimizer, 'get_scaled_loss') else total
    
        grads = tape.gradient(scaled, model.trainable_variables)
        if hasattr(optimizer, 'get_unscaled_gradients'):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    
        #_ema_update(teacher, model, tf.constant(ema_m, tf.float32))
        #_ema_update(teacher.trainable_variables, model.trainable_variables, tf.constant(ema_m, tf.float32))
        _ema_update(teacher, model, float(ema_m))
    
        return total, Lx, Lu, tf.reduce_mean(gamma)


    @tf.function(reduce_retracing=True)
    def _val_step(x, y_ids):
        _, logits = _forward_probs(model, x, training=False)
        return supervised_loss_logits(y_ids, logits)

    history = {'val_loss': [], 'hardness': []}
    best_val = float('inf')

    Nl = int(lab_imgs.shape[0]); Nu = int(unl_imgs.shape[0])
    nbl = math.ceil(Nl / batch_size); nbu = math.ceil(Nu / batch_size)
    steps = max(nbl, nbu)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        idxL = np.random.permutation(Nl)
        idxU = np.random.permutation(Nu)
        sum_tot = sum_sup = sum_uns = sum_hard = 0.0

        for step in range(steps):
            sL = (step % nbl) * batch_size; sU = (step % nbu) * batch_size
            bL = idxL[sL:sL+batch_size];     bU = idxU[sU:sU+batch_size]
            xL = tf.gather(lab_imgs,  bL)
            yL = tf.gather(lab_ids,   bL)
            xU = tf.gather(unl_imgs,  bU)

            total, Lx, Lu, hard = _train_step(xL, yL, xU, float(tau), float(lambda_u), float(ema_momentum))
            sum_tot  += float(total.numpy())
            sum_sup  += float(Lx.numpy())
            sum_uns  += float(Lu.numpy())
            sum_hard += float(hard.numpy())

            if (step+1) % 20 == 0 or (step+1) == steps:
                print(f"  step {step+1}/{steps} | loss {sum_tot/(step+1):.4f} | sup {sum_sup/(step+1):.4f} | Lu {sum_uns/(step+1):.4f} | avg_hard {sum_hard/(step+1):.4f}")

        # Validation
        V = int(val_imgs.shape[0]); vsteps = math.ceil(V / batch_size)
        v_losses = []
        for i in range(vsteps):
            s = i * batch_size; e = min(s + batch_size, V)
            v_losses.append(float(_val_step(val_imgs[s:e], val_ids[s:e]).numpy()))
        vloss = float(np.mean(v_losses)) if v_losses else float('inf')
        history['val_loss'].append(vloss); history['hardness'].append(sum_hard/steps)
        print(f"Validation Loss: {vloss:.4f} | avg_hard {history['hardness'][-1]:.4f}")

        if vloss < best_val:
            best_val = vloss
            if filepath_h5.endswith('.h5'):
                model.save(filepath_h5, include_optimizer=False, save_format='h5')
            else:
                model.save(filepath_h5, include_optimizer=False)

    best_model = tf.keras.models.load_model(filepath_h5, compile=False)
    inference_model = best_model

    from functions import benchmark_multiclass
    val_pred_dir       = os.path.join(dataset_base_dir, 'val_predictions', approach, modelname)
    test_pred_dir      = os.path.join(dataset_base_dir, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(dataset_base_dir, 'train_unlabeled_predictions', approach, modelname)

    mPA_val,  mIoU_val  = benchmark_multiclass(inference_model, val_images_dir,  val_masks_dir,  val_pred_dir,  h, w, c, class_to_color_mapping, print_results=False)
    mPA_test, mIoU_test = benchmark_multiclass(inference_model, test_images_dir, test_masks_dir, test_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    mPA_unl,  mIoU_unl  = benchmark_multiclass(inference_model, train_unlabeled_images_dir, train_unlabeled_masks_dir, unlabeled_pred_dir, h, w, c, class_to_color_mapping, print_results=False)

    return mPA_val, mPA_test, mPA_unl, mIoU_val, mIoU_test, mIoU_unl, history




##############################################################################################################################################################################################
###                         UPS
##############################################################################################################################################################################################




class ImagePathLoader(Sequence):
    """
    A Keras-compatible Sequence that loads images from a list of file paths.
    This avoids loading an entire dataset into RAM.
    """
    def __init__(self, image_paths, batch_size, image_size):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.image_size = image_size # (height, width)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        for f in batch_paths:
            img = cv2.imread(f)
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            img = img / 255.0  # Normalize
            batch_images.append(img)
        return np.array(batch_images, dtype=np.float32)


def get_uncertainty_for_chunk(model, image_loader, num_passes=10, temperature=2.0):
    """
    Calculates MC Dropout uncertainty for a chunk of images provided by an ImagePathLoader.
    This version is truly memory-efficient as it never holds the full dataset or
    full aggregate tensors in memory.

    Args:
        model: Keras model with Dropout layers.
        image_loader: An ImagePathLoader instance for the current chunk.
        num_passes: Number of MC Dropout passes.
        temperature: Temperature for scaling logits before softmax.

    Returns:
        A tuple of (mean_probabilities, uncertainty_map) as NumPy arrays for the chunk.
    """
    # Get the total number of images and output shape for this chunk
    num_images_in_chunk = len(image_loader.image_paths)
    first_batch = image_loader[0]
    logits_shape = model(first_batch, training=False).shape
    output_shape = (num_images_in_chunk,) + logits_shape[1:]
    
    # These aggregate tensors are now small enough to fit in memory
    mean_agg = np.zeros(output_shape, dtype=np.float32)
    m2_agg = np.zeros(output_shape, dtype=np.float32)

    # Welford's algorithm
    for i in tqdm(range(1, num_passes + 1), desc="MC Passes on Chunk"):
        # For each pass, we reconstruct the full chunk's predictions
        current_pass_probs = model.predict(image_loader, verbose=0)
        
        # Apply temperature scaling and softmax
        scaled_logits = current_pass_probs / temperature
        # Softmax in a numerically stable way
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        delta = probs - mean_agg
        mean_agg += delta / i
        delta2 = probs - mean_agg
        m2_agg += delta * delta2

    if num_passes < 2:
        variance = np.zeros_like(mean_agg)
    else:
        variance = m2_agg / (num_passes - 1)
    
    std_dev = np.sqrt(variance)
    return mean_agg, std_dev


def get_uncertainty_and_predictions_mc_dropout_mem_efficient(model, images, num_passes=10, temperature=2.0, batch_size=16):
    """
    Performs multiple stochastic forward passes with extreme memory efficiency.
    It processes data batch-by-batch for both inference and statistics updates, 
    avoiding the creation of large intermediate tensors for the full dataset.
    This is critical for very large datasets like Cityscapes.

    Args:
        model: Keras model with Dropout layers.
        images: Input images NumPy array [N, H, W, C].
        num_passes: Number of MC Dropout passes.
        temperature: Temperature for scaling logits before softmax.
        batch_size: Batch size for inference passes.

    Returns:
        A tuple of (mean_probabilities [N,H,W,K], uncertainty_map [N,H,W,K]) as NumPy arrays.
    """
    dataset = tf.data.Dataset.from_tensor_slices(images).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Initialize tensors for online statistics on the CPU to avoid GPU OOM.
    with tf.device('/cpu:0'):
        # Get output shape from a single batch
        first_batch = next(iter(dataset))
        logits_shape = model(first_batch, training=False).shape
        output_shape = (images.shape[0],) + logits_shape[1:]
        
        # Use tf.Variable for in-place updates. This is the key to memory efficiency.
        mean_agg = tf.Variable(tf.zeros(output_shape, dtype=tf.float32), trainable=False, name="mean_agg")
        m2_agg = tf.Variable(tf.zeros(output_shape, dtype=tf.float32), trainable=False, name="m2_agg")

    # Welford's algorithm applied in a streaming, per-batch manner
    for i in tqdm(range(1, num_passes + 1), desc="MC Dropout Passes"):
        # Iterate over the dataset batch by batch for the current pass
        for batch_idx, batch_images in enumerate(dataset):
            # 1. Get predictions for the current batch
            logits = model(batch_images, training=True)
            scaled_logits = logits / temperature
            probs_batch = tf.nn.softmax(tf.cast(scaled_logits, tf.float32), axis=-1)
            
            # 2. Determine the slice of the aggregate tensors to update
            start_idx = batch_idx * batch_size
            end_idx = start_idx + tf.shape(batch_images)[0]
            
            # 3. Perform Welford's update on the slice
            with tf.device('/cpu:0'):
                # Slicing a tf.Variable creates a read-only tensor, so we read, compute, and assign back.
                mean_slice = tf.gather(mean_agg, tf.range(start_idx, end_idx))
                m2_slice = tf.gather(m2_agg, tf.range(start_idx, end_idx))

                delta = probs_batch - mean_slice
                updated_mean_slice = mean_slice + delta / i
                delta2 = probs_batch - updated_mean_slice
                updated_m2_slice = m2_slice + delta * delta2
                
                # Assign the updated slice back to the main variable
                mean_agg.scatter_update(tf.IndexedSlices(updated_mean_slice, tf.range(start_idx, end_idx)))
                m2_agg.scatter_update(tf.IndexedSlices(updated_m2_slice, tf.range(start_idx, end_idx)))

    # Finalize calculations on the CPU
    with tf.device('/cpu:0'):
        if num_passes < 2:
            variance = tf.zeros_like(mean_agg.value())
        else:
            variance = m2_agg.value() / (num_passes - 1)
        
        std_dev_per_class = tf.sqrt(variance)

    # Return as NumPy arrays to free up TensorFlow memory
    return mean_agg.numpy(), std_dev_per_class.numpy()



def train_multiclass_ups_mem_eff(
    model, base_model_to_save, filepath_h5, modelname,
    # Labeled data passed as a tuple of NumPy arrays
    labeled_data_np,
    # Unlabeled data and pseudo-labels are passed as separate NumPy arrays
    unlabeled_imgs_np,
    pseudo_labels,
    selection_mask_pos,
    selection_mask_neg,
    # Validation data can remain a tf.data.Dataset
    val_ds,
    lambda_u, epochs, optimizer, h, w, c, num_classes, class_to_color_mapping,
    batch_size,
    supervised_warmup_steps=50,
    # --- Benchmarking Parameters ---
    val_images_dir=None, val_masks_dir=None,
    test_images_dir=None, test_masks_dir=None,
    train_unlabeled_images_dir=None, train_unlabeled_masks_dir=None,
    dataset_base_dir=None,
    approach='ups'
):
    """
    Memory-Efficient UPS training loop for multi-class semantic segmentation.

    This version uses a `tf.data.Dataset.from_generator` to stream batches of
    unlabeled data, pseudo-labels, and selection masks. This approach avoids
    loading the entire set of (potentially huge) pseudo-label tensors into memory,
    preventing OOM errors on large datasets like Cityscapes.

    Args:
        model: The student Keras model to be trained.
        base_model_to_save: A non-dropout version of the model for saving weights.
        filepath_h5: Path to save the best model weights.
        modelname: Name for the current model run (used in logging/saving).
        labeled_data_np: A tuple of (images_np, labels_np) for labeled data.
        unlabeled_imgs_np: NumPy array of unlabeled images.
        pseudo_labels: NumPy array of one-hot encoded pseudo-labels for unlabeled images.
        selection_mask_pos: NumPy array of the positive selection mask (g_pos).
        selection_mask_neg: NumPy array of the negative selection mask (g_neg).
        val_ds: A tf.data.Dataset for validation.
        lambda_u: Weight for the unlabeled loss component.
        epochs: Number of epochs to train the student.
        optimizer: The TensorFlow optimizer.
        h, w, c: Image height, width, and channels.
        num_classes: Number of segmentation classes.
        class_to_color_mapping: Dictionary for mapping class IDs to colors for benchmarking.
        batch_size: The training batch size.
        supervised_warmup_steps: Number of initial steps to train on labeled data only.
        ... (benchmarking paths and names)
    """
    # These imports are here to make the function self-contained, assuming they are available
    from functions import benchmark_multiclass, _ups_unlabeled_loss_singlelabel, ups_data_generator

    ce_sup = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # --- START: Training Step Functions ---
    @tf.function
    def supervised_only_step(data):
        x_l, y_l = data
        with tf.GradientTape() as tape:
            logits_l = model(x_l, training=True)
            loss_l = ce_sup(y_l, logits_l)
            scaled_loss = optimizer.get_scaled_loss(loss_l) if hasattr(optimizer, 'get_scaled_loss') else loss_l
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        if hasattr(optimizer, 'get_unscaled_gradients'): grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_l

    @tf.function
    def combined_step(data):
        (x_l, y_l), (x_u, y_pseudo_1h, g_pos, g_neg) = data
        with tf.GradientTape() as tape:
            logits_l = model(x_l, training=True)
            loss_l = ce_sup(y_l, logits_l)
            logits_u = model(x_u, training=True)
            loss_u = _ups_unlabeled_loss_singlelabel(y_pseudo_1h, logits_u, g_pos, g_neg)
            total_loss = loss_l + lambda_u * loss_u
            scaled_loss = optimizer.get_scaled_loss(total_loss) if hasattr(optimizer, 'get_scaled_loss') else total_loss
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        if hasattr(optimizer, 'get_unscaled_gradients'): grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_l, loss_u
    
    @tf.function
    def val_step(x, y):
        logits = model(x, training=False)
        return ce_sup(y, logits)
    # --- END: Training Step Functions ---

    # Supervised Warmup Phase (if specified)
    if supervised_warmup_steps > 0:
        print(f"Starting supervised warmup for {supervised_warmup_steps} steps...")
        lab_ds_for_warmup = tf.data.Dataset.from_tensor_slices(labeled_data_np).shuffle(len(labeled_data_np[0])).batch(batch_size).repeat()
        warmup_losses = []
        for i, (x_l, y_l) in enumerate(lab_ds_for_warmup.take(supervised_warmup_steps)):
            loss = supervised_only_step((x_l, y_l))
            warmup_losses.append(loss.numpy())
        print(f"Warmup complete. Average supervised loss: {np.mean(warmup_losses):.4f}")

    # --- START: Main Training using Generator ---
    best_val_loss = float('inf')
    history = {'loss_sup': [], 'loss_unsup': [], 'val_loss': []}
    
    num_unlabeled = len(unlabeled_imgs_np)
    num_steps_per_epoch = num_unlabeled // batch_size
    
    # Bundle all unlabeled data components into a single tuple for the generator
    unlabeled_data_for_gen = (unlabeled_imgs_np, pseudo_labels, selection_mask_pos, selection_mask_neg)
    
    # Define the generator's output signature. This is CRUCIAL for tf.data.
    # It tells TensorFlow the shape and type of the tensors the generator will yield.
    output_signature = (
        # Labeled data part: (image, label)
        (tf.TensorSpec(shape=(None, h, w, c), dtype=tf.float32), 
         tf.TensorSpec(shape=(None, h, w), dtype=tf.int32)),
        # Unlabeled data part: (image, pseudo_label, pos_mask, neg_mask)
        (tf.TensorSpec(shape=(None, h, w, c), dtype=tf.float32), 
         tf.TensorSpec(shape=(None, h, w, num_classes), dtype=tf.float32), 
         tf.TensorSpec(shape=(None, h, w, num_classes), dtype=tf.float32), 
         tf.TensorSpec(shape=(None, h, w, num_classes), dtype=tf.float32))
    )

    for epoch in range(epochs):
        # Create a new generator dataset for each epoch. The lambda function ensures
        # that the generator (and its internal shuffling) is re-initialized.
        train_ds_generator = tf.data.Dataset.from_generator(
            lambda: ups_data_generator(labeled_data_np, unlabeled_data_for_gen, batch_size),
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        sup_losses, unsup_losses = [], []
        
        # Use tqdm for a nice progress bar
        pbar = tqdm(train_ds_generator.take(num_steps_per_epoch), total=num_steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}")
        for data in pbar:
            l, u = combined_step(data)
            sup_losses.append(l.numpy())
            unsup_losses.append(u.numpy())
            # Update progress bar with running averages
            pbar.set_postfix({'sup_loss': f'{np.mean(sup_losses):.4f}', 'unsup_loss': f'{np.mean(unsup_losses):.4f}'})
        
        # Validation loop
        val_losses = []
        for x_v, y_v in val_ds:
            val_losses.append(val_step(x_v, y_v).numpy())
        
        avg_sup = np.mean(sup_losses) if sup_losses else 0
        avg_unsup = np.mean(unsup_losses) if unsup_losses else 0
        avg_val = np.mean(val_losses) if val_losses else float('inf')
        
        history['loss_sup'].append(avg_sup)
        history['loss_unsup'].append(avg_unsup)
        history['val_loss'].append(avg_val)
        print(f"Epoch {epoch+1}/{epochs} -> Sup Loss: {avg_sup:.4f}, Unsup Loss: {avg_unsup:.4f}, Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            print(f"  -> Val loss improved from {best_val_loss:.4f} to {avg_val:.4f}. Saving model.")
            best_val_loss = avg_val
            base_model_to_save.set_weights(model.get_weights())
            base_model_to_save.save(filepath_h5, include_optimizer=False)
    # --- END: Main Training ---
    
    # --- START: Benchmarking ---
    print("\nTraining complete. Loading best model for final benchmarking...")
    best_model = tf.keras.models.load_model(filepath_h5, compile=False)
    
    val_pred_dir = os.path.join(dataset_base_dir, 'val_predictions', approach, modelname)
    test_pred_dir = os.path.join(dataset_base_dir, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(dataset_base_dir, 'train_unlabeled_predictions', approach, modelname)
    
    print("Benchmarking on Validation set...")
    mPA_val, mIoU_val = benchmark_multiclass(best_model, val_images_dir, val_masks_dir, val_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    
    print("Benchmarking on Test set...")
    mPA_test, mIoU_test = benchmark_multiclass(best_model, test_images_dir, test_masks_dir, test_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    
    print("Benchmarking on Unlabeled Training set...")
    mPA_unl, mIoU_unl = benchmark_multiclass(best_model, train_unlabeled_images_dir, train_unlabeled_masks_dir, unlabeled_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    
    return mPA_val, mPA_test, mPA_unl, mIoU_val, mIoU_test, mIoU_unl, history


#def create_image_dataset_from_files(image_dir, image_height, image_width, batch_size):
#    """
#    Creates an efficient, file-based tf.data.Dataset that streams images from disk.
#    """
#    # Get a list of all image file paths
#    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png"))) # Or "*.jpg", etc.
#    num_images = len(image_paths)
#    
#    # Create a dataset of file paths (very lightweight)
#    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
#    
#    def load_and_preprocess_image(path):
#        # Read the file from disk
#        img = tf.io.read_file(path)
#        # Decode the image
#        img = tf.io.decode_png(img, channels=3)
#        # Resize to the model's expected input
#        img = tf.image.resize(img, [image_height, image_width])
#        # Normalize to [0, 1]
#        img = tf.cast(img, tf.float32) / 255.0
#        return img
#
#    # Map the loading function to each path, batch, and prefetch for performance
#    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
#    image_ds = image_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#    
#    return image_ds, num_images
#
#
#@tf.function
#def _process_batch_for_pseudo_labels(model, batch_images, num_passes, temperature, TAU_POS, KAPPA_POS, TAU_NEG, KAPPA_NEG, NUM_CLASSES):
#    """
#    Performs the entire pseudo-label generation process for a single batch in memory.
#    This is a highly efficient, streamlined function.
#    """
#    # --- Part 1: MC Dropout Predictions (from _predict_batch_mc) ---
#    
#    # Run one pass to get the shape and initialize stats
#    initial_logits = model(batch_images, training=True)
#    initial_probs = tf.nn.softmax(tf.cast(initial_logits / temperature, tf.float32), axis=-1)
#    mean_agg = tf.identity(initial_probs)
#    m2_agg = tf.zeros_like(initial_probs)
#
#    # Loop for the remaining passes
#    i = tf.constant(2)
#    cond = lambda i, mean_agg, m2_agg: i <= num_passes
#    def body(i, mean_agg, m2_agg):
#        logits = model(batch_images, training=True)
#        probs = tf.nn.softmax(tf.cast(logits / temperature, tf.float32), axis=-1)
#        delta = probs - mean_agg
#        mean_agg += delta / tf.cast(i, dtype=tf.float32)
#        delta2 = probs - mean_agg
#        m2_agg += delta * delta2
#        return i + 1, mean_agg, m2_agg
#    _, mean_agg, m2_agg = tf.while_loop(cond, body, [i, mean_agg, m2_agg])
#
#    if num_passes < 2:
#        variance = tf.zeros_like(mean_agg)
#    else:
#        variance = m2_agg / tf.cast(num_passes - 1, dtype=tf.float32)
#    uncertainty_batch = tf.sqrt(variance)
#    mean_probs_batch = mean_agg
#
#    # --- Part 2: Generate Masks and Labels (from generate_pseudo_data...) ---
#    
#    pos_select_mask_bool = (mean_probs_batch >= TAU_POS) & (uncertainty_batch <= KAPPA_POS)
#    neg_select_mask_bool = (mean_probs_batch <= TAU_NEG) & (uncertainty_batch <= KAPPA_NEG)
#    
#    pos_select_mask = tf.cast(pos_select_mask_bool, tf.float32)
#    neg_select_mask = tf.cast(neg_select_mask_bool, tf.float32)
#    
#    argmax_indices = tf.argmax(mean_probs_batch, axis=-1)
#    pseudo_labels_1h = tf.one_hot(argmax_indices, NUM_CLASSES, dtype=tf.float32)
#    
#    return pos_select_mask, neg_select_mask, pseudo_labels_1h
#
#
#
#
#
#
#def generate_pseudo_data_in_batches(
#    mean_probs_mmap, uncertainty_mmap,
#    TAU_POS, KAPPA_POS, TAU_NEG, KAPPA_NEG,
#    NUM_CLASSES, BATCH_SIZE_CPU, temp_dir="."
#):
#    """
#    Processes large prediction memmaps in batches to generate pseudo-labels and masks
#    without exhausting system RAM.
#    """
#    print("Processing predictions in batches to generate pseudo-data...")
#    num_images = mean_probs_mmap.shape[0]
#    output_shape = mean_probs_mmap.shape
#
#    # 1. Create new memmap files for the outputs of this stage
#    pos_mask_path = os.path.join(temp_dir, 'pos_select_mask.mmap')
#    neg_mask_path = os.path.join(temp_dir, 'neg_select_mask.mmap')
#    pseudo_labels_path = os.path.join(temp_dir, 'pseudo_labels_1h.mmap')
#
#    pos_mask_mmap = np.memmap(pos_mask_path, dtype='float32', mode='w+', shape=output_shape)
#    neg_mask_mmap = np.memmap(neg_mask_path, dtype='float32', mode='w+', shape=output_shape)
#    pseudo_labels_mmap = np.memmap(pseudo_labels_path, dtype='float32', mode='w+', shape=output_shape)
#
#    # 2. Loop through the data in manageable chunks
#    for start in tqdm(range(0, num_images, BATCH_SIZE_CPU), desc="Generating Pseudo-Labels"):
#        end = min(start + BATCH_SIZE_CPU, num_images)
#
#        # 3. Load only a small batch into RAM
#        mean_batch = mean_probs_mmap[start:end]
#        uncertainty_batch = uncertainty_mmap[start:end]
#
#        # 4. Perform all calculations on the small batch
#        pos_bool_batch = (mean_batch >= TAU_POS) & (uncertainty_batch <= KAPPA_POS)
#        neg_bool_batch = (mean_batch <= TAU_NEG) & (uncertainty_batch <= KAPPA_NEG)
#        
#        argmax_batch = np.argmax(mean_batch, axis=-1)
#        pseudo_labels_1h_batch = tf.one_hot(argmax_batch, NUM_CLASSES, dtype=tf.float32).numpy()
#
#        # 5. Write the processed batch to the output memmap files
#        pos_mask_mmap[start:end] = pos_bool_batch.astype(np.float32)
#        neg_mask_mmap[start:end] = neg_bool_batch.astype(np.float32)
#        pseudo_labels_mmap[start:end] = pseudo_labels_1h_batch
#        
#    # Return the new memmap objects
#    return pos_mask_mmap, neg_mask_mmap, pseudo_labels_mmap
#
#def get_uncertainty_and_predictions_mc_dropout_gpu_safe(
#    model, images, num_passes=10, temperature=2.0, batch_size=1, temp_dir="."
#):
#    """
#    Processes batches on the GPU and saves results directly to disk.
#    
#    *** REQUIRES MIXED PRECISION TO BE ENABLED to prevent GPU OOM errors. ***
#    """
#    dataset = tf.data.Dataset.from_tensor_slices(images).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#
#    output_shape = (images.shape[0],) + model.output_shape[1:]
#    mean_path = os.path.join(temp_dir, 'mean_probs.mmap')
#    uncertainty_path = os.path.join(temp_dir, 'uncertainty.mmap')
#    
#    mean_probs_mmap = np.memmap(mean_path, dtype='float32', mode='w+', shape=output_shape)
#    uncertainty_mmap = np.memmap(uncertainty_path, dtype='float32', mode='w+', shape=output_shape)
#    
#    print("Processing batches on GPU and saving results to disk...")
#    start_index = 0
#    for batch_images in tqdm(dataset):
#        mean_probs_batch, uncertainty_batch = _predict_batch_mc(
#            model, batch_images, num_passes, temperature
#        )
#        
#        batch_size_current = batch_images.shape[0]
#        end_index = start_index + batch_size_current
#        
#        # --- FIX IS HERE ---
#        # The .numpy() method handles the GPU-to-CPU data transfer automatically.
#        mean_probs_mmap[start_index:end_index, ...] = mean_probs_batch.numpy()
#        uncertainty_mmap[start_index:end_index, ...] = uncertainty_batch.numpy()
#        # --- END FIX ---
#        
#        start_index = end_index
#
#    print(f"Results saved to {mean_path} and {uncertainty_path}")
#    return mean_probs_mmap, uncertainty_mmap
#
#
## Use tf.function for a significant performance boost
#@tf.function
#def _predict_batch_mc(model, batch_images, num_passes, temperature):
#    """
#    Helper function to run MC passes on a single batch using a memory-efficient
#    iterative algorithm (Welford's) to prevent GPU OOM errors.
#    """
#    # Get the shape for the statistics tensors by running one prediction.
#    initial_logits = model(batch_images, training=True)
#    initial_probs = tf.nn.softmax(tf.cast(initial_logits / temperature, tf.float32), axis=-1)
#
#    # Initialize tensors for Welford's algorithm on the same device as the model's output.
#    mean_agg = tf.identity(initial_probs) # Start with the first pass
#    m2_agg = tf.zeros_like(initial_probs)
#
#    # Loop for the remaining passes (from 2 to num_passes).
#    # We use tf.while_loop for better graph compilation and control.
#    i = tf.constant(2)
#    
#    cond = lambda i, mean_agg, m2_agg: i <= num_passes
#    
#    def body(i, mean_agg, m2_agg):
#        logits = model(batch_images, training=True)
#        probs = tf.nn.softmax(tf.cast(logits / temperature, tf.float32), axis=-1)
#        
#        # Welford's algorithm update steps
#        delta = probs - mean_agg
#        mean_agg += delta / tf.cast(i, dtype=tf.float32)
#        delta2 = probs - mean_agg
#        m2_agg += delta * delta2
#        
#        return i + 1, mean_agg, m2_agg
#
#    _, mean_agg, m2_agg = tf.while_loop(cond, body, [i, mean_agg, m2_agg])
#
#    # Finalize calculations
#    if num_passes < 2:
#        variance = tf.zeros_like(mean_agg)
#    else:
#        variance = m2_agg / tf.cast(num_passes - 1, dtype=tf.float32)
#    
#    std_dev = tf.sqrt(variance)
#
#    return mean_agg, std_dev
#
#def get_uncertainty_and_predictions_mc_dropout_efficient(model, images, num_passes=10, temperature=2.0, batch_size=16):
#    """
#    Performs MC Dropout in a memory-efficient way by processing batch-by-batch.
#    
#    Args:
#        model: Keras model with Dropout layers.
#        images: Input images NumPy array [N, H, W, C].
#        num_passes: Number of MC Dropout passes.
#        temperature: Temperature for scaling logits before softmax.
#        batch_size: Batch size for inference passes.
#
#    Returns:
#        A tuple of (mean_probabilities [N,H,W,K], uncertainty_map [N,H,W,K]).
#    """
#    dataset = tf.data.Dataset.from_tensor_slices(images).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#    
#    # Lists to store the final results from each batch
#    all_mean_probs = []
#    all_uncertainties = []
#
#    print("Processing batches memory-efficiently...")
#    for batch_images in tqdm(dataset):
#        # Perform all MC passes for the current batch
#        mean_probs_batch, uncertainty_batch = _predict_batch_mc(model, batch_images, num_passes, temperature)
#        
#        # Append results to lists (these are now small tensors)
#        all_mean_probs.append(mean_probs_batch)
#        all_uncertainties.append(uncertainty_batch)
#        
#    # Concatenate all the batch results at the end
#    print("Concatenating final results...")
#    final_mean_probs = tf.concat(all_mean_probs, axis=0)
#    final_uncertainties = tf.concat(all_uncertainties, axis=0)
#
#    return final_mean_probs, final_uncertainties





























def get_uncertainty_and_predictions_mc_dropout(model, images, num_passes=10, temperature=2.0, batch_size=16):
    """
    Performs multiple stochastic forward passes to get mean predictions and PER-CLASS uncertainty.
    Uses a memory-efficient online algorithm (Welford's) to avoid storing all passes in memory,
    preventing OOM errors on large datasets. Includes a progress bar.
    Args:
        model: Keras model with Dropout layers.
        images: Input images NumPy array [N, H, W, C].
        num_passes: Number of MC Dropout passes.
        temperature: Temperature for scaling logits before softmax.
        batch_size: Batch size for inference passes.
    Returns:
        A tuple of (mean_probabilities [N,H,W,K], uncertainty_map [N,H,W,K]).
    """
    dataset = tf.data.Dataset.from_tensor_slices(images).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Initialize tensors for online statistics on the CPU to avoid GPU OOM.
    # CRITICAL FIX: Use tf.Variable for mutable state.
    with tf.device('/cpu:0'):
        first_batch = next(iter(dataset))
        logits_shape = model(first_batch, training=False).shape # Use training=False for shape inference, it's safer.
        output_shape = (images.shape[0],) + logits_shape[1:]
        
        mean_agg = tf.Variable(tf.zeros(output_shape, dtype=tf.float32), trainable=False)
        m2_agg = tf.Variable(tf.zeros(output_shape, dtype=tf.float32), trainable=False)

    # Welford's algorithm for online variance
    for i in tqdm(range(1, num_passes + 1), desc="MC Dropout Passes"):
        current_pass_probs_list = []
        for batch_images in dataset:
            logits = model(batch_images, training=True) # training=True to enable dropout
            scaled_logits = logits / temperature
            probs = tf.nn.softmax(tf.cast(scaled_logits, tf.float32), axis=-1)
            current_pass_probs_list.append(probs.numpy())
        
        current_pass_probs = np.concatenate(current_pass_probs_list, axis=0)
        
        with tf.device('/cpu:0'):
            # Convert numpy array back to tensor for TF operations
            current_pass_tensor = tf.convert_to_tensor(current_pass_probs, dtype=tf.float32)
            delta = current_pass_tensor - mean_agg
            mean_agg.assign_add(delta / i)
            delta2 = current_pass_tensor - mean_agg
            m2_agg.assign_add(delta * delta2)

    # Finalize calculations
    with tf.device('/cpu:0'):
        if num_passes < 2:
            variance = tf.zeros_like(mean_agg)
        else:
            variance = m2_agg / (num_passes - 1)
        
        std_dev_per_class = tf.sqrt(variance)

    return mean_agg.value(), std_dev_per_class # Return tensors, not variables


def get_uncertainty_and_predictions_mc_dropout_sigmoid(model, images, num_passes=10, batch_size=16):
    """
    Performs multiple stochastic forward passes for multi-label sigmoid models (HeLa),
    returning PER-CHANNEL uncertainty. Uses a memory-efficient online algorithm.
    Args:
        model: Keras model with Dropout layers and sigmoid output.
        images: Input images NumPy array [N, H, W, C].
        num_passes: Number of MC Dropout passes.
        batch_size: Batch size for inference passes.
    Returns:
        A tuple of (mean_probabilities [N,H,W,C], uncertainty_map [N,H,W,C]).
    """
    dataset = tf.data.Dataset.from_tensor_slices(images).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Initialize tensors for online statistics on the CPU to avoid GPU OOM.
    with tf.device('/cpu:0'):
        first_batch = next(iter(dataset))
        probs_shape = model(first_batch, training=False).shape
        output_shape = (images.shape[0],) + probs_shape[1:]
        
        mean_agg = tf.Variable(tf.zeros(output_shape, dtype=tf.float32), trainable=False)
        m2_agg = tf.Variable(tf.zeros(output_shape, dtype=tf.float32), trainable=False)

    # Welford's algorithm for online variance
    for i in tqdm(range(1, num_passes + 1), desc="MC Dropout Passes (Sigmoid)"):
        current_pass_probs_list = []
        for batch_images in dataset:
            # Model with sigmoid output already produces probabilities
            probs = model(batch_images, training=True) # training=True to enable dropout
            current_pass_probs_list.append(tf.cast(probs, tf.float32).numpy())
        
        current_pass_probs = np.concatenate(current_pass_probs_list, axis=0)
        
        with tf.device('/cpu:0'):
            current_pass_tensor = tf.convert_to_tensor(current_pass_probs, dtype=tf.float32)
            delta = current_pass_tensor - mean_agg
            mean_agg.assign_add(delta / i)
            delta2 = current_pass_tensor - mean_agg
            m2_agg.assign_add(delta * delta2)

    # Finalize calculations
    with tf.device('/cpu:0'):
        if num_passes < 2:
            variance = tf.zeros_like(mean_agg)
        else:
            variance = m2_agg / (num_passes - 1)
        
        std_dev_per_channel = tf.sqrt(variance)

    return mean_agg.value(), std_dev_per_channel



def _ups_unlabeled_loss_singlelabel(y_true_hard_1h, y_pred_logits, g_pos_mask, g_neg_mask):
    # Cast + sanitize dtypes early
    y_true_hard_1h = tf.cast(y_true_hard_1h, tf.float32)
    g_pos_mask = tf.cast(g_pos_mask, tf.float32)
    g_neg_mask = tf.cast(g_neg_mask, tf.float32)
    y_pred_logits = tf.cast(y_pred_logits, tf.float32)

    # Replace non-finite logits BEFORE any math
    y_pred_logits = tf.where(tf.math.is_finite(y_pred_logits), y_pred_logits, tf.zeros_like(y_pred_logits))

    # Clip logits for stability, softmax, then clip probs
    y_pred_logits = tf.clip_by_value(y_pred_logits, -30.0, 30.0)
    p = tf.nn.softmax(y_pred_logits, axis=-1)
    p = tf.where(tf.math.is_finite(p), p, tf.zeros_like(p) + 0.5)  # fallback to 0.5 if needed
    p = tf.clip_by_value(p, 1e-6, 1.0 - 1e-6)

    # Was the pseudo-label class selected POSITIVE?
    pos_for_pseudo = tf.reduce_sum(g_pos_mask * y_true_hard_1h, axis=-1) > 0.5  # [B,H,W] bool

    # POS branch: CE only when pseudo class is positively selected
    ce = -tf.reduce_sum(y_true_hard_1h * tf.math.log(p), axis=-1)  # [B,H,W]
    loss_pos = ce * tf.cast(pos_for_pseudo, tf.float32)

    # NEG branch: LNCE over selected negatives ONLY when no positive selected
    lnce_per_class = -(1.0 - y_true_hard_1h) * tf.math.log(1.0 - p)  # [B,H,W,K]
    lnce_selected = tf.reduce_sum(lnce_per_class * g_neg_mask, axis=-1)  # [B,H,W]
    num_neg_selected = tf.reduce_sum(g_neg_mask, axis=-1)  # [B,H,W]
    lnce_norm = tf.where(num_neg_selected > 0.0, lnce_selected / num_neg_selected, tf.zeros_like(lnce_selected))

    loss_neg = lnce_norm * tf.cast(tf.logical_not(pos_for_pseudo), tf.float32)

    # If neither pos nor neg selected, contribute 0
    has_any = (tf.reduce_sum(g_pos_mask, axis=-1) + tf.reduce_sum(g_neg_mask, axis=-1)) > 0.0  # [B,H,W] bool
    total = (loss_pos + loss_neg) * tf.cast(has_any, tf.float32)

    return tf.reduce_mean(total)



def _ups_unlabeled_loss_multilabel(y_true_soft, y_pred_probs, selection_mask, channel_weights):
    """
    Faithful UPS loss for multi-label (sigmoid) case (Eq. 4 from the paper).
    Args:
        y_true_soft: [B,H,W,C] soft pseudo-labels (mean probabilities from teacher).
        y_pred_probs: [B,H,W,C] student's sigmoid probabilities.
        selection_mask: [B,H,W,C] g_mask of selected positive/negative entries.
        channel_weights: [1,1,1,C] weights for each channel.
    """
    import tensorflow.keras.backend as K

    y_true_soft = tf.cast(y_true_soft, tf.float32)
    y_pred_probs = tf.cast(y_pred_probs, tf.float32)
    selection_mask = tf.cast(selection_mask, tf.float32)

    # Use K.binary_crossentropy to get per-element loss [B,H,W,C]
    bce_per_element = K.binary_crossentropy(y_true_soft, y_pred_probs, from_logits=False)
    
    # Apply channel weights and then the selection mask
    weighted_bce = bce_per_element * channel_weights
    masked_bce = weighted_bce * selection_mask
    
    # Normalize by number of selected entries per sample to make loss independent of selection ratio
    num_selected_per_sample = tf.reduce_sum(selection_mask, axis=[1, 2, 3])
    sum_loss_per_sample = tf.reduce_sum(masked_bce, axis=[1, 2, 3])
    
    loss_per_sample = tf.math.divide_no_nan(sum_loss_per_sample, num_selected_per_sample)
    
    return tf.reduce_mean(loss_per_sample)



def ups_data_generator(labeled_data, unlabeled_data, batch_size):
    """
    A generator that yields batches for UPS training.
    
    Args:
        labeled_data: A tuple of (lab_imgs_np, lab_ids_np).
        unlabeled_data: A tuple of (unl_imgs_np, pseudo_labels_1h, pos_mask, neg_mask).
        batch_size: The size of each batch.
    """
    lab_imgs, lab_ids = labeled_data
    unl_imgs, pseudo_1h, pos_mask, neg_mask = unlabeled_data

    # Create indices for both datasets
    lab_indices = np.arange(len(lab_imgs))
    unl_indices = np.arange(len(unl_imgs))

    # Shuffle unlabeled indices for each epoch
    np.random.shuffle(unl_indices)
    
    num_unl_batches = len(unl_imgs) // batch_size
    
    for i in range(num_unl_batches):
        # Get a batch of unlabeled data
        batch_unl_indices = unl_indices[i * batch_size : (i + 1) * batch_size]
        
        # Get a random batch of labeled data to match
        batch_lab_indices = np.random.choice(lab_indices, size=batch_size, replace=True)
        
        # Yield the combined batch
        yield (lab_imgs[batch_lab_indices], lab_ids[batch_lab_indices]), \
              (unl_imgs[batch_unl_indices], 
               pseudo_1h[batch_unl_indices], 
               pos_mask[batch_unl_indices], 
               neg_mask[batch_unl_indices])






def train_isic_ups(
    model, base_model_to_save, filepath_h5, modelname,
    labeled_ds, unlabeled_ds, val_ds,
    pseudo_labels, selection_mask_pos, selection_mask_neg,
    lambda_u, epochs, optimizer, h, w, c,
    approach='ups'
):
    """
    Faithful UPS training loop for one student on ISIC 2018.
    """
    import paths
    from functions import benchmark_ISIC2018

    first_l_batch = next(iter(labeled_ds.take(1)))
    B = int(first_l_batch[0].shape[0])
    
    unl_ds_with_pseudo = tf.data.Dataset.from_tensor_slices(
        (unlabeled_ds, pseudo_labels, selection_mask_pos, selection_mask_neg)
    ).shuffle(len(unlabeled_ds)).batch(B)

    train_ds = tf.data.Dataset.zip((labeled_ds.repeat(), unl_ds_with_pseudo.repeat()))
    
    ce_sup = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(data):
        (x_l, y_l), (x_u, y_pseudo_1h, g_pos, g_neg) = data
        
        with tf.GradientTape() as tape:
            logits_l = model(x_l, training=True)
            loss_l = ce_sup(y_l, logits_l)

            logits_u = model(x_u, training=True)
            loss_u = _ups_unlabeled_loss_singlelabel(y_pseudo_1h, logits_u, g_pos, g_neg)
            
            total_loss = loss_l + lambda_u * loss_u
            scaled_loss = optimizer.get_scaled_loss(total_loss) if hasattr(optimizer, 'get_scaled_loss') else total_loss
        
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        if hasattr(optimizer, 'get_unscaled_gradients'):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_l, loss_u

    @tf.function
    def val_step(x, y):
        logits = model(x, training=False)
        return ce_sup(y, logits)

    best_val_loss = float('inf')
    history = {'loss_sup': [], 'loss_unsup': [], 'val_loss': []}
    
    num_steps_per_epoch = tf.data.experimental.cardinality(unl_ds_with_pseudo).numpy()
    if num_steps_per_epoch == tf.data.experimental.UNKNOWN_CARDINALITY:
        num_steps_per_epoch = len(unlabeled_ds) // B

    for epoch in range(epochs):
        sup_losses, unsup_losses = [], []
        for data in train_ds.take(num_steps_per_epoch):
            l, u = train_step(data)
            sup_losses.append(l.numpy()); unsup_losses.append(u.numpy())
        
        val_losses = []
        for x_v, y_v in val_ds:
            val_losses.append(val_step(x_v, y_v).numpy())
        
        avg_sup=np.mean(sup_losses); avg_unsup=np.mean(unsup_losses); avg_val=np.mean(val_losses)
        history['loss_sup'].append(avg_sup); history['loss_unsup'].append(avg_unsup); history['val_loss'].append(avg_val)
        print(f"Epoch {epoch+1}/{epochs} -> Sup Loss: {avg_sup:.4f}, Unsup Loss: {avg_unsup:.4f}, Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            print("  -> Val loss improved. Saving model.")
            best_val_loss = avg_val
            base_model_to_save.set_weights(model.get_weights())
            base_model_to_save.save(filepath_h5, include_optimizer=False)

    best_model = tf.keras.models.load_model(filepath_h5, compile=False)
    inp=best_model.input; logits=best_model.output; probs=tf.keras.layers.Softmax(dtype='float32')(logits); lesion_prob=probs[...,1:]
    inference_model = tf.keras.Model(inputs=inp, outputs=lesion_prob)
    
    val_pred_dir=os.path.join(paths.ISIC_2018_BASE_DIR,'val_predictions',approach,modelname); test_pred_dir=os.path.join(paths.ISIC_2018_BASE_DIR,'test_predictions',approach,modelname); unlabeled_pred_dir=os.path.join(paths.ISIC_2018_BASE_DIR,'train_unlabeled_predictions',approach,modelname)
    mIoU_val,d_val=benchmark_ISIC2018(inference_model,paths.ISIC_2018_VAL_IMAGES_DIR,paths.ISIC_2018_VAL_MASKS_DIR,val_pred_dir,h,w,c)
    mIoU_test,d_test=benchmark_ISIC2018(inference_model,paths.ISIC_2018_TEST_IMAGES_DIR,paths.ISIC_2018_TEST_MASKS_DIR,test_pred_dir,h,w,c)
    mIoU_unl,d_unl=benchmark_ISIC2018(inference_model,paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR,paths.ISIC_2018_TRAIN_UNLABELED_MASKS_DIR,unlabeled_pred_dir,h,w,c)
    
    return mIoU_val, mIoU_test, mIoU_unl, d_val, d_test, d_unl, history


def train_multiclass_ups(
    model, base_model_to_save, filepath_h5, modelname,
    # Pass numpy arrays directly instead of tf.data.Dataset
    labeled_data_np, unlabeled_data_np, val_ds,
    pseudo_labels, selection_mask_pos, selection_mask_neg,
    lambda_u, epochs, optimizer, h, w, c, num_classes, class_to_color_mapping,
    batch_size, # Need batch_size to create the generator
    supervised_warmup_steps=50,
    # Dataset-specific paths for benchmarking
    val_images_dir=None, val_masks_dir=None,
    test_images_dir=None, test_masks_dir=None,
    train_unlabeled_images_dir=None, train_unlabeled_masks_dir=None,
    dataset_base_dir=None,
    approach='ups'
):
    import paths
    from functions import benchmark_multiclass

    ce_sup = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # --- START: Training Steps (Unchanged) ---
    @tf.function
    def supervised_only_step(data):
        x_l, y_l = data
        with tf.GradientTape() as tape:
            logits_l = model(x_l, training=True)
            loss_l = ce_sup(y_l, logits_l)
            scaled_loss = optimizer.get_scaled_loss(loss_l) if hasattr(optimizer, 'get_scaled_loss') else loss_l
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        if hasattr(optimizer, 'get_unscaled_gradients'): grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_l

    @tf.function
    def combined_step(data):
        (x_l, y_l), (x_u, y_pseudo_1h, g_pos, g_neg) = data
        with tf.GradientTape() as tape:
            logits_l = model(x_l, training=True)
            loss_l = ce_sup(y_l, logits_l)
            logits_u = model(x_u, training=True)
            loss_u = _ups_unlabeled_loss_singlelabel(y_pseudo_1h, logits_u, g_pos, g_neg)
            total_loss = loss_l + lambda_u * loss_u
            scaled_loss = optimizer.get_scaled_loss(total_loss) if hasattr(optimizer, 'get_scaled_loss') else total_loss
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        if hasattr(optimizer, 'get_unscaled_gradients'): grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_l, loss_u
    
    @tf.function
    def val_step(x, y):
        logits = model(x, training=False)
        return ce_sup(y, logits)
    # --- END: Training Steps ---

    # Supervised Warmup Phase (if needed)
    if supervised_warmup_steps > 0:
        print(f"Starting supervised warmup for {supervised_warmup_steps} steps...")
        lab_ds_for_warmup = tf.data.Dataset.from_tensor_slices(labeled_data_np).shuffle(len(labeled_data_np[0])).batch(batch_size).repeat()
        warmup_losses = []
        for i, (x_l, y_l) in enumerate(lab_ds_for_warmup):
            if i >= supervised_warmup_steps: break
            loss = supervised_only_step((x_l, y_l))
            warmup_losses.append(loss.numpy())
        print(f"Warmup complete. Average supervised loss: {np.mean(warmup_losses):.4f}")

    # --- START: Main Training using Generator ---
    best_val_loss = float('inf')
    history = {'loss_sup': [], 'loss_unsup': [], 'val_loss': []}
    
    num_unlabeled = len(unlabeled_data_np)
    num_steps_per_epoch = num_unlabeled // batch_size
    
    # Define the generator signature
    output_signature = (
        (tf.TensorSpec(shape=(None, h, w, c), dtype=tf.float32), tf.TensorSpec(shape=(None, h, w), dtype=tf.int32)),
        (tf.TensorSpec(shape=(None, h, w, c), dtype=tf.float32), 
         tf.TensorSpec(shape=(None, h, w, num_classes), dtype=tf.float32), 
         tf.TensorSpec(shape=(None, h, w, num_classes), dtype=tf.float32), 
         tf.TensorSpec(shape=(None, h, w, num_classes), dtype=tf.float32))
    )

    for epoch in range(epochs):
        # Create a new generator for each epoch to ensure shuffling
        train_generator = tf.data.Dataset.from_generator(
            lambda: ups_data_generator(labeled_data_np, (unlabeled_data_np, pseudo_labels, selection_mask_pos, selection_mask_neg), batch_size),
            output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

        sup_losses, unsup_losses = [], []
        for data in train_generator.take(num_steps_per_epoch):
            l, u = combined_step(data)
            sup_losses.append(l.numpy()); unsup_losses.append(u.numpy())
        
        val_losses = []
        for x_v, y_v in val_ds:
            val_losses.append(val_step(x_v, y_v).numpy())
        
        avg_sup=np.mean(sup_losses); avg_unsup=np.mean(unsup_losses); avg_val=np.mean(val_losses)
        history['loss_sup'].append(avg_sup); history['loss_unsup'].append(avg_unsup); history['val_loss'].append(avg_val)
        print(f"Epoch {epoch+1}/{epochs} -> Sup Loss: {avg_sup:.4f}, Unsup Loss: {avg_unsup:.4f}, Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            print("  -> Val loss improved. Saving model.")
            best_val_loss = avg_val
            base_model_to_save.set_weights(model.get_weights())
            base_model_to_save.save(filepath_h5, include_optimizer=False)
    # --- END: Main Training ---
    
    # Benchmarking
    best_model = tf.keras.models.load_model(filepath_h5, compile=False)
    
    val_pred_dir = os.path.join(dataset_base_dir, 'val_predictions', approach, modelname)
    test_pred_dir = os.path.join(dataset_base_dir, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(dataset_base_dir, 'train_unlabeled_predictions', approach, modelname)
    
    mPA_val, mIoU_val = benchmark_multiclass(best_model, val_images_dir, val_masks_dir, val_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    mPA_test, mIoU_test = benchmark_multiclass(best_model, test_images_dir, test_masks_dir, test_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    mPA_unl, mIoU_unl = benchmark_multiclass(best_model, train_unlabeled_images_dir, train_unlabeled_masks_dir, unlabeled_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    
    return mPA_val, mPA_test, mPA_unl, mIoU_val, mIoU_test, mIoU_unl, history





def train_hela_ups(
    model, base_model_to_save, filepath_h5, modelname,
    labeled_data_np, unlabeled_data_np, val_ds,
    pseudo_labels, selection_mask,
    lambda_u, epochs, optimizer, h, w, c,
    batch_size, channel_weights,
    supervised_warmup_steps=50,
    approach='ups'
):
    """
    Faithful UPS training loop for one student on the HeLa multi-label dataset.
    """
    import paths
    from functions import benchmark_hela
    import tensorflow.keras.backend as K

    w_ch = tf.reshape(tf.constant(channel_weights, tf.float32), [1, 1, 1, 3])

    def supervised_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        bce_per_element = K.binary_crossentropy(y_true, y_pred, from_logits=False)
        weighted_bce = bce_per_element * w_ch
        return tf.reduce_mean(weighted_bce)

    # The rest of the function remains the same as the previous correct version...
    unl_ds_with_pseudo = tf.data.Dataset.from_tensor_slices(
        (unlabeled_data_np, pseudo_labels, selection_mask)
    ).shuffle(len(unlabeled_data_np)).batch(batch_size)
    
    lab_ds = tf.data.Dataset.from_tensor_slices(labeled_data_np).shuffle(len(labeled_data_np[0])).batch(batch_size)
    train_ds = tf.data.Dataset.zip((lab_ds.repeat(), unl_ds_with_pseudo.repeat()))

    @tf.function
    def supervised_only_step(data):
        x_l, y_l = data
        with tf.GradientTape() as tape:
            p_l = model(x_l, training=True)
            loss_l = supervised_loss(y_l, p_l)
            scaled_loss = optimizer.get_scaled_loss(loss_l) if hasattr(optimizer, 'get_scaled_loss') else loss_l
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        if hasattr(optimizer, 'get_unscaled_gradients'): grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_l

    @tf.function
    def combined_step(data):
        (x_l, y_l), (x_u, y_pseudo, g_mask) = data
        with tf.GradientTape() as tape:
            p_l = model(x_l, training=True)
            loss_l = supervised_loss(y_l, p_l)
            
            p_u = model(x_u, training=True)
            loss_u = _ups_unlabeled_loss_multilabel(y_pseudo, p_u, g_mask, w_ch)
            
            total_loss = loss_l + lambda_u * loss_u
            scaled_loss = optimizer.get_scaled_loss(total_loss) if hasattr(optimizer, 'get_scaled_loss') else total_loss
        
        grads = tape.gradient(scaled_loss, model.trainable_variables)
        if hasattr(optimizer, 'get_unscaled_gradients'): grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_l, loss_u
    
    @tf.function
    def val_step(x, y):
        p = model(x, training=False)
        return supervised_loss(y, p)

    if supervised_warmup_steps > 0:
        print(f"Starting supervised warmup for {supervised_warmup_steps} steps...")
        warmup_losses = []
        for i, (x_l, y_l) in enumerate(lab_ds.repeat()):
            if i >= supervised_warmup_steps: break
            loss = supervised_only_step((x_l, y_l))
            warmup_losses.append(loss.numpy())
        print(f"Warmup complete. Average supervised loss: {np.mean(warmup_losses):.4f}")

    best_val_loss = float('inf')
    history = {'loss_sup': [], 'loss_unsup': [], 'val_loss': []}
    num_steps_per_epoch = len(unlabeled_data_np) // batch_size
    
    for epoch in range(epochs):
        sup_losses, unsup_losses = [], []
        for data in train_ds.take(num_steps_per_epoch):
            l, u = combined_step(data)
            sup_losses.append(l.numpy()); unsup_losses.append(u.numpy())
        
        val_losses = []
        for x_v, y_v in val_ds:
            val_losses.append(val_step(x_v, y_v).numpy())
        
        avg_sup=np.mean(sup_losses); avg_unsup=np.mean(unsup_losses); avg_val=np.mean(val_losses)
        history['loss_sup'].append(avg_sup); history['loss_unsup'].append(avg_unsup); history['val_loss'].append(avg_val)
        print(f"Epoch {epoch+1}/{epochs} -> Sup Loss: {avg_sup:.4f}, Unsup Loss: {avg_unsup:.4f}, Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            print("  -> Val loss improved. Saving model.")
            best_val_loss = avg_val
            base_model_to_save.set_weights(model.get_weights())
            base_model_to_save.save(filepath_h5, include_optimizer=False)

    best_model = tf.keras.models.load_model(filepath_h5, compile=False)
    
    val_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'val_predictions', approach, modelname)
    test_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)
    
    mIoU_val, mIoU_ad_val, mcce_val = benchmark_hela(best_model, paths.HELA_VAL_DIR, val_pred_dir, h, w, c)
    mIoU_test, mIoU_ad_test, mcce_test = benchmark_hela(best_model, paths.HELA_TEST_DIR, test_pred_dir, h, w, c)
    mIoU_unl, mIoU_ad_unl, mcce_unl = benchmark_hela(best_model, paths.HELA_TRAIN_UNLABELED_DIR, unlabeled_pred_dir, h, w, c)

    return (mIoU_val, mIoU_test, mIoU_unl,
            mIoU_ad_val, mIoU_ad_test, mIoU_ad_unl,
            mcce_val, mcce_test, mcce_unl, history)



##############################################################################################################################################################################################
###                         U2PL — Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels
##############################################################################################################################################################################################



# =========================
# U2PL utilities (updated)
# =========================
import math, os, numpy as np, tensorflow as tf
from collections import deque  # no longer used for banks, retained for compatibility

# ---------- light, safe augs ----------
@tf.function(reduce_retracing=True)
def u2pl_weak_photometric_batch(x: tf.Tensor):
    """Weak color jitter for RGB inputs: x in [0,255]."""
    x = tf.cast(x, tf.float32)
    alpha = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.95, 1.05, dtype=x.dtype)
    beta  = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -10.0, 10.0, dtype=x.dtype)
    return tf.clip_by_value(x * alpha + beta, 0., 255.)

weak_photometric_batch = u2pl_weak_photometric_batch



# ---------- numerically-stable entropy ----------
def _bernoulli_entropy_sum(p_sigmoid: tf.Tensor, eps=1e-6) -> tf.Tensor:
    """
    Stable Bernoulli entropy for multi-label probs.
    Args:
      p_sigmoid: [B,H,W,C] in [0,1]
    Returns:
      [B,H,W] sum over channels
    """
    p = tf.cast(p_sigmoid, tf.float32)
    p = tf.clip_by_value(p, eps, 1.0 - eps)
    h = -(p * tf.math.log(p) + (1.0 - p) * tf.math.log1p(-p))
    return tf.reduce_sum(h, axis=-1)

# ---------- mask combo ids ----------
def _mask_to_combo_id(binary_mask: tf.Tensor) -> tf.Tensor:
    C = tf.shape(binary_mask)[-1]
    powers_of_2 = 2 ** tf.range(C, dtype=tf.int32)               # [C]
    powers_of_2 = tf.reshape(powers_of_2, [1, 1, 1, C])          # [1,1,1,C]
    ids = tf.reduce_sum(tf.cast(binary_mask, tf.int32) * powers_of_2, axis=-1)
    return ids  # [B,H,W]

# ---------- math helpers ----------
def _softmax(logits): return tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1)

def _entropy_from_probs(p: tf.Tensor, eps=1e-6) -> tf.Tensor:
    p = tf.cast(p, tf.float32)
    p = tf.clip_by_value(p, eps, 1.0)
    return -tf.reduce_sum(p * tf.math.log(p), axis=-1)  # [B,H,W]

def _fast_quantile(x: tf.Tensor, q: tf.Tensor) -> tf.Tensor:
    """
    Fast per-batch quantile via top-k (no full sort).
    x: [B,...] -> flattened to [B,N]
    q in [0,1], returns [B] threshold such that ~q of values are <= gamma.
    """
    x_flat = tf.reshape(x, [tf.shape(x)[0], -1])  # [B,N]
    N = tf.shape(x_flat)[1]
    # k items to keep from the top end: k = ceil((1-q)*N)
    k = tf.cast(tf.math.ceil((1.0 - q) * tf.cast(N, tf.float32)), tf.int32)
    k = tf.maximum(k, 1)
    topk = tf.math.top_k(x_flat, k=k, sorted=False).values       # [B,k]
    gamma = tf.reduce_min(topk, axis=-1)                         # [B]
    return gamma

def _quantile(x: tf.Tensor, q: tf.Tensor):
    # drop-in wrapper that now uses the fast top-k method
    return _fast_quantile(x, q)

def _rank_descending(p: tf.Tensor) -> tf.Tensor:
    neg = -p
    idx1 = tf.argsort(neg, axis=-1, stable=True)
    inv  = tf.argsort(idx1, axis=-1, stable=True)
    return tf.cast(inv + 1, tf.int32)  # [B,H,W,K]

# ---------- split model ----------
def _build_feature_and_tail_models(model: tf.keras.Model):
    try:
        K = int(model.output_shape[-1])
    except Exception:
        return None, None
    last_idx = None
    for idx in range(len(model.layers) - 1, -1, -1):
        L = model.layers[idx]
        if isinstance(L, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D, tf.keras.layers.Conv2DTranspose)):
            try:
                if getattr(L, "filters", None) == K and len(L.output_shape) == 4:
                    last_idx = idx; break
            except Exception:
                continue
    if last_idx is None:
        try:
            last_idx = len(model.layers) - 2
            _ = model.layers[last_idx].output_shape
        except Exception:
            return None, None

    feat_tensor = model.layers[last_idx].input
    feature_model = tf.keras.Model(inputs=model.inputs, outputs=feat_tensor, name="u2pl_features")

    tail_in = tf.keras.Input(shape=feat_tensor.shape[1:], name="u2pl_tail_in")
    x = tail_in
    for L in model.layers[last_idx:]:
        x = L(x)
    tail_model = tf.keras.Model(inputs=tail_in, outputs=x, name="u2pl_classifier_tail")
    return feature_model, tail_model

def _build_projection_head(feat_shape: tf.TensorShape, emb_dim: int = 256) -> tf.keras.Model:
    inp = tf.keras.Input(shape=feat_shape[1:], name="u2pl_proj_in")
    intermediate_dim = int(feat_shape[-1]) // 2
    x = tf.keras.layers.Conv2D(intermediate_dim, 3, padding='same', use_bias=False, name="u2pl_proj_conv1")(inp)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="u2pl_proj_bn1")(x)
    x = tf.keras.layers.ReLU(name="u2pl_proj_relu1")(x)
    x = tf.keras.layers.Conv2D(emb_dim, 3, padding='same', use_bias=False, name="u2pl_proj_conv2")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="u2pl_proj_bn2")(x)
    x = tf.keras.layers.ReLU(name="u2pl_proj_relu2")(x)
    return tf.keras.Model(inputs=inp, outputs=x, name="u2pl_proj")

def _clone_model_weights(src: tf.keras.Model) -> tf.keras.Model:
    clone = tf.keras.models.clone_model(src)
    _ = clone([tf.zeros_like(t) for t in src.inputs] if isinstance(src.inputs, (list,tuple)) else tf.zeros_like(src.input))
    clone.set_weights(src.get_weights())
    return clone

@tf.function(reduce_retracing=True)
def _ema_update_models(teacher_models, student_models, momentum: float):
    m = tf.convert_to_tensor(momentum, tf.float32)
    for tm, sm in zip(teacher_models, student_models):
        for tw, sw in zip(tm.trainable_variables, sm.trainable_variables):
            tw.assign(m * tw + (1. - m) * tf.cast(sw, tw.dtype))

# ---------- InfoNCE on class prototype ----------
def _l2n(x, eps=1e-12):
    n = tf.norm(x, axis=-1, keepdims=True)
    return x / tf.maximum(n, eps)

def _info_nce_proto(anchors: tf.Tensor, proto_pos: tf.Tensor, negs: tf.Tensor, temperature: float = 0.07):
    a = _l2n(anchors)                           # [M,E]
    p = _l2n(tf.expand_dims(proto_pos, 0))      # [1,E]
    n = _l2n(negs)                               # [N,E]
    sim_pos = tf.squeeze(tf.matmul(a, p, transpose_b=True), axis=-1) / temperature     # [M]
    sim_neg = tf.matmul(a, n, transpose_b=True) / temperature                           # [M,N]
    log_den = tf.math.reduce_logsumexp(tf.concat([tf.expand_dims(sim_pos, -1), sim_neg], axis=-1), axis=-1)  # [M]
    loss = -tf.reduce_mean(sim_pos - log_den)
    return loss


# ---------- GPU-resident memory bank (no retracing) ----------
class TFBank(tf.Module):
    """
    Simple ring-buffer memory bank kept on device.
    Stores [capacity, E] float32 features.
    """
    def __init__(self, capacity: int, emb_dim: int, name=None):
        super().__init__(name=name)
        self.capacity = tf.constant(int(capacity), dtype=tf.int32)
        self.emb_dim  = int(emb_dim)

        self.buf  = tf.Variable(
            tf.zeros([capacity, emb_dim], dtype=tf.float32),
            trainable=False,
            name=(f"{name}_buf" if name else None),
        )
        self.size = tf.Variable(0, dtype=tf.int32, trainable=False,
                                name=(f"{name}_size" if name else None))
        self.ptr  = tf.Variable(0, dtype=tf.int32, trainable=False,
                                name=(f"{name}_ptr" if name else None))

        # Wrap implementations with fixed input signatures to avoid retracing.
        self._push_fn = tf.function(
            self._push_impl,
            input_signature=[tf.TensorSpec(shape=[None, self.emb_dim], dtype=tf.float32)],
            experimental_relax_shapes=True,
        )
        self._sample_fn = tf.function(
            self._sample_impl,
            input_signature=[tf.TensorSpec(shape=[], dtype=tf.int32)],
            experimental_relax_shapes=True,
        )

    # Public APIs stay the same, we just ensure Tensor args.
    def push(self, feats: tf.Tensor):
        feats = tf.cast(feats, tf.float32)
        return self._push_fn(feats)

    def sample(self, N):
        if not isinstance(N, tf.Tensor):
            N = tf.convert_to_tensor(N, dtype=tf.int32)  # <-- important
        else:
            N = tf.cast(N, tf.int32)
        return self._sample_fn(N)

    # ---- graph implementations ----
    def _push_impl(self, feats: tf.Tensor):
        # feats: [N, emb_dim]
        n   = tf.shape(feats)[0]
        cap = self.capacity
        idx = tf.math.mod(tf.range(self.ptr, self.ptr + n), cap)              # [N]
        updated = tf.tensor_scatter_nd_update(self.buf, tf.expand_dims(idx, 1), feats)
        self.buf.assign(updated)
        self.ptr.assign(tf.math.mod(self.ptr + n, cap))
        self.size.assign(tf.minimum(self.size + n, cap))

    def _sample_impl(self, N: tf.Tensor):
        # N: scalar int32
        m = tf.minimum(N, self.size)
        def _sample():
            idx = tf.random.uniform([m], 0, self.size, dtype=tf.int32)
            return tf.gather(self.buf, idx)
        def _empty():
            return tf.zeros([0, self.emb_dim], tf.float32)
        return tf.cond(m > 0, _sample, _empty)



def train_isic_2018_u2pl(
    model,                 # student (logits, 2 classes)
    filepath_h5,
    modelname,
    h, w, c,
    # U2PL hypers
    alpha0=0.2,
    lambda_u_eta=1.0,
    temperature=0.5,
    prt_rl=3, prt_rh=20,
    delta_p=0.3,
    bank_capacity_per_class=4096,
    M_anchors_per_class=50,
    N_negs_per_class=256,
    ema_momentum=0.996,
    lambda_c=0.1,
    # General
    epochs=120,
    batch_size=8,
    optimizer=None,
    dataset_name_for_aug='ISIC_2018',
    approach='u2pl'
):
    """Faithful U2PL trainer for ISIC 2018 (2-class softmax) with compiled train_step."""
    from functions import load_labeled_data_single_mask, load_unlabeled_images, benchmark_ISIC2018, _build_optimizer_weights
    import paths
    import numpy as np, math, os, tensorflow as tf

    if optimizer is None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    # --- Data ---
    lab_imgs_np, lab_mask_np = load_labeled_data_single_mask(paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR,
                                                             paths.ISIC_2018_TRAIN_LABELED_MASKS_DIR)
    unl_imgs_np = load_unlabeled_images(paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR)
    val_imgs_np, val_mask_np = load_labeled_data_single_mask(paths.ISIC_2018_VAL_IMAGES_DIR,
                                                             paths.ISIC_2018_VAL_MASKS_DIR)

    lab_imgs  = tf.convert_to_tensor(lab_imgs_np,  tf.float32)    # [N,H,W,C]
    lab_ids   = tf.convert_to_tensor(lab_mask_np,  tf.int32)      # [N,H,W]
    unl_imgs  = tf.convert_to_tensor(unl_imgs_np,  tf.float32)    # [M,H,W,C]
    val_imgs  = tf.convert_to_tensor(val_imgs_np,  tf.float32)    # [V,H,W,C]
    val_ids   = tf.convert_to_tensor(val_mask_np,  tf.int32)      # [V,H,W]
    num_classes = 2

    # ---------- losses ----------
    ce_sup_logits = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    def supervised_ce(y_ids, logits):
        """
        Final, numerically stable supervised loss combining CE and Dice, 
        with robust label resizing and manual log-softmax for stability.
        """
        # --- 1. Canonicalize y_ids to [B, H, W] ---
        y = tf.cast(y_ids, tf.int32)
        
        # This part is now robust and should be correct
        if y.shape.rank == 4:
            shape = tf.shape(y)
            if shape[1] <= num_classes and shape[1] < shape[2]: 
                y = tf.transpose(y, [0, 2, 3, 1])
            if y.shape[-1] == 1:
                y = tf.squeeze(y, axis=-1)
            else:
                y = tf.argmax(y, axis=-1, output_type=tf.int32)
        y_ids_matched = y

        # --- 2. Resize Labels to Match Logits ---
        logits_shape = tf.shape(logits)
        h_l, w_l = logits_shape[1], logits_shape[2]
        
        y_shape = tf.shape(y_ids_matched)
        h_y, w_y = y_shape[1], y_shape[2]

        needs_resize = tf.logical_or(tf.not_equal(h_y, h_l), tf.not_equal(w_y, w_l))
        
        def resize_labels():
            y_4d = tf.expand_dims(y_ids_matched, axis=-1)
            y_resized_4d = tf.image.resize(y_4d, [h_l, w_l], method='nearest')
            return tf.squeeze(y_resized_4d, axis=-1)

        def do_nothing():
            return y_ids_matched

        y_ids_matched = tf.cond(needs_resize, resize_labels, do_nothing)
        
        # --- 3. Calculate Losses with Maximum Stability ---
        logits = tf.cast(logits, tf.float32)
        K = logits_shape[-1]
        
        # -- MANUAL CROSS-ENTROPY --
        # This is the core of the fix. It's more stable than the Keras layer.
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        y_true_one_hot = tf.one_hot(y_ids_matched, K, dtype=tf.float32)
        loss_ce_per_pixel = -tf.reduce_sum(y_true_one_hot * log_probs, axis=-1)
        loss_ce = tf.reduce_mean(loss_ce_per_pixel)

        # -- DICE LOSS --
        y_pred_probs = tf.nn.softmax(logits, axis=-1)
        
        y_true_flat = tf.reshape(y_true_one_hot, [-1, K])
        y_pred_flat = tf.reshape(y_pred_probs, [-1, K])
        
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
        denominator = tf.reduce_sum(y_true_flat, axis=0) + tf.reduce_sum(y_pred_flat, axis=0)
        
        dice_per_class = (2. * intersection + 1e-6) / (denominator + 1e-6)
        loss_dice = 1.0 - tf.reduce_mean(dice_per_class)

        # Combine losses
        return 0.5 * loss_ce + 0.5 * loss_dice


    # --- build ---
    _ = model(tf.zeros([1, h, w, c], tf.float32), training=False)
    _build_optimizer_weights(optimizer, model.trainable_variables)
    feat_s, tail_s = _build_feature_and_tail_models(model)
    if feat_s is None or tail_s is None: raise RuntimeError("U2PL: cannot split model.")
    proj_s = _build_projection_head(feat_s.outputs[0].shape)
    teacher_model = _clone_model_weights(model); teacher_model.trainable = False
    teacher_feat, teacher_tail = _build_feature_and_tail_models(teacher_model)
    teacher_proj = _clone_model_weights(proj_s); teacher_proj.trainable = False
    emb_dim = int(proj_s.output_shape[-1])

    banks = [TFBank(bank_capacity_per_class, emb_dim, name=f"bank_cls_{i}") for i in range(num_classes)]

    # --- Forwards ---
    def student_forward(x, training):
        f = tf.cast(feat_s(x, training=training), tf.float32)
        logits = tf.cast(tail_s(f, training=training), tf.float32)
        reps   = tf.cast(proj_s(f, training=training), tf.float32)
        return logits, reps
    def teacher_forward(x):
        f = tf.cast(teacher_feat(x, training=False), tf.float32)
        logits = tf.cast(teacher_tail(f, training=False), tf.float32)
        reps   = tf.cast(teacher_proj(f, training=False), tf.float32)
        return logits, reps

    # --- Losses (unchanged paradigm) ---
    def unsup_ce(student_logits, teacher_probs, mask01):
        ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')(teacher_probs, student_logits)  # keep as-is
        ce = ce * tf.squeeze(mask01, -1)
        num = tf.maximum(tf.reduce_sum(mask01, axis=[1,2,3]), 1.0)
        s = tf.reduce_sum(ce, axis=[1,2]); return tf.reduce_mean(s / num)

    # --- Schedule & metrics ---
    Nl = int(lab_imgs.shape[0]); Nu = int(unl_imgs.shape[0])
    steps_per_epoch = max(int(math.ceil(Nl/batch_size)), int(math.ceil(Nu/batch_size)))
    T_total = tf.constant(float(epochs * steps_per_epoch), tf.float32)
    mx = tf.keras.metrics.Mean(); mu = tf.keras.metrics.Mean(); mc = tf.keras.metrics.Mean(); mhe = tf.keras.metrics.Mean()

    # ---- small helpers ----
    def _flatten_hw(x): return tf.reshape(x, [-1, tf.shape(x)[-1]])
    num_classes_t = tf.constant(num_classes, tf.int32)
    N_negs_t = tf.constant(N_negs_per_class, tf.int32)

    # graph-safe bank access
    def _bank_push(idx_i: tf.Tensor, feats: tf.Tensor):
        idx_i = tf.cast(idx_i, tf.int32)
        def _mk_push(k):
            return (lambda: (banks[k].push(feats), tf.constant(0, tf.int32))[1])
        return tf.switch_case(idx_i, branch_fns=[_mk_push(k) for k in range(num_classes)])

    def _bank_sample(idx_i: tf.Tensor, N: tf.Tensor) -> tf.Tensor:
        idx_i = tf.cast(idx_i, tf.int32)
        def _mk_sample(k):
            return (lambda: banks[k].sample(N))
        return tf.switch_case(idx_i, branch_fns=[_mk_sample(k) for k in range(num_classes)])

    @tf.function(experimental_relax_shapes=True)
    def train_step(xLw, yL_ids, xUw, step_idx_f: tf.Tensor):
        # pre passes
        logits_L_s_pre, reps_L_s_pre = student_forward(xLw, training=False)
        logits_U_s_pre, reps_U_s_pre = student_forward(xUw, training=False)
        logits_L_t, reps_L_t = teacher_forward(xLw); logits_U_t, reps_U_t = teacher_forward(xUw)
        probs_L_t = _softmax(logits_L_t); probs_U_t = _softmax(logits_U_t)

        Lx = supervised_ce(yL_ids, logits_L_s_pre)

        # reliability mask
        H_u = _entropy_from_probs(probs_U_t)          # [B,H,W]
        mean_entropy = tf.reduce_mean(H_u)
        a_t = alpha0 * (1.0 - (step_idx_f / tf.maximum(T_total, 1.0)))
        gamma = _quantile(H_u, 1.0 - a_t)             # [B]
        reliable = H_u <= gamma[:, None, None]
        m_rel = tf.cast(reliable, tf.float32)[..., None]

        yhat_t = tf.argmax(probs_U_t, axis=-1, output_type=tf.int32)
        Lu = unsup_ce(logits_U_s_pre, tf.one_hot(yhat_t, num_classes_t, dtype=tf.float32), m_rel)
        lam_u = lambda_u_eta * tf.cast(tf.size(H_u), tf.float32) / tf.maximum(tf.reduce_sum(m_rel), 1e-8)

        # flatten
        reps_L_s_pre_f, reps_U_s_pre_f = _flatten_hw(reps_L_s_pre), _flatten_hw(reps_U_s_pre)
        reps_L_t_f, reps_U_t_f         = _flatten_hw(reps_L_t),     _flatten_hw(reps_U_t)
        pL_f, pU_f                     = _flatten_hw(probs_L_t),    _flatten_hw(probs_U_t)
        ranks_L = _rank_descending(probs_L_t); ranks_U = _rank_descending(probs_U_t)
        ranks_L_f, ranks_U_f           = _flatten_hw(tf.cast(ranks_L, tf.float32)), _flatten_hw(tf.cast(ranks_U, tf.float32))
        yL_f = tf.reshape(yL_ids, [-1])
        NpixL = tf.cast(tf.size(yL_f), tf.int32)
        reliable_flat = tf.reshape(reliable, [-1])

        # containers
        ta_idx   = tf.TensorArray(tf.int32,   size=0, dynamic_size=True, infer_shape=False)
        ta_proto = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
        ta_negs  = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
        classes_used = tf.constant(0, tf.int32)
        Lc_total = tf.constant(0.0, tf.float32)

        i0 = tf.constant(0, tf.int32)
        def cond_c(ci, *_): return tf.less(ci, num_classes_t)

        def body_c(ci, classes_used, ta_idx, ta_proto, ta_negs, Lc_total):
            # labeled in class
            idx_Al = tf.cond(
                NpixL > 0,
                lambda: tf.cast(tf.where(tf.equal(yL_f, ci))[:,0], tf.int32),
                lambda: tf.zeros([0], tf.int32)
            )
            # high prob labeled
            def filter_highprob_l():
                if tf.executing_eagerly():
                    pass
                p_in = tf.gather(pL_f[:, ci], idx_Al)
                keep = tf.where(p_in > delta_p)[:,0]
                return tf.gather(idx_Al, keep)
            idx_Al = tf.cond(tf.shape(idx_Al)[0] > 0, filter_highprob_l, lambda: idx_Al)

            # unlabeled reliable class ci with high prob
            yhat_U_f = tf.reshape(yhat_t, [-1])
            in_rel_ci = tf.logical_and(tf.equal(yhat_U_f, ci), reliable_flat)
            idx_Au = tf.cast(tf.where(in_rel_ci)[:,0], tf.int32)
            def filter_highprob_u():
                p_in = tf.gather(pU_f[:, ci], idx_Au)
                keep = tf.where(p_in > delta_p)[:,0]
                return tf.gather(idx_Au, keep)
            idx_Au = tf.cond(tf.shape(idx_Au)[0] > 0, filter_highprob_u, lambda: idx_Au)

            idx_all_t = tf.concat([idx_Al, NpixL + idx_Au], axis=0)
            has_any = tf.greater(tf.shape(idx_all_t)[0], 0)

            def do_class():
                # anchors
                idx_A_sampled = tf.random.shuffle(idx_all_t)[:M_anchors_per_class]
                reps_concat_pre = tf.concat([reps_L_s_pre_f, reps_U_s_pre_f], axis=0)
                anchors_pre = tf.gather(reps_concat_pre, idx_A_sampled)

                # prototype from teacher
                reps_concat_t = tf.concat([reps_L_t_f, reps_U_t_f], axis=0)
                proto_pos = tf.reduce_mean(tf.gather(reps_concat_t, idx_all_t), axis=0)

                # build current_negs
                current_negs = tf.zeros([0, emb_dim], tf.float32)

                # labeled not-ci with rank<=prt_rl
                not_c = tf.cast(tf.where(tf.not_equal(yL_f, ci))[:,0], tf.int32)
                def add_not_c():
                    r_c_L = tf.gather(ranks_L_f[:, ci], not_c)
                    seln = tf.where(tf.cast(r_c_L, tf.int32) <= prt_rl)[:,0]
                    idx_Nl = tf.gather(not_c, seln)
                    feats = tf.gather(reps_L_t_f, idx_Nl)
                    return tf.concat([current_negs, feats], axis=0)
                current_negs1 = tf.cond(tf.shape(not_c)[0] > 0, add_not_c, lambda: current_negs)

                # unlabeled unreliable with prt_rl <= rank < prt_rh
                unreli_idx = tf.where(tf.logical_not(reliable_flat))[:,0]
                def add_unreli():
                    r_c_U = tf.gather(ranks_U_f[:, ci], unreli_idx)
                    cond = tf.logical_and(tf.cast(r_c_U, tf.int32) >= prt_rl,
                                          tf.cast(r_c_U, tf.int32) <  prt_rh)
                    idx_Nu = tf.boolean_mask(unreli_idx, cond)
                    feats = tf.gather(reps_U_t_f, idx_Nu)
                    return tf.concat([current_negs1, feats], axis=0)
                current_negs2 = tf.cond(tf.shape(unreli_idx)[0] > 0, add_unreli, lambda: current_negs1)

                # push + sample from bank
                def do_push(): return ( _bank_push(ci, current_negs2), tf.constant(0, tf.int32) )[1]
                _ = tf.cond(tf.greater(tf.shape(current_negs2)[0], 0), do_push, lambda: tf.constant(0, tf.int32))

                sampled_negs = _bank_sample(ci, N_negs_t)
                has_sampled = tf.greater(tf.shape(sampled_negs)[0], 0)

                # fallback to current_negs if bank empty
                sampled_final = tf.cond(has_sampled, lambda: sampled_negs, lambda: current_negs2)
                has_final = tf.greater(tf.shape(sampled_final)[0], 0)

                def take_it():
                    Lc_c = _info_nce_proto(anchors_pre, tf.stop_gradient(proto_pos), tf.stop_gradient(sampled_final), temperature)
                    ta_idx2   = ta_idx.write(classes_used, idx_A_sampled)
                    ta_proto2 = ta_proto.write(classes_used, tf.stop_gradient(proto_pos))
                    ta_negs2  = ta_negs.write(classes_used, tf.stop_gradient(sampled_final))
                    return (classes_used + 1, ta_idx2, ta_proto2, ta_negs2, Lc_total + Lc_c)
                def skip_it():
                    return (classes_used, ta_idx, ta_proto, ta_negs, Lc_total)

                return tf.cond(has_final, take_it, skip_it)

            def no_class():
                return (classes_used, ta_idx, ta_proto, ta_negs, Lc_total)

            classes_used2, ta_idx2, ta_proto2, ta_negs2, Lc_total2 = tf.cond(has_any, do_class, no_class)
            return ci + 1, classes_used2, ta_idx2, ta_proto2, ta_negs2, Lc_total2

        _, classes_used, ta_idx, ta_proto, ta_negs, Lc_total = tf.while_loop(
            cond_c, body_c,
            loop_vars=[i0, classes_used, ta_idx, ta_proto, ta_negs, Lc_total],
            parallel_iterations=1
        )

        Lc_mean_pre = tf.math.divide_no_nan(Lc_total, tf.cast(tf.maximum(classes_used, 1), tf.float32))

        # grad pass
        with tf.GradientTape() as tape:
            logits_L_s_grad, reps_L_s_grad = student_forward(xLw, training=True)
            logits_U_s_grad, reps_U_s_grad = student_forward(xUw, training=True)
            loss_Lx = supervised_ce(yL_ids, logits_L_s_grad)
            loss_Lu = unsup_ce(logits_U_s_grad, tf.one_hot(yhat_t, num_classes_t, dtype=tf.float32), m_rel)

            reps_s_grad_flat = tf.concat([
                tf.reshape(reps_L_s_grad, [-1, tf.shape(reps_L_s_grad)[-1]]),
                tf.reshape(reps_U_s_grad, [-1, tf.shape(reps_U_s_grad)[-1]])
            ], axis=0)

            def compute_contrastive_loss():
                j0 = tf.constant(0, tf.int32)
                Lc_total_grad_inner = tf.constant(0.0, tf.float32)
                def cond_j(j, Lc_acc): return tf.less(j, classes_used)
                def body_j(j, Lc_acc):
                    idx_j   = ta_idx.read(j)
                    proto_j = ta_proto.read(j)
                    negs_j  = ta_negs.read(j)
                    anchors_live = tf.gather(reps_s_grad_flat, idx_j)
                    Lc_c_grad = _info_nce_proto(anchors_live, proto_j, negs_j, temperature)
                    return j + 1, Lc_acc + Lc_c_grad
                _, Lc_total_grad_inner = tf.while_loop(cond_j, body_j, [j0, Lc_total_grad_inner], parallel_iterations=1)
                return tf.math.divide_no_nan(Lc_total_grad_inner, tf.cast(tf.maximum(classes_used, 1), tf.float32))

            Lc_mean_grad = tf.cond(tf.greater(classes_used, 0), compute_contrastive_loss, lambda: tf.constant(0.0, tf.float32))

            total_loss = loss_Lx + lam_u * loss_Lu + lambda_c * Lc_mean_grad
            scaled_loss = optimizer.get_scaled_loss(total_loss) if hasattr(optimizer, "get_scaled_loss") else total_loss

        vars_to_train = model.trainable_variables + proj_s.trainable_variables
        grads = tape.gradient(scaled_loss, vars_to_train)
        if hasattr(optimizer, "get_unscaled_gradients"): grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, vars_to_train))

        _ema_update_models([teacher_model, teacher_proj], [model, proj_s], float(ema_momentum))

        return Lx, Lu, Lc_mean_pre, lam_u, mean_entropy

    @tf.function(reduce_retracing=True)
    def _val_step(x, y):
        logits_v_s, _ = student_forward(x, training=False)
        return supervised_ce(y, logits_v_s)

    # --- Loop ---
    history = {'val_loss': [], 'alpha': [], 'avg_entropy': []}
    best_val = float('inf'); step_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        idxL = np.random.permutation(Nl); idxU = np.random.permutation(Nu)
        mx.reset_state(); mu.reset_state(); mc.reset_state(); mhe.reset_state()

        for step in range(steps_per_epoch):
            step_counter += 1
            sL = (step % int(math.ceil(Nl/batch_size))) * batch_size; sU = (step % int(math.ceil(Nu/batch_size))) * batch_size
            bL = idxL[sL:sL+batch_size]; bU = idxU[sU:sU+batch_size]
            xL = tf.gather(lab_imgs, bL); yL = tf.gather(lab_ids,  bL); xU = tf.gather(unl_imgs,  bU)
            (xL, yL) = random_hflip_batch(xL, yL); (xU, _) = random_hflip_batch(xU, None)
            xLw = weak_photometric_batch(xL); xUw = weak_photometric_batch(xU)

            Lx, Lu, Lc_mean, lam_u, mean_H = train_step(xLw, yL, xUw, tf.constant(float(step_counter), tf.float32))
            mx.update_state(Lx); mu.update_state(Lu); mc.update_state(Lc_mean); mhe.update_state(mean_H)

            if (step+1) % 20 == 0 or (step+1) == steps_per_epoch:
                print(f"  step {step+1}/{steps_per_epoch} | "
                      f"Lx {mx.result().numpy():.4f} | "
                      f"Lu {mu.result().numpy():.4f} (λu={lam_u.numpy():.2f}) | "
                      f"Lc {mc.result().numpy():.4f} | "
                      f"α_unrel {alpha0 * (1.0 - (float(step_counter)/(epochs*steps_per_epoch))):.3f}")
                mx.reset_state(); mu.reset_state(); mc.reset_state()

        # Validation
        V = int(val_imgs.shape[0]); vsteps = math.ceil(V / batch_size); v_losses = []
        for i in range(vsteps):
            s = i * batch_size; e = min(s + batch_size, V)
            v_losses.append(float(_val_step(val_imgs[s:e], val_ids[s:e]).numpy()))
        vloss = float(np.mean(v_losses)) if v_losses else float('inf')

        avg_entropy_epoch = float(mhe.result().numpy()) if mhe.count.numpy() > 0 else 0.0
        a_t_unrel_epoch = alpha0 * (1.0 - (float(step_counter)/(epochs*steps_per_epoch)))
        history['val_loss'].append(vloss)
        history['alpha'].append(a_t_unrel_epoch)
        history['avg_entropy'].append(avg_entropy_epoch)
        print(f"Validation Loss: {vloss:.4f} | Avg Entropy: {avg_entropy_epoch:.4f} | α_unrel {a_t_unrel_epoch:.3f}")

        if vloss < best_val:
            best_val = vloss
            model.save(filepath_h5, include_optimizer=False, save_format="h5" if filepath_h5.endswith(".h5") else "tf")

    # Benchmark (unchanged)
    best_model = tf.keras.models.load_model(filepath_h5, compile=False)
    inputs = best_model.inputs
    logits = best_model.outputs[0] if isinstance(best_model.outputs, (list, tuple)) else best_model.output
    softmax = tf.keras.layers.Softmax(axis=-1, name="infer_softmax")(tf.cast(logits, tf.float32))
    lesion_prob = tf.keras.layers.Lambda(lambda z: z[..., 1:2], name="lesion_prob")(softmax)
    inference_model = tf.keras.Model(inputs=inputs, outputs=lesion_prob)

    val_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'val_predictions', approach, modelname)
    test_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(paths.ISIC_2018_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)

    mIoU_val, dice_val = benchmark_ISIC2018(inference_model, paths.ISIC_2018_VAL_IMAGES_DIR,  paths.ISIC_2018_VAL_MASKS_DIR,  val_pred_dir,  h, w, c)
    mIoU_tst, dice_tst = benchmark_ISIC2018(inference_model, paths.ISIC_2018_TEST_IMAGES_DIR, paths.ISIC_2018_TEST_MASKS_DIR, test_pred_dir, h, w, c)
    mIoU_unl, dice_unl = benchmark_ISIC2018(inference_model, paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR, paths.ISIC_2018_TRAIN_UNLABELED_MASKS_DIR, unlabeled_pred_dir, h, w, c)

    return mIoU_val, mIoU_tst, mIoU_unl, dice_val, dice_tst, dice_unl, history





def train_hela_u2pl(
    model,
    filepath_h5,
    modelname,
    h, w, c,
    # U2PL hypers (adapted for HeLa)
    alpha0=0.2,
    lambda_u_eta=1.0,
    temperature=0.5,
    bank_capacity_per_class=4096,
    M_anchors_per_class=50,
    N_negs_per_class=256,
    ema_momentum=0.996,
    lambda_c=0.1,
    # General
    epochs=120,
    batch_size=8,
    optimizer=None,
    channel_weights=(1.0, 1.0, 3.0),
    dataset_name_for_aug='HeLa',
    approach='u2pl',
):
    """
    U2PL for HeLa with a single compiled train_step (graph).
    Keeps the original two-pass student flow and contrastive logic.
    """
    from functions import load_images_and_masks_hela, load_unlabeled_images, benchmark_hela, _build_optimizer_weights
    import paths
    import numpy as np
    import math
    tf_dtype_f = tf.float32
    tf_dtype_i = tf.int32

    if optimizer is None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    # --- Data ---
    l_imgs, lA, lD, lP = load_images_and_masks_hela(paths.HELA_TRAIN_LABELED_DIR)
    lab_masks_np = np.stack([lA, lD, lP], axis=-1).astype(np.float32)
    u_imgs_np    = load_unlabeled_images(paths.HELA_TRAIN_UNLABELED_BRIGHTFIELD_DIR, read_as_grayscale=True)
    v_imgs, vA, vD, vP = load_images_and_masks_hela(paths.HELA_VAL_DIR)
    val_masks_np = np.stack([vA, vD, vP], axis=-1).astype(np.float32)

    lab_imgs  = tf.expand_dims(tf.convert_to_tensor(l_imgs,       tf_dtype_f), -1)
    unl_imgs  = tf.expand_dims(tf.convert_to_tensor(u_imgs_np,    tf_dtype_f), -1)
    val_imgs  = tf.expand_dims(tf.convert_to_tensor(v_imgs,       tf_dtype_f), -1)
    lab_masks = tf.convert_to_tensor(lab_masks_np, tf_dtype_f)
    val_masks = tf.convert_to_tensor(val_masks_np, tf_dtype_f)

    num_combinations = 8

    # --- Model split ---
    _ = model(tf.zeros([1, h, w, c], tf_dtype_f), training=False); _build_optimizer_weights(optimizer, model.trainable_variables)
    feat_s, tail_s = _build_feature_and_tail_models(model)
    if feat_s is None or tail_s is None: raise RuntimeError("U2PL: cannot split HeLa model.")
    proj_s = _build_projection_head(feat_s.outputs[0].shape)
    teacher_model = _clone_model_weights(model); teacher_model.trainable = False
    teacher_feat, teacher_tail = _build_feature_and_tail_models(teacher_model)
    teacher_proj = _clone_model_weights(proj_s); teacher_proj.trainable = False
    emb_dim = int(proj_s.output_shape[-1])

    # --- Banks (GPU resident, fixed signatures; from earlier patch) ---
    banks = [TFBank(bank_capacity_per_class, emb_dim, name=f"bank_combo_{i}") for i in range(num_combinations)]

    # --- Forwards ---
    def student_forward(x, training):
        f = tf.cast(feat_s(x, training=training), tf_dtype_f)
        probs = tf.cast(tail_s(f, training=training), tf_dtype_f)  # expects sigmoid outputs
        reps  = tf.cast(proj_s(f, training=training), tf_dtype_f)
        return probs, reps
    def teacher_forward(x):
        f = tf.cast(teacher_feat(x, training=False), tf_dtype_f)
        probs = tf.cast(teacher_tail(f, training=False), tf_dtype_f)
        reps  = tf.cast(teacher_proj(f, training=False), tf_dtype_f)
        return probs, reps

    # --- Sigmoid BCE (per-pixel, per-channel) ---
    w_ch = tf.reshape(tf.constant(channel_weights, tf_dtype_f), [1,1,1,3])
    def _bce_ppc(y_true, y_pred):
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf_dtype_f), 1e-6, 1. - 1e-6)
        y_true = tf.cast(y_true, tf_dtype_f)
        return -(y_true * tf.math.log(y_pred) + (1. - y_true) * tf.math.log1p(-y_pred))  # [B,H,W,3]
    def supervised_bce(y_true, y_pred):
        return tf.reduce_mean(_bce_ppc(y_true, y_pred) * w_ch)
    def unsup_bce(student_probs, teacher_probs, mask01):
        loss_map = _bce_ppc(teacher_probs, student_probs) * w_ch  # [B,H,W,3]
        masked   = loss_map * mask01                               # mask01 [B,H,W,1]
        denom = tf.maximum(tf.reduce_sum(mask01) * tf.cast(tf.shape(loss_map)[-1], tf_dtype_f), 1.0)
        return tf.reduce_sum(masked) / denom

    # --- Schedule ---
    Nl = int(lab_imgs.shape[0]); Nu = int(unl_imgs.shape[0])
    steps_per_epoch = max(int(math.ceil(Nl/batch_size)), int(math.ceil(Nu/batch_size)))
    T_total = tf.constant(float(epochs * steps_per_epoch), tf_dtype_f)

    # --- Metrics (device-friendly) ---
    mx = tf.keras.metrics.Mean(); mu = tf.keras.metrics.Mean(); mc = tf.keras.metrics.Mean(); mhe = tf.keras.metrics.Mean()

    # ===========================
    # Compiled training step
    # ===========================
    @tf.function(experimental_relax_shapes=True)
    def train_step(xLw, yL, xUw, step_idx_f: tf.Tensor):
        """
        xLw: [B,H,W,1]  yL: [B,H,W,3]  xUw: [B,H,W,1]  step_idx_f: scalar float32
        Returns: Lx, Lu, Lc_mean_pre, lam_u, mean_entropy
        """
    
        # ---- helpers to access banks inside a graph (no Python indexing with tensors) ----
        def _bank_push(idx_i: tf.Tensor, feats: tf.Tensor):
            idx_i = tf.cast(idx_i, tf.int32)
    
            # each branch must be a 0-arg fn returning same dtype; return a dummy scalar
            def _mk_push(k):
                return (lambda: (banks[k].push(feats), tf.constant(0, tf.int32))[1])
    
            return tf.switch_case(idx_i, branch_fns=[_mk_push(k) for k in range(num_combinations)])
    
        def _bank_sample(idx_i: tf.Tensor, N: tf.Tensor) -> tf.Tensor:
            idx_i = tf.cast(idx_i, tf.int32)
    
            def _mk_sample(k):
                return (lambda: banks[k].sample(N))
    
            return tf.switch_case(idx_i, branch_fns=[_mk_sample(k) for k in range(num_combinations)])
    
        # ---- teacher & student pre ----
        probs_L_s_pre, reps_L_s_pre = student_forward(xLw, training=False)
        probs_U_s_pre, reps_U_s_pre = student_forward(xUw, training=False)
        probs_L_t, reps_L_t = teacher_forward(xLw)
        probs_U_t, reps_U_t = teacher_forward(xUw)
    
        Lx = supervised_bce(yL, probs_L_s_pre)
    
        # ---- reliability mask from teacher ----
        H_u = _bernoulli_entropy_sum(probs_U_t)                      # [B,H,W]
        mean_entropy = tf.reduce_mean(H_u)
        a_t = alpha0 * (1.0 - (step_idx_f / tf.maximum(T_total, 1.0)))  # scalar
        gamma = _quantile(H_u, 1.0 - a_t)                             # [B]
        reliable_mask = H_u <= gamma[:, None, None]                   # [B,H,W]
        m_rel = tf.cast(reliable_mask, tf.float32)[..., None]         # [B,H,W,1]
    
        Lu = unsup_bce(probs_U_s_pre, probs_U_t, m_rel)
        lam_u = lambda_u_eta * tf.cast(tf.size(H_u), tf.float32) / tf.maximum(tf.reduce_sum(m_rel), 1e-8)
    
        # ---- flatten helpers ----
        def _flatten_hw(x): return tf.reshape(x, [-1, tf.shape(x)[-1]])
        reps_L_s_flat_pre, reps_U_s_flat_pre = _flatten_hw(reps_L_s_pre), _flatten_hw(reps_U_s_pre)
        reps_L_t_flat,  reps_U_t_flat  = _flatten_hw(reps_L_t),  _flatten_hw(reps_U_t)
    
        yL_binary = tf.cast(yL > 0.5, tf.float32)
        yL_combo_ids = _mask_to_combo_id(yL_binary)                   # [B,H,W]
        yhat_U_binary = tf.cast(probs_U_t > 0.5, tf.float32)
        yhat_U_combo_ids = _mask_to_combo_id(yhat_U_binary)
        yL_combo_flat = tf.reshape(yL_combo_ids, [-1])
        yhat_U_combo_flat = tf.reshape(yhat_U_combo_ids, [-1])
        reliable_flat = tf.reshape(reliable_mask, [-1])
    
        NpixL = tf.cast(tf.size(yL_combo_flat), tf.int32)
    
        # precompute "unreliable" reps once (teacher on U)
        unreli_indices_t = tf.where(tf.logical_not(reliable_flat))[:, 0]          # [M?]
        unreli_reps = tf.gather(reps_U_t_flat, unreli_indices_t)                  # [M?,E]
    
        # containers for contrastive components
        ta_idx   = tf.TensorArray(tf.int32,   size=0, dynamic_size=True, infer_shape=False)
        ta_proto = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
        ta_negs  = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
    
        combos_used = tf.constant(0, tf.int32)
        Lc_total = tf.constant(0.0, tf.float32)
    
        # loop over combos using tf.while_loop (no Python list ops)
        i0 = tf.constant(0, tf.int32)
        def loop_cond(i, *_): return tf.less(i, tf.constant(num_combinations, tf.int32))
    
        def loop_body(i, combos_used, ta_idx, ta_proto, ta_negs, Lc_total):
            # labeled indices for combo i
            idx_Al = tf.cond(
                NpixL > 0,
                lambda: tf.cast(tf.where(tf.equal(yL_combo_flat, i))[:,0], tf.int32),
                lambda: tf.zeros([0], tf.int32)
            )
            # unlabeled reliable indices with this combo
            combined_mask = tf.logical_and(tf.equal(yhat_U_combo_flat, i), reliable_flat)
            idx_Au = tf.cast(tf.where(combined_mask)[:,0], tf.int32)  # may be empty
    
            idx_all_teacher = tf.concat([idx_Al, NpixL + idx_Au], axis=0)  # [K]
            has_any = tf.greater(tf.shape(idx_all_teacher)[0], 0)
    
            def do_combo():
                # sample anchors
                idx_A_sampled = tf.random.shuffle(idx_all_teacher)[:M_anchors_per_class]     # [<=M]
                # anchors from student-pre reps (L + U concatenated)
                reps_concat_pre = tf.concat([reps_L_s_flat_pre, reps_U_s_flat_pre], axis=0)  # [L+U,E]
                anchors_pre = tf.gather(reps_concat_pre, idx_A_sampled)                      # [m,E]
    
                # prototype from teacher reps
                reps_concat_t = tf.concat([reps_L_t_flat, reps_U_t_flat], axis=0)            # [L+U,E]
                proto_pos = tf.reduce_mean(tf.gather(reps_concat_t, idx_all_teacher), axis=0)  # [E]
    
                # push unreliables to this combo's bank
                def do_push(): return _bank_push(i, unreli_reps)
                def no_push(): return tf.constant(0, tf.int32)
                _ = tf.cond(tf.greater(tf.shape(unreli_reps)[0], 0), do_push, no_push)
    
                # build negatives by iterating other combos (no Python lists)
                N_each = tf.constant(N_negs_per_class // (num_combinations - 1) + 1, tf.int32)
                neg_c0 = tf.constant(0, tf.int32)
                sampled_negs0 = tf.zeros([0, emb_dim], tf.float32)
    
                def neg_cond(neg_c, sampled_negs):
                    return tf.less(neg_c, tf.constant(num_combinations, tf.int32))
    
                def neg_body(neg_c, sampled_negs):
                    is_other = tf.not_equal(neg_c, i)
                    def add_from_bank():
                        s = _bank_sample(neg_c, N_each)                            # [?,E]
                        def add():
                            return tf.concat([sampled_negs, s], axis=0)
                        return tf.cond(tf.greater(tf.shape(s)[0], 0), add, lambda: sampled_negs)
                    new_sn = tf.cond(is_other, add_from_bank, lambda: sampled_negs)
                    return neg_c + 1, new_sn
    
                _, sampled_negs = tf.while_loop(
                    neg_cond, neg_body, [neg_c0, sampled_negs0],
                    parallel_iterations=1,
                    shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, emb_dim])]
                )
    
                has_negs = tf.greater(tf.shape(sampled_negs)[0], 0)
                def take_it():
                    Lc_c = _info_nce_proto(anchors_pre, tf.stop_gradient(proto_pos), tf.stop_gradient(sampled_negs), temperature)
                    ta_idx2   = ta_idx.write(combos_used, idx_A_sampled)
                    ta_proto2 = ta_proto.write(combos_used, tf.stop_gradient(proto_pos))
                    ta_negs2  = ta_negs.write(combos_used, tf.stop_gradient(sampled_negs))
                    return (combos_used + 1, ta_idx2, ta_proto2, ta_negs2, Lc_total + Lc_c)
                def skip_it():
                    return (combos_used, ta_idx, ta_proto, ta_negs, Lc_total)
    
                return tf.cond(has_negs, take_it, skip_it)
    
            def no_combo():
                return (combos_used, ta_idx, ta_proto, ta_negs, Lc_total)
    
            combos_used2, ta_idx2, ta_proto2, ta_negs2, Lc_total2 = tf.cond(has_any, do_combo, no_combo)
            return i + 1, combos_used2, ta_idx2, ta_proto2, ta_negs2, Lc_total2
    
        _, combos_used, ta_idx, ta_proto, ta_negs, Lc_total = tf.while_loop(
            loop_cond, loop_body,
            loop_vars=[i0, combos_used, ta_idx, ta_proto, ta_negs, Lc_total],
            parallel_iterations=1
        )
    
        Lc_mean_pre = tf.math.divide_no_nan(Lc_total, tf.cast(tf.maximum(combos_used, 1), tf.float32))
    
        # ---- Grad pass (recompute student; reuse proto/neg and indices) ----
        with tf.GradientTape() as tape:
            probs_L_s_grad, reps_L_s_grad = student_forward(xLw, training=True)
            probs_U_s_grad, reps_U_s_grad = student_forward(xUw, training=True)
    
            loss_Lx = supervised_bce(yL, probs_L_s_grad)
            loss_Lu = unsup_bce(probs_U_s_grad, probs_U_t, m_rel)
    
            reps_s_grad_flat = tf.concat(
                [tf.reshape(reps_L_s_grad, [-1, tf.shape(reps_L_s_grad)[-1]]),
                 tf.reshape(reps_U_s_grad, [-1, tf.shape(reps_U_s_grad)[-1]])],
                axis=0
            )
    
            j0 = tf.constant(0, tf.int32)
            Lc_total_grad = tf.constant(0.0, tf.float32)
    
            def cond_j(j, Lc_acc): return tf.less(j, combos_used)
            def body_j(j, Lc_acc):
                idx_j   = ta_idx.read(j)     # [m]
                proto_j = ta_proto.read(j)   # [E]
                negs_j  = ta_negs.read(j)    # [n,E]
                anchors_live = tf.gather(reps_s_grad_flat, idx_j)  # [m,E]
                Lc_c_grad = _info_nce_proto(anchors_live, proto_j, negs_j, temperature)
                return j + 1, Lc_acc + Lc_c_grad
    
            _, Lc_total_grad = tf.while_loop(cond_j, body_j, [j0, Lc_total_grad], parallel_iterations=1)
            Lc_mean_grad = tf.math.divide_no_nan(Lc_total_grad, tf.cast(tf.maximum(combos_used, 1), tf.float32))
    
            total_loss = loss_Lx + lam_u * loss_Lu + lambda_c * Lc_mean_grad
            scaled_loss = optimizer.get_scaled_loss(total_loss) if hasattr(optimizer, "get_scaled_loss") else total_loss
    
        vars_to_train = model.trainable_variables + proj_s.trainable_variables
        grads = tape.gradient(scaled_loss, vars_to_train)
        if hasattr(optimizer, "get_unscaled_gradients"): grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, vars_to_train))
    
        # EMA
        _ema_update_models([teacher_model, teacher_proj], [model, proj_s], float(ema_momentum))
    
        return Lx, Lu, Lc_mean_pre, lam_u, mean_entropy
        


    # --- History & loop ---
    history = {'val_loss': [], 'alpha': [], 'avg_entropy': []}
    best_val = float('inf'); step_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        idxL = np.random.permutation(Nl); idxU = np.random.permutation(Nu)
        mx.reset_state(); mu.reset_state(); mc.reset_state(); mhe.reset_state()

        for step in range(steps_per_epoch):
            step_counter += 1
            sL = (step % int(math.ceil(Nl/batch_size))) * batch_size; sU = (step % int(math.ceil(Nu/batch_size))) * batch_size
            bL = idxL[sL:sL+batch_size]; bU = idxU[sU:sU+batch_size]
            xL = tf.gather(lab_imgs, bL); yL = tf.gather(lab_masks,  bL); xU = tf.gather(unl_imgs,  bU)

            # light augs outside; they are already tf.function’d
            (xL, yL) = random_hflip_batch(xL, yL); (xU, _) = random_hflip_batch(xU, None)
            xLw = weak_gray_batch(xL); xUw = weak_gray_batch(xU)

            # compiled step
            Lx, Lu, Lc_mean, lam_u, mean_H = train_step(xLw, yL, xUw, tf.constant(float(step_counter), tf_dtype_f))

            # metrics
            mx.update_state(Lx); mu.update_state(Lu); mc.update_state(Lc_mean); mhe.update_state(mean_H)

            if (step+1) % 20 == 0 or (step+1) == steps_per_epoch:
                print(f"  step {step+1}/{steps_per_epoch} | "
                      f"Lx {mx.result().numpy():.4f} | "
                      f"Lu {mu.result().numpy():.4f} (λu={lam_u.numpy():.2f}) | "
                      f"Lc {mc.result().numpy():.4f} | "
                      f"α_unrel {alpha0 * (1.0 - (float(step_counter)/(epochs*steps_per_epoch))):.3f}")
                mx.reset_state(); mu.reset_state(); mc.reset_state()

        # --- Validation ---
        V = int(val_imgs.shape[0]); vsteps = math.ceil(V / batch_size); v_losses = []
        for i in range(vsteps):
            s = i * batch_size; e = min(s + batch_size, V)
            probs_v_s, _ = student_forward(val_imgs[s:e], training=False)
            v_losses.append(float(supervised_bce(val_masks[s:e], probs_v_s).numpy()))
        vloss = float(np.mean(v_losses)) if v_losses else float('inf')

        avg_entropy_epoch = float(mhe.result().numpy()) if mhe.count.numpy() > 0 else 0.0
        a_t_unrel_epoch = alpha0 * (1.0 - (float(step_counter)/(epochs*steps_per_epoch)))
        history['val_loss'].append(vloss)
        history['alpha'].append(a_t_unrel_epoch)
        history['avg_entropy'].append(avg_entropy_epoch)
        print(f"Validation Loss: {vloss:.4f} | Avg Entropy: {avg_entropy_epoch:.4f} | α_unrel {a_t_unrel_epoch:.3f}")

        if vloss < best_val:
            best_val = vloss
            model.save(filepath_h5, include_optimizer=False, save_format="h5" if filepath_h5.endswith(".h5") else "tf")

    # --- Benchmarking ---
    best_model = tf.keras.models.load_model(filepath_h5, compile=False)
    val_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'val_predictions', approach, modelname)
    test_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(paths.HELA_BASE_DIR, 'train_unlabeled_predictions', approach, modelname)

    mIoU_val, mIoU_ad_val, mcce_val = benchmark_hela(best_model, paths.HELA_VAL_DIR,  val_pred_dir,  h, w, c)
    mIoU_tst, mIoU_ad_tst, mcce_tst = benchmark_hela(best_model, paths.HELA_TEST_DIR, test_pred_dir, h, w, c)
    mIoU_unl, mIoU_ad_unl, mcce_unl = benchmark_hela(best_model, paths.HELA_TRAIN_UNLABELED_DIR, unlabeled_pred_dir, h, w, c)

    return (mIoU_val, mIoU_tst, mIoU_unl,
            mIoU_ad_val, mIoU_ad_tst, mIoU_ad_unl,
            mcce_val, mcce_tst, mcce_unl, history)



def train_multiclass_u2pl(
    model,
    filepath_h5,
    modelname,
    h, w, c, num_classes, class_to_color_mapping,
    # U2PL hypers
    alpha0=0.2,
    lambda_u_eta=1.0,
    temperature=0.5,
    prt_rl=3, prt_rh=20,
    delta_p=0.3,
    bank_capacity_per_class=4096,
    M_anchors_per_class=50,
    N_negs_per_class=256,
    ema_momentum=0.996,
    lambda_c=0.1,
    # General
    epochs=120,
    batch_size=8,
    optimizer=None,
    # dataset I/O
    train_labeled_images_dir=None,
    train_labeled_masks_dir=None,
    train_unlabeled_images_dir=None,
    val_images_dir=None,
    val_masks_dir=None,
    test_images_dir=None,
    test_masks_dir=None,
    train_unlabeled_masks_dir=None,
    dataset_base_dir=None,
    dataset_name_for_aug='SUIM',
    approach='u2pl',
):
    """Faithful U2PL trainer for multi-class segmentation with compiled train_step."""
    from functions import (load_labeled_data_multiclass_mask, load_unlabeled_images, benchmark_multiclass, _build_optimizer_weights)
    import numpy as np, math, os, tensorflow as tf

    if optimizer is None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    # --- Data ---
    lab_imgs_np, lab_1hot_np = load_labeled_data_multiclass_mask(train_labeled_images_dir, train_labeled_masks_dir, num_classes)
    unl_imgs_np = load_unlabeled_images(train_unlabeled_images_dir)
    val_imgs_np, val_1hot_np = load_labeled_data_multiclass_mask(val_images_dir, val_masks_dir, num_classes)
    lab_ids_np = np.argmax(lab_1hot_np, axis=-1).astype(np.int32)
    val_ids_np = np.argmax(val_1hot_np, axis=-1).astype(np.int32)
    lab_imgs  = tf.convert_to_tensor(lab_imgs_np,  tf.float32); lab_ids   = tf.convert_to_tensor(lab_ids_np,   tf.int32)
    unl_imgs  = tf.convert_to_tensor(unl_imgs_np,  tf.float32); val_imgs  = tf.convert_to_tensor(val_imgs_np,  tf.float32)
    val_ids   = tf.convert_to_tensor(val_ids_np,   tf.int32)

    # --- Model split ---
    _ = model(tf.zeros([1, h, w, c], tf.float32), training=False); _build_optimizer_weights(optimizer, model.trainable_variables)
    feat_s, tail_s = _build_feature_and_tail_models(model)
    if feat_s is None or tail_s is None: raise RuntimeError("U2PL: cannot split model.")
    proj_s = _build_projection_head(feat_s.outputs[0].shape)
    teacher_model = _clone_model_weights(model); teacher_model.trainable = False
    teacher_feat, teacher_tail = _build_feature_and_tail_models(teacher_model)
    teacher_proj = _clone_model_weights(proj_s); teacher_proj.trainable = False
    emb_dim = int(proj_s.output_shape[-1])

    # --- Banks ---
    banks = [TFBank(bank_capacity_per_class, emb_dim, name=f"bank_cls_{i}") for i in range(num_classes)]

    # --- Forwards ---
    def student_forward(x, training):
        f = tf.cast(feat_s(x, training=training), tf.float32)
        logits = tf.cast(tail_s(f, training=training), tf.float32)
        reps   = tf.cast(proj_s(f, training=training), tf.float32)
        return logits, reps
    def teacher_forward(x):
        f = tf.cast(teacher_feat(x, training=False), tf.float32)
        logits = tf.cast(teacher_tail(f, training=False), tf.float32)
        reps   = tf.cast(teacher_proj(f, training=False), tf.float32)
        return logits, reps

    # --- Losses ---
    ce_sup_logits = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    def supervised_ce(y_ids, logits):
        y = tf.cast(y_ids, tf.int32)
        if y.shape.rank == 4:
            shape = tf.shape(y)
            if shape[1] <= num_classes and shape[1] < shape[2]: 
                y = tf.transpose(y, [0, 2, 3, 1])
            if y.shape[-1] == 1:
                y = tf.squeeze(y, axis=-1)
            else:
                y = tf.argmax(y, axis=-1, output_type=tf.int32)
        y_ids_matched = y
        logits_shape = tf.shape(logits)
        h_l, w_l = logits_shape[1], logits_shape[2]
        y_shape = tf.shape(y_ids_matched)
        h_y, w_y = y_shape[1], y_shape[2]
        needs_resize = tf.logical_or(tf.not_equal(h_y, h_l), tf.not_equal(w_y, w_l))
        def resize_labels():
            y_4d = tf.expand_dims(y_ids_matched, axis=-1)
            y_resized_4d = tf.image.resize(y_4d, [h_l, w_l], method='nearest')
            return tf.squeeze(y_resized_4d, axis=-1)
        def do_nothing():
            return y_ids_matched
        y_ids_matched = tf.cond(needs_resize, resize_labels, do_nothing)
        logits = tf.cast(logits, tf.float32)
        logits = tf.clip_by_value(logits, -30.0, 30.0)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        y_true_one_hot = tf.one_hot(y_ids_matched, num_classes, dtype=tf.float32)
        loss_ce_per_pixel = -tf.reduce_sum(y_true_one_hot * log_probs, axis=-1)
        loss_ce = tf.reduce_mean(loss_ce_per_pixel)
        y_pred_probs = tf.nn.softmax(logits, axis=-1)
        y_true_flat = tf.reshape(y_true_one_hot, [-1, num_classes])
        y_pred_flat = tf.reshape(y_pred_probs, [-1, num_classes])
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
        denominator = tf.reduce_sum(y_true_flat, axis=0) + tf.reduce_sum(y_pred_flat, axis=0)
        dice_per_class = (2. * intersection + 1e-6) / (denominator + 1e-6)
        loss_dice = 1.0 - tf.reduce_mean(dice_per_class)
        return 0.5 * loss_ce + 0.5 * loss_dice
        
    def unsup_ce(student_logits, teacher_probs, mask01):
        ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')(teacher_probs, student_logits)
        ce = ce * tf.squeeze(mask01, -1); num = tf.maximum(tf.reduce_sum(mask01, axis=[1,2,3]), 1.0)
        s = tf.reduce_sum(ce, axis=[1,2]); return tf.reduce_mean(s / num)

    # --- Schedule & metrics ---
    Nl = int(lab_imgs.shape[0]); Nu = int(unl_imgs.shape[0])
    steps_per_epoch = max(int(math.ceil(Nl/batch_size)), int(math.ceil(Nu/batch_size)))
    T_total = tf.constant(float(epochs * steps_per_epoch), tf.float32)
    mx = tf.keras.metrics.Mean(); mu = tf.keras.metrics.Mean(); mc = tf.keras.metrics.Mean(); mhe = tf.keras.metrics.Mean()

    def _flatten_hw(x): return tf.reshape(x, [-1, tf.shape(x)[-1]])
    K_t = tf.constant(num_classes, tf.int32)
    N_negs_t = tf.constant(N_negs_per_class, tf.int32)
    
    # Define a constant for sub-sampling negatives to control memory
    MAX_NEGS_PER_SOURCE = 1024

    # graph-safe bank access
    def _bank_push(idx_i: tf.Tensor, feats: tf.Tensor):
        idx_i = tf.cast(idx_i, tf.int32)
        def _mk_push(k):
            return (lambda: (banks[k].push(feats), tf.constant(0, tf.int32))[1])
        return tf.switch_case(idx_i, branch_fns=[_mk_push(k) for k in range(num_classes)])
    def _bank_sample(idx_i: tf.Tensor, N: tf.Tensor) -> tf.Tensor:
        idx_i = tf.cast(idx_i, tf.int32)
        def _mk_sample(k):
            return (lambda: banks[k].sample(N))
        return tf.switch_case(idx_i, branch_fns=[_mk_sample(k) for k in range(num_classes)])

    @tf.function(experimental_relax_shapes=True)
    def train_step(xLw, yL_ids, xUw, step_idx_f: tf.Tensor):
        logits_L_s_pre, reps_L_s_pre = student_forward(xLw, training=False)
        logits_U_s_pre, reps_U_s_pre = student_forward(xUw, training=False)
        logits_L_t, reps_L_t = teacher_forward(xLw); logits_U_t, reps_U_t = teacher_forward(xUw)
        probs_L_t = _softmax(logits_L_t); probs_U_t = _softmax(logits_U_t)
        Lx = supervised_ce(yL_ids, logits_L_s_pre)
        H_u = _entropy_from_probs(probs_U_t)
        mean_entropy = tf.reduce_mean(H_u)
        a_t = alpha0 * (1.0 - (step_idx_f / tf.maximum(T_total, 1.0)))
        gamma = _quantile(H_u, 1.0 - a_t)
        reliable = H_u <= gamma[:, None, None]
        m_rel = tf.cast(reliable, tf.float32)[..., None]
        yhat_t = tf.argmax(probs_U_t, axis=-1, output_type=tf.int32)
        Lu = unsup_ce(logits_U_s_pre, tf.one_hot(yhat_t, K_t, dtype=tf.float32), m_rel)
        lam_u = lambda_u_eta * tf.cast(tf.size(H_u), tf.float32) / tf.maximum(tf.reduce_sum(m_rel), 1e-8)
        
        reps_L_s_pre_f, reps_U_s_pre_f = _flatten_hw(reps_L_s_pre), _flatten_hw(reps_U_s_pre)
        reps_L_t_f, reps_U_t_f         = _flatten_hw(reps_L_t),     _flatten_hw(reps_U_t)
        pL_f, pU_f                     = _flatten_hw(probs_L_t),    _flatten_hw(probs_U_t)
        ranks_L = _rank_descending(probs_L_t); ranks_U = _rank_descending(probs_U_t)
        ranks_L_f, ranks_U_f           = _flatten_hw(tf.cast(ranks_L, tf.float32)), _flatten_hw(tf.cast(ranks_U, tf.float32))
        yL_f = tf.reshape(yL_ids, [-1])
        NpixL = tf.cast(tf.size(yL_f), tf.int32)
        reliable_flat = tf.reshape(reliable, [-1])

        ta_idx   = tf.TensorArray(tf.int32,   size=0, dynamic_size=True, infer_shape=False); ta_proto = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False); ta_negs  = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
        classes_used = tf.constant(0, tf.int32); Lc_total = tf.constant(0.0, tf.float32)

        i0 = tf.constant(0, tf.int32)
        def cond_c(ci, *_): return tf.less(ci, K_t)
        def body_c(ci, classes_used, ta_idx, ta_proto, ta_negs, Lc_total):
            idx_Al = tf.cond(NpixL > 0, lambda: tf.cast(tf.where(tf.equal(yL_f, ci))[:,0], tf.int32), lambda: tf.zeros([0], tf.int32))
            idx_Al = tf.cond(tf.shape(idx_Al)[0] > 0, lambda: tf.gather(idx_Al, tf.where(tf.gather(pL_f[:, ci], idx_Al) > delta_p)[:,0]), lambda: idx_Al)
            yhat_U_f = tf.reshape(yhat_t, [-1])
            idx_Au = tf.cast(tf.where(tf.logical_and(tf.equal(yhat_U_f, ci), reliable_flat))[:,0], tf.int32)
            idx_Au = tf.cond(tf.shape(idx_Au)[0] > 0, lambda: tf.gather(idx_Au, tf.where(tf.gather(pU_f[:, ci], idx_Au) > delta_p)[:,0]), lambda: idx_Au)
            idx_all_teacher = tf.concat([idx_Al, NpixL + idx_Au], axis=0)
            has_any = tf.greater(tf.shape(idx_all_teacher)[0], 0)

            def do_class():
                idx_A_sampled = tf.random.shuffle(idx_all_teacher)[:M_anchors_per_class]
                reps_concat_pre = tf.concat([reps_L_s_pre_f, reps_U_s_pre_f], axis=0)
                anchors_pre = tf.gather(reps_concat_pre, idx_A_sampled)
                reps_concat_t = tf.concat([reps_L_t_f, reps_U_t_f], axis=0)
                proto_pos = tf.reduce_mean(tf.gather(reps_concat_t, idx_all_teacher), axis=0)

                # --- MEMORY FIX STARTS HERE ---
                not_c_idx = tf.cast(tf.where(tf.not_equal(yL_f, ci))[:,0], tf.int32)
                def get_labeled_negs():
                    r_c_L = tf.gather(ranks_L_f[:, ci], not_c_idx)
                    seln_idx = tf.where(tf.cast(r_c_L, tf.int32) <= prt_rl)[:,0]
                    seln_idx_shuffled = tf.random.shuffle(seln_idx)[:MAX_NEGS_PER_SOURCE]
                    return tf.gather(reps_L_t_f, tf.gather(not_c_idx, seln_idx_shuffled))
                labeled_negs = tf.cond(tf.shape(not_c_idx)[0] > 0, get_labeled_negs, lambda: tf.zeros([0, emb_dim], tf.float32))
                
                unreli_idx = tf.where(tf.logical_not(reliable_flat))[:,0]
                def get_unlabeled_negs():
                    r_c_U = tf.gather(ranks_U_f[:, ci], unreli_idx)
                    cond = tf.logical_and(tf.cast(r_c_U, tf.int32) >= prt_rl, tf.cast(r_c_U, tf.int32) <  prt_rh)
                    valid_unreli_indices = tf.boolean_mask(unreli_idx, cond)
                    valid_unreli_indices_shuffled = tf.random.shuffle(valid_unreli_indices)[:MAX_NEGS_PER_SOURCE]
                    return tf.gather(reps_U_t_f, valid_unreli_indices_shuffled)
                unlabeled_negs = tf.cond(tf.shape(unreli_idx)[0] > 0, get_unlabeled_negs, lambda: tf.zeros([0, emb_dim], tf.float32))
                
                current_negs_to_push = tf.concat([labeled_negs, unlabeled_negs], axis=0)
                # --- MEMORY FIX ENDS HERE ---

                def do_push(): return (_bank_push(ci, current_negs_to_push), tf.constant(0, tf.int32))[1]
                _ = tf.cond(tf.greater(tf.shape(current_negs_to_push)[0], 0), do_push, lambda: tf.constant(0, tf.int32))
                sampled_negs = _bank_sample(ci, N_negs_t)
                sampled_negs = tf.cond(tf.greater(tf.shape(sampled_negs)[0], 0), lambda: sampled_negs, lambda: current_negs_to_push)
                has_final = tf.greater(tf.shape(sampled_negs)[0], 0)

                def take_it():
                    Lc_c = _info_nce_proto(anchors_pre, tf.stop_gradient(proto_pos), tf.stop_gradient(sampled_negs), temperature)
                    ta_idx2   = ta_idx.write(classes_used, idx_A_sampled)
                    ta_proto2 = ta_proto.write(classes_used, tf.stop_gradient(proto_pos))
                    ta_negs2  = ta_negs.write(classes_used, tf.stop_gradient(sampled_negs))
                    return (classes_used + 1, ta_idx2, ta_proto2, ta_negs2, Lc_total + Lc_c)
                def skip_it(): return (classes_used, ta_idx, ta_proto, ta_negs, Lc_total)
                return tf.cond(has_final, take_it, skip_it)
            def no_class(): return (classes_used, ta_idx, ta_proto, ta_negs, Lc_total)
            classes_used2, ta_idx2, ta_proto2, ta_negs2, Lc_total2 = tf.cond(has_any, do_class, no_class)
            return ci + 1, classes_used2, ta_idx2, ta_proto2, ta_negs2, Lc_total2

        _, classes_used, ta_idx, ta_proto, ta_negs, Lc_total = tf.while_loop(cond_c, body_c, loop_vars=[i0, classes_used, ta_idx, ta_proto, ta_negs, Lc_total], parallel_iterations=1)
        Lc_mean_pre = tf.math.divide_no_nan(Lc_total, tf.cast(tf.maximum(classes_used, 1), tf.float32))

        with tf.GradientTape() as tape:
            logits_L_s_grad, reps_L_s_grad = student_forward(xLw, training=True)
            logits_U_s_grad, reps_U_s_grad = student_forward(xUw, training=True)
            loss_Lx = supervised_ce(yL_ids, logits_L_s_grad)
            loss_Lu = unsup_ce(logits_U_s_grad, tf.one_hot(yhat_t, K_t, dtype=tf.float32), m_rel)
            reps_s_grad_flat = tf.concat([_flatten_hw(reps_L_s_grad), _flatten_hw(reps_U_s_grad)], axis=0)
            
            def compute_contrastive_loss():
                j0 = tf.constant(0, tf.int32); Lc_total_grad_inner = tf.constant(0.0, tf.float32)
                def cond_j(j, Lc_acc): return tf.less(j, classes_used)
                def body_j(j, Lc_acc):
                    idx_j, proto_j, negs_j = ta_idx.read(j), ta_proto.read(j), ta_negs.read(j)
                    anchors_live = tf.gather(reps_s_grad_flat, idx_j)
                    Lc_c_grad = _info_nce_proto(anchors_live, proto_j, negs_j, temperature)
                    return j + 1, Lc_acc + Lc_c_grad
                _, Lc_total_grad_inner = tf.while_loop(cond_j, body_j, [j0, Lc_total_grad_inner], parallel_iterations=1)
                return tf.math.divide_no_nan(Lc_total_grad_inner, tf.cast(tf.maximum(classes_used, 1), tf.float32))
            
            Lc_mean_grad = tf.cond(tf.greater(classes_used, 0), compute_contrastive_loss, lambda: tf.constant(0.0, tf.float32))
            total_loss = loss_Lx + lam_u * loss_Lu + lambda_c * Lc_mean_grad
            scaled_loss = optimizer.get_scaled_loss(total_loss) if hasattr(optimizer, "get_scaled_loss") else total_loss

        vars_to_train = model.trainable_variables + proj_s.trainable_variables
        grads = tape.gradient(scaled_loss, vars_to_train)
        if hasattr(optimizer, "get_unscaled_gradients"): grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, vars_to_train))
        _ema_update_models([teacher_model, teacher_proj], [model, proj_s], float(ema_momentum))
        return Lx, Lu, Lc_mean_pre, lam_u, mean_entropy

    @tf.function(reduce_retracing=True)
    def _val_step(x, y_ids):
        logits_v_s, _ = student_forward(x, training=False)
        return supervised_ce(y_ids, logits_v_s)

    # --- Loop ---
    history = {'val_loss': [], 'alpha': [], 'avg_entropy': []}
    best_val = float('inf'); step_counter = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        idxL = np.random.permutation(Nl); idxU = np.random.permutation(Nu)
        mx.reset_state(); mu.reset_state(); mc.reset_state(); mhe.reset_state()

        for step in range(steps_per_epoch):
            step_counter += 1
            sL = (step % int(math.ceil(Nl/batch_size))) * batch_size; sU = (step % int(math.ceil(Nu/batch_size))) * batch_size
            bL = idxL[sL:sL+batch_size]; bU = idxU[sU:sU+batch_size]
            xL = tf.gather(lab_imgs, bL); yL = tf.gather(lab_ids,  bL); xU = tf.gather(unl_imgs,  bU)
            (xL, yL) = random_hflip_batch(xL, yL); (xU, _)  = random_hflip_batch(xU, None)
            xLw = weak_photometric_batch(xL)
            xUw = weak_photometric_batch(xU)

            Lx, Lu, Lc_mean, lam_u, mean_H = train_step(xLw, yL, xUw, tf.constant(float(step_counter), tf.float32))
            mx.update_state(Lx); mu.update_state(Lu); mc.update_state(Lc_mean); mhe.update_state(mean_H)

            if (step+1) % 20 == 0 or (step+1) == steps_per_epoch:
                print(f"  step {step+1}/{steps_per_epoch} | "
                      f"Lx {mx.result().numpy():.4f} | "
                      f"Lu {mu.result().numpy():.4f} (λu={lam_u.numpy():.2f}) | "
                      f"Lc {mc.result().numpy():.4f} | "
                      f"α_unrel {alpha0 * (1.0 - (float(step_counter)/(epochs*steps_per_epoch))):.3f}")
                mx.reset_state(); mu.reset_state(); mc.reset_state()

        # --- Validation ---
        V = int(val_imgs.shape[0]); vsteps = math.ceil(V / batch_size); v_losses = []
        for i in range(vsteps):
            s = i * batch_size; e = min(s + batch_size, V)
            v_losses.append(float(_val_step(val_imgs[s:e], val_ids[s:e]).numpy()))
        vloss = float(np.mean(v_losses)) if v_losses else float('inf')

        avg_entropy_epoch = float(mhe.result().numpy()) if mhe.count.numpy() > 0 else 0.0
        a_t_unrel_epoch = alpha0 * (1.0 - (float(step_counter)/(epochs*steps_per_epoch)))
        history['val_loss'].append(vloss)
        history['alpha'].append(a_t_unrel_epoch)
        history['avg_entropy'].append(avg_entropy_epoch)
        print(f"Validation Loss: {vloss:.4f} | Avg Entropy: {avg_entropy_epoch:.4f} | α_unrel {a_t_unrel_epoch:.3f}")

        if vloss < best_val:
            best_val = vloss
            model.save(filepath_h5, include_optimizer=False, save_format="h5" if filepath_h5.endswith(".h5") else "tf")

    # --- Benchmarking ---
    best_model = tf.keras.models.load_model(filepath_h5, compile=False)
    val_pred_dir = os.path.join(dataset_base_dir, 'val_predictions', approach, modelname)
    test_pred_dir = os.path.join(dataset_base_dir, 'test_predictions', approach, modelname)
    unlabeled_pred_dir = os.path.join(dataset_base_dir, 'train_unlabeled_predictions', approach, modelname)

    mPA_val, mIoU_val = benchmark_multiclass(best_model, val_images_dir, val_masks_dir, val_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    mPA_test, mIoU_test = benchmark_multiclass(best_model, test_images_dir, test_masks_dir, test_pred_dir, h, w, c, class_to_color_mapping, print_results=False)
    mPA_unl, mIoU_unl = benchmark_multiclass(best_model, train_unlabeled_images_dir, train_unlabeled_masks_dir, unlabeled_pred_dir, h, w, c, class_to_color_mapping, print_results=False)

    return mPA_val, mPA_test, mPA_unl, mIoU_val, mIoU_test, mIoU_unl, history










##############################################################################################################################################################################################
###                         UniMatch (Unified Dual-Stream Perturbations)
##############################################################################################################################################################################################

def unimatch_loss(p_teacher, p_student_s1, p_student_s2, p_student_fp, mask, lambda_u=0.5, mu_u=0.5):
    """
    UniMatch Loss for Single-Label (Softmax) tasks.
    """
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none')
    
    loss_fp = cce(p_teacher, p_student_fp)
    loss_s1 = cce(p_teacher, p_student_s1)
    loss_s2 = cce(p_teacher, p_student_s2)
    
    loss_dual = (loss_s1 + loss_s2) * 0.5
    loss_unsup_pixel = lambda_u * loss_fp + mu_u * loss_dual
    
    # Mask is [B, H, W, 1] or [B, H, W], ensure [B, H, W]
    if len(mask.shape) == 4:
        mask = tf.squeeze(mask, -1)
        
    loss_unsup_masked = loss_unsup_pixel * mask
    total_valid = tf.reduce_sum(mask) + 1e-6
    return tf.reduce_sum(loss_unsup_masked) / total_valid


def unimatch_loss_sigmoid(p_teacher, p_student_s1, p_student_s2, p_student_fp, mask, channel_weights, lambda_u=0.5, mu_u=0.5):
    """
    UniMatch Loss for Multi-Label (Sigmoid) tasks (HeLa).
    Accepts mask of shape [B, H, W, C] to mask individual channels.
    """
    w_ch = tf.reshape(tf.constant(channel_weights, tf.float32), [1, 1, 1, 3])
    
    def bce_map(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        return -(y_true * tf.math.log(y_pred) + (1. - y_true) * tf.math.log1p(-y_pred))

    loss_fp = bce_map(p_teacher, p_student_fp)
    loss_s1 = bce_map(p_teacher, p_student_s1)
    loss_s2 = bce_map(p_teacher, p_student_s2)
    
    loss_dual = (loss_s1 + loss_s2) * 0.5
    
    # Combined map [B, H, W, C]
    loss_unsup_map = (lambda_u * loss_fp + mu_u * loss_dual) * w_ch
    
    # Masking
    loss_unsup_masked = loss_unsup_map * mask
    
    total_valid = tf.reduce_sum(mask) + 1e-6
    return tf.reduce_sum(loss_unsup_masked) / total_valid


def hela_strong_aug(x):
    """
    HeLa-specific strong augmentation. 
    Standard 'strong' augs (blur 7x7) wipe out cells. We use milder settings.
    """
    x = tf.cast(x, tf.float32)
    
    # 1. Random Brightness/Contrast (Moderate)
    alpha = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.8, 1.2)
    beta  = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -15.0, 15.0)
    x = tf.clip_by_value(x * alpha + beta, 0., 255.)
    
    # 2. Random Noise (Standard)
    noise = tf.random.normal(tf.shape(x), stddev=5.0)
    x = tf.clip_by_value(x + noise, 0., 255.)
    
    # 3. Random Blur (Very mild or skip)
    # Only blur 50% of the time, and use small kernel
    do_blur = tf.random.uniform([]) > 0.5
    if do_blur:
        x = tfa.image.gaussian_filter2d(x, filter_shape=(3, 3), sigma=1.0)
        
    return x


def unimatch_loss_sigmoid_hard(p_teacher, p_student_s1, p_student_s2, p_student_fp, mask, channel_weights, lambda_u, mu_u):
    """
    Computes UniMatch loss using HARD targets for consistency.
    """
    w_ch = tf.reshape(tf.constant(channel_weights, tf.float32), [1, 1, 1, 3])
    
    # 1. Convert Teacher to HARD Labels (0.0 or 1.0) based on 0.5 cut
    # Note: The 'mask' input already decided WHICH pixels are confident.
    # Here we just determine WHAT the label is.
    y_target = tf.cast(p_teacher > 0.5, tf.float32)
    
    def bce_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        return -(y_true * tf.math.log(y_pred) + (1. - y_true) * tf.math.log1p(-y_pred))

    # 2. Compute BCE against Hard Targets
    loss_fp = bce_loss(y_target, p_student_fp)
    loss_s1 = bce_loss(y_target, p_student_s1)
    loss_s2 = bce_loss(y_target, p_student_s2)
    
    # 3. Weighted Sum
    loss_dual = (loss_s1 + loss_s2) * 0.5
    loss_pixel_channel = (lambda_u * loss_fp + mu_u * loss_dual) * w_ch
    
    # 4. Apply Confidence Mask
    # mask is [B, H, W, 3]
    loss_masked = loss_pixel_channel * mask
    
    # 5. Normalize
    # We normalize by the sum of weights of valid pixels to avoid scaling issues
    valid_weight_sum = tf.reduce_sum(mask * w_ch) + 1e-6
    return tf.reduce_sum(loss_masked) / valid_weight_sum


def initialize_unimatch_with_baseline(encoder, decoder, baseline_path, h, w, c):
    """
    Loads weights from a pre-trained standard U-Net into the split UniMatch Encoder/Decoder.
    """
    if not os.path.exists(baseline_path):
        print(f"Warning: Baseline model not found at {baseline_path}. Training from scratch.")
        return

    print(f"Warm-starting UniMatch from: {baseline_path}")
    
    # 1. Load the baseline (supervised) model
    # compile=False avoids issues if custom losses/metrics aren't registered
    base_model = tf.keras.models.load_model(baseline_path, compile=False)

    # 2. Construct a proxy model joining Encoder and Decoder
    # This creates a graph identical to the standard U-Net
    inputs = tf.keras.Input(shape=(h, w, c))
    feats = encoder(inputs)
    outputs = decoder(feats)
    proxy_model = tf.keras.Model(inputs, outputs)

    # 3. Transfer weights
    # Because the architecture logic in unet.py is identical, the weight shapes align perfectly.
    try:
        proxy_model.set_weights(base_model.get_weights())
        print(" -> Weights transferred successfully.")
    except ValueError as e:
        print(f" -> STRICT WEIGHT LOADING FAILED: {e}")
        print(" -> Attempting layer-by-name transfer...")
        # Fallback: Iterative transfer
        # (This helps if auto-generated layer names differ slightly but shapes match)
        for layer in base_model.layers:
            if not layer.weights: continue
            
            # Try finding in encoder
            try:
                target = encoder.get_layer(layer.name)
                target.set_weights(layer.get_weights())
            except:
                # Try finding in decoder
                try:
                    target = decoder.get_layer(layer.name)
                    target.set_weights(layer.get_weights())
                except:
                    pass
        print(" -> Layer-by-name transfer attempt complete.")

    # Clean up heavy models to free memory
    del base_model
    del proxy_model
    gc.collect()


def mild_strong_aug(x):
    """
    Milder strong augmentation for Medical/Underwater domains.
    - No large kernel blurs.
    - Preserves color hue better.
    - Lower noise variance.
    """
    x = tf.cast(x, tf.float32)
    
    # 1. Brightness/Contrast (Moderate)
    alpha = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.7, 1.3) # +/- 30%
    beta  = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -20.0, 20.0) # +/- 20 pixel shift
    x = tf.clip_by_value(x * alpha + beta, 0., 255.)
    
    # 2. Saturation (Crucial for ISIC/SUIM to not destroy color info completely)
    # tf.image.random_saturation requires 3 channels
    x = tf.image.random_saturation(x, 0.7, 1.3)
    x = tf.clip_by_value(x, 0., 255.)

    # 3. Noise (Reduced)
    noise = tf.random.normal(tf.shape(x), stddev=8.0) # Reduced from 10-20
    x = tf.clip_by_value(x + noise, 0., 255.)
    
    # 4. Blur (Rare and Weak)
    # Only blur 20% of the time (was 50-100% in strong augs), use 3x3 kernel
    do_blur = tf.random.uniform([]) < 0.2
    if do_blur:
        x = tfa.image.gaussian_filter2d(x, filter_shape=(3, 3), sigma=0.5)
        
    return x

# ---------------------------------------------------------------------------------------
# TRAIN LOOP: ISIC 2018 (Binary/Sparse Categorical) 
# ---------------------------------------------------------------------------------------
def train_isic_unimatch_parts(
    encoder, decoder, filepath_h5, modelname, h, w, c,
    epochs, batch_size, optimizer,
    confidence_threshold=0.95, # Keep 0.95 for ISIC (Binary is easier)
    lambda_u=0.5, mu_u=0.5
):
    import paths
    import numpy as np
    
    lab_imgs_np, lab_masks_np = load_labeled_data_single_mask(paths.ISIC_2018_TRAIN_LABELED_IMAGES_DIR, paths.ISIC_2018_TRAIN_LABELED_MASKS_DIR)
    unl_imgs_np = load_unlabeled_images(paths.ISIC_2018_TRAIN_UNLABELED_IMAGES_DIR)
    val_imgs_np, val_masks_np = load_labeled_data_single_mask(paths.ISIC_2018_VAL_IMAGES_DIR, paths.ISIC_2018_VAL_MASKS_DIR)

    lab_imgs = tf.convert_to_tensor(lab_imgs_np, tf.float32)
    lab_masks = tf.convert_to_tensor(lab_masks_np, tf.int32)
    unl_imgs = tf.convert_to_tensor(unl_imgs_np, tf.float32)
    val_imgs = tf.convert_to_tensor(val_imgs_np, tf.float32)
    val_masks = tf.convert_to_tensor(val_masks_np, tf.int32)

    # Weights Init
    dummy = tf.zeros([1, h, w, c], tf.float32)
    _ = decoder(encoder(dummy))
    all_vars = encoder.trainable_variables + decoder.trainable_variables
    _build_optimizer_weights(optimizer, all_vars)
    
    sup_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function(reduce_retracing=True)
    def train_step(x_lab, y_lab, x_unl):
        x_lab_aug, y_lab_aug = random_hflip_batch(x_lab, y_lab)
        y_lab_aug = tf.cond(tf.equal(tf.rank(y_lab_aug), 4), lambda: tf.squeeze(y_lab_aug, axis=-1), lambda: y_lab_aug)
        x_lab_w = weak_photometric_batch(x_lab_aug)
        
        x_unl_w, _ = random_hflip_batch(x_unl, None)
        x_unl_w = weak_photometric_batch(x_unl_w)
        
        # --- CHANGED: Use mild_strong_aug instead of generic strong ---
        x_unl_s1 = mild_strong_aug(x_unl_w)
        x_unl_s2 = mild_strong_aug(x_unl_w)

        with tf.GradientTape() as tape:
            logits_lab = decoder(encoder(x_lab_w, training=True), training=True)
            loss_sup = sup_loss_fn(y_lab_aug, logits_lab)
            
            feats_w = encoder(x_unl_w, training=False)
            logits_w = decoder(feats_w, training=False)
            p_w = tf.nn.softmax(logits_w, axis=-1)
            
            max_probs = tf.reduce_max(p_w, axis=-1, keepdims=True)
            mask = tf.cast(max_probs >= confidence_threshold, tf.float32)
            mask_ratio = tf.reduce_mean(mask)

            p_s1 = tf.nn.softmax(decoder(encoder(x_unl_s1, training=True), training=True), axis=-1)
            p_s2 = tf.nn.softmax(decoder(encoder(x_unl_s2, training=True), training=True), axis=-1)
            
            # --- CHANGED: Reduced dropout from 0.5 to 0.2 ---
            feats_fp_raw = encoder(x_unl_w, training=True)
            c5_p = tf.nn.dropout(feats_fp_raw[0], rate=0.2) 
            feats_fp = [c5_p] + feats_fp_raw[1:]
            
            p_fp = tf.nn.softmax(decoder(feats_fp, training=True), axis=-1)
            loss_u = unimatch_loss(p_w, p_s1, p_s2, p_fp, mask, lambda_u, mu_u)
            
            scaled_loss = optimizer.get_scaled_loss(loss_sup + loss_u)

        grads = tape.gradient(scaled_loss, all_vars)
        grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, all_vars))
        return loss_sup + loss_u, loss_sup, loss_u, mask_ratio

    @tf.function
    def val_step(x, y):
        return sup_loss_fn(y, decoder(encoder(x, training=False), training=False))

    N_l = len(lab_imgs_np); N_u = len(unl_imgs_np)
    steps = max(int(np.ceil(N_l/batch_size)), int(np.ceil(N_u/batch_size)))
    history = {'loss': [], 'loss_sup': [], 'loss_unsup': [], 'val_loss': [], 'mask_ratio': []}
    best_val = float('inf')

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        il = np.random.permutation(N_l); iu = np.random.permutation(N_u)
        e_t, e_s, e_u, e_m = 0., 0., 0., 0.
        for step in tqdm(range(steps)):
            sl = (step % int(np.ceil(N_l/batch_size))) * batch_size
            su = (step % int(np.ceil(N_u/batch_size))) * batch_size
            lt, ls, lu, mr = train_step(tf.gather(lab_imgs, il[sl:sl+batch_size]), tf.gather(lab_masks, il[sl:sl+batch_size]), tf.gather(unl_imgs, iu[su:su+batch_size]))
            e_t+=lt; e_s+=ls; e_u+=lu; e_m+=mr
        
        vl = [val_step(val_imgs[i:i+batch_size], val_masks[i:i+batch_size]) for i in range(0, len(val_imgs), batch_size)]
        mv = np.mean(vl)
        history['val_loss'].append(mv); history['loss'].append(e_t.numpy()/steps); history['loss_sup'].append(e_s.numpy()/steps); history['loss_unsup'].append(e_u.numpy()/steps); history['mask_ratio'].append(e_m.numpy()/steps)
        print(f"Val: {mv:.4f} | Sup: {history['loss_sup'][-1]:.4f} | Unsup: {history['loss_unsup'][-1]:.4f} | Mask: {history['mask_ratio'][-1]:.2%}")
        
        if mv < best_val:
            best_val = mv
            fi = tf.keras.Input((h, w, c)); tf.keras.Model(fi, decoder(encoder(fi))).save(filepath_h5, include_optimizer=False)
            
    return history


# ---------------------------------------------------------------------------------------
# TRAIN LOOP: HeLa (Multilabel Sigmoid)
# ---------------------------------------------------------------------------------------
def train_hela_unimatch_parts(
    encoder, decoder, filepath_h5, modelname, h, w, c,
    epochs, batch_size, optimizer,
    confidence_threshold=0.95,
    lambda_u=0.5, mu_u=0.5,
    channel_weights=(1.0, 1.0, 3.0)
):
    import paths
    import numpy as np
    
    # Data
    l_imgs, lA, lD, lP = load_images_and_masks_hela(paths.HELA_TRAIN_LABELED_DIR)
    # Stack masks [N, H, W, 3]
    lab_masks_np = np.stack([lA, lD, lP], axis=-1).astype(np.float32)
    u_imgs_np = load_unlabeled_images(paths.HELA_TRAIN_UNLABELED_BRIGHTFIELD_DIR, read_as_grayscale=True)
    v_imgs, vA, vD, vP = load_images_and_masks_hela(paths.HELA_VAL_DIR)
    val_masks_np = np.stack([vA, vD, vP], axis=-1).astype(np.float32)

    # Tensors
    lab_imgs = tf.expand_dims(tf.convert_to_tensor(l_imgs, tf.float32), -1)
    lab_masks = tf.convert_to_tensor(lab_masks_np, tf.float32)
    unl_imgs = tf.expand_dims(tf.convert_to_tensor(u_imgs_np, tf.float32), -1)
    val_imgs = tf.expand_dims(tf.convert_to_tensor(v_imgs, tf.float32), -1)
    val_masks = tf.convert_to_tensor(val_masks_np, tf.float32)

    # Init
    dummy = tf.zeros([1, h, w, c], tf.float32)
    _ = decoder(encoder(dummy))
    all_vars = encoder.trainable_variables + decoder.trainable_variables
    _build_optimizer_weights(optimizer, all_vars)

    w_ch = tf.reshape(tf.constant(channel_weights, tf.float32), [1,1,1,3])
    
    def supervised_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.-1e-7)
        loss = -(y_true * tf.math.log(y_pred) + (1. - y_true) * tf.math.log1p(-y_pred))
        return tf.reduce_mean(loss * w_ch)

    @tf.function(reduce_retracing=True)
    def train_step(x_lab, y_lab, x_unl):
        # 1. Augmentations
        x_lab_aug, y_lab_aug = random_hflip_batch(x_lab, y_lab)
        x_unl_w, _ = random_hflip_batch(x_unl, None)
        
        # Weak & Strong
        x_unl_w_aug = weak_gray_batch(x_unl_w)
        x_unl_s1 = hela_strong_aug(x_unl_w) # Use specific HeLa strong aug
        x_unl_s2 = hela_strong_aug(x_unl_w)

        with tf.GradientTape() as tape:
            # 2. Supervised
            l_sup = supervised_loss(y_lab_aug, decoder(encoder(x_lab_aug, training=True), training=True))
            
            # 3. Teacher (Weak)
            p_w = decoder(encoder(x_unl_w_aug, training=False), training=False)
            
            # 4. Masking (Multi-label logic)
            # A pixel/channel is confident if p > tau (Positive) OR p < 1-tau (Negative)
            is_pos = p_w > confidence_threshold
            is_neg = p_w < (1.0 - confidence_threshold)
            conf_mask = tf.cast(is_pos | is_neg, tf.float32) # [B, H, W, 3]
            mr = tf.reduce_mean(conf_mask)
            
            # 5. Students (Strong)
            p_s1 = decoder(encoder(x_unl_s1, training=True), training=True)
            p_s2 = decoder(encoder(x_unl_s2, training=True), training=True)
            
            # 6. Feature Perturbation
            # Use milder dropout (0.1 or 0.2) for HeLa to preserve tiny cell features
            f_p = encoder(x_unl_w_aug, training=True)
            f_p_drop = [tf.nn.dropout(f_p[0], rate=0.2)] + f_p[1:] 
            p_fp = decoder(f_p_drop, training=True)
            
            # 7. Unsupervised Loss (Hard Targets)
            l_u = unimatch_loss_sigmoid_hard(p_w, p_s1, p_s2, p_fp, conf_mask, channel_weights, lambda_u, mu_u)
            
            total = l_sup + l_u
            scaled = optimizer.get_scaled_loss(total)

        grads = tape.gradient(scaled, all_vars)
        grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, all_vars))
        return total, l_sup, l_u, mr

    @tf.function
    def val_step(x, y):
        return supervised_loss(y, decoder(encoder(x, training=False), training=False))

    # Training Loop
    N_l = len(l_imgs); N_u = len(u_imgs_np)
    steps = max(int(np.ceil(N_l/batch_size)), int(np.ceil(N_u/batch_size)))
    history = {'loss': [], 'loss_sup': [], 'loss_unsup': [], 'val_loss': [], 'mask_ratio': []}
    best_val = float('inf')

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        il = np.random.permutation(N_l); iu = np.random.permutation(N_u)
        e_t, e_s, e_u, e_m = 0., 0., 0., 0.
        
        for step in tqdm(range(steps)):
            sl = (step % int(np.ceil(N_l/batch_size))) * batch_size
            su = (step % int(np.ceil(N_u/batch_size))) * batch_size
            lt, ls, lu, mr = train_step(
                tf.gather(lab_imgs, il[sl:sl+batch_size]), 
                tf.gather(lab_masks, il[sl:sl+batch_size]), 
                tf.gather(unl_imgs, iu[su:su+batch_size])
            )
            e_t+=lt; e_s+=ls; e_u+=lu; e_m+=mr
        
        # Validation
        vl = [val_step(val_imgs[i:i+batch_size], val_masks[i:i+batch_size]) for i in range(0, len(val_imgs), batch_size)]
        mv = np.mean(vl)
        
        history['val_loss'].append(mv)
        history['loss'].append(e_t.numpy()/steps)
        history['loss_sup'].append(e_s.numpy()/steps)
        history['loss_unsup'].append(e_u.numpy()/steps)
        history['mask_ratio'].append(e_m.numpy()/steps)
        
        print(f"Val: {mv:.4f} | Sup: {history['loss_sup'][-1]:.4f} | Unsup: {history['loss_unsup'][-1]:.4f} | Mask: {history['mask_ratio'][-1]:.2%}")
        
        if mv < best_val:
            best_val = mv
            fi = tf.keras.Input((h, w, c))
            tf.keras.Model(fi, decoder(encoder(fi))).save(filepath_h5, include_optimizer=False)
            
    return history


# ---------------------------------------------------------------------------------------
# TRAIN LOOP: Multiclass - FIXED RANK CHECK
# ---------------------------------------------------------------------------------------
def train_multiclass_unimatch_parts(
    encoder, decoder, filepath_h5, modelname, h, w, c, num_classes,
    epochs, batch_size, optimizer,
    train_labeled_images_dir, train_labeled_masks_dir,
    train_unlabeled_images_dir, val_images_dir, val_masks_dir,
    dataset_name_for_aug='SUIM',
    confidence_threshold=0.7, # LOWERED DEFAULT from 0.95
    lambda_u=0.5, mu_u=0.5
):
    import paths
    import numpy as np
    from functions import load_labeled_data_multiclass_mask, load_unlabeled_images
    
    lab_imgs_np, lab_1hot_np = load_labeled_data_multiclass_mask(train_labeled_images_dir, train_labeled_masks_dir, num_classes)
    unl_imgs_np = load_unlabeled_images(train_unlabeled_images_dir)
    val_imgs_np, val_1hot_np = load_labeled_data_multiclass_mask(val_images_dir, val_masks_dir, num_classes)
    lab_masks_np = np.argmax(lab_1hot_np, axis=-1).astype(np.int32)
    val_masks_np = np.argmax(val_1hot_np, axis=-1).astype(np.int32)

    lab_imgs = tf.convert_to_tensor(lab_imgs_np, tf.float32); lab_masks = tf.convert_to_tensor(lab_masks_np, tf.int32)
    unl_imgs = tf.convert_to_tensor(unl_imgs_np, tf.float32); val_imgs = tf.convert_to_tensor(val_imgs_np, tf.float32)
    val_masks = tf.convert_to_tensor(val_masks_np, tf.int32)

    dummy = tf.zeros([1, h, w, c], tf.float32)
    _ = decoder(encoder(dummy))
    all_vars = encoder.trainable_variables + decoder.trainable_variables
    _build_optimizer_weights(optimizer, all_vars)

    sup_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function(reduce_retracing=True)
    def train_step(x_lab, y_lab, x_unl):
        x_lab_aug, y_lab_aug = random_hflip_batch(x_lab, y_lab)
        y_lab_aug = tf.cond(tf.equal(tf.rank(y_lab_aug), 4), lambda: tf.squeeze(y_lab_aug, axis=-1), lambda: y_lab_aug)
        x_lab_w = weak_photometric_batch(x_lab_aug)
        
        x_unl_w, _ = random_hflip_batch(x_unl, None)
        x_unl_w = weak_photometric_batch(x_unl_w)
        
        # --- CHANGED: Use mild_strong_aug ---
        x_unl_s1 = mild_strong_aug(x_unl_w)
        x_unl_s2 = mild_strong_aug(x_unl_w)

        with tf.GradientTape() as tape:
            l_sup = sup_loss_fn(y_lab_aug, decoder(encoder(x_lab_w, training=True), training=True))
            
            p_w = tf.nn.softmax(decoder(encoder(x_unl_w, training=False), training=False), axis=-1)
            mask = tf.cast(tf.reduce_max(p_w, axis=-1, keepdims=True) >= confidence_threshold, tf.float32)
            mr = tf.reduce_mean(mask)
            
            p_s1 = tf.nn.softmax(decoder(encoder(x_unl_s1, training=True), training=True), axis=-1)
            p_s2 = tf.nn.softmax(decoder(encoder(x_unl_s2, training=True), training=True), axis=-1)
            
            # --- CHANGED: Reduced dropout to 0.2 ---
            f_p = encoder(x_unl_w, training=True)
            f_p = [tf.nn.dropout(f_p[0], rate=0.2)] + f_p[1:]
            p_fp = tf.nn.softmax(decoder(f_p, training=True), axis=-1)
            
            l_u = unimatch_loss(p_w, p_s1, p_s2, p_fp, mask, lambda_u, mu_u)
            scaled = optimizer.get_scaled_loss(l_sup + l_u)

        grads = tape.gradient(scaled, all_vars)
        grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, all_vars))
        return l_sup + l_u, l_sup, l_u, mr

    @tf.function
    def val_step(x, y):
        return sup_loss_fn(y, decoder(encoder(x, training=False), training=False))

    N_l = len(lab_imgs); N_u = len(unl_imgs)
    steps = max(int(np.ceil(N_l/batch_size)), int(np.ceil(N_u/batch_size)))
    history = {'val_loss': [], 'mask_ratio': []}
    best_val = float('inf')

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        il = np.random.permutation(N_l); iu = np.random.permutation(N_u)
        e_m = 0.
        for step in tqdm(range(steps)):
            sl = (step % int(np.ceil(N_l/batch_size))) * batch_size
            su = (step % int(np.ceil(N_u/batch_size))) * batch_size
            _, _, _, mr = train_step(tf.gather(lab_imgs, il[sl:sl+batch_size]), tf.gather(lab_masks, il[sl:sl+batch_size]), tf.gather(unl_imgs, iu[su:su+batch_size]))
            e_m += mr
        
        vl = [val_step(val_imgs[i:i+batch_size], val_masks[i:i+batch_size]) for i in range(0, len(val_imgs), batch_size)]
        mv = np.mean(vl)
        history['val_loss'].append(mv); history['mask_ratio'].append(e_m/steps)
        print(f"Val: {mv:.4f} | Mask: {e_m/steps:.2%}")
        
        if mv < best_val:
            best_val = mv
            fi = tf.keras.Input((h, w, c))
            tf.keras.Model(fi, decoder(encoder(fi))).save(filepath_h5, include_optimizer=False)
            
    return history














