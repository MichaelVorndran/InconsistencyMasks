
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import io
import contextlib
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import configparser
import random
import shutil
import gc
import csv
import math
import pandas as pd
from tqdm import tqdm

config = configparser.ConfigParser()
config.read(os.path.join('config.ini'))


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
        loss = self.cce(y_true, y_pred)
        
        # Create a mask to remove the loss for class 0
        mask = 1. - y_true[:, 0]
        loss *= mask
            
        return tf.reduce_mean(loss)



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
    mean_cell_count_error = round(cell_count_delta / len(mIoUs), 3)

    return mIoU, mIoU_ad, mean_cell_count_error




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
    Create pseudo labels for ISIC 2018 dataset images using IM predictions.

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
