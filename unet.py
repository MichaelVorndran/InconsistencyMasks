import tensorflow as tf


def input_block(x, filters, actifu, kernel_ini):
    x = tf.keras.layers.Lambda(lambda x: x / 255)(x)
    x = tf.keras.layers.Conv2D(int(filters), (1, 1), activation=actifu,  kernel_initializer=kernel_ini, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x

def encoder_block(x, filters_c1, filters_c2, ks, actifu, kernel_ini, dropoutrate):
    x = tf.keras.layers.Conv2D(filters_c1, (ks, ks), activation=actifu,  kernel_initializer=kernel_ini, padding='same')(x)
    x = tf.keras.layers.Conv2D(filters_c2, (1, 1), activation=actifu, kernel_initializer=kernel_ini, padding='same')(x)
    if dropoutrate > 0:
        x = tf.keras.layers.Dropout(dropoutrate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    p = tf.keras.layers.MaxPooling2D((2,2))(x)

    return x, p


def bottleneck_block(x, filters_c1, filters_c2, ks, actifu, kernel_ini, dropoutrate):
    x = tf.keras.layers.Conv2D(filters_c1, (ks, ks), activation=actifu,  kernel_initializer=kernel_ini, padding='same')(x)
    x = tf.keras.layers.Conv2D(filters_c2, (1, 1), activation=actifu, kernel_initializer=kernel_ini, padding='same')(x)
    if dropoutrate > 0:
        x = tf.keras.layers.Dropout(dropoutrate)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x

def decoder_block(x, y, filters_c1, filters_c2, ks, actifu, kernel_ini, dropoutrate):
    u = tf.keras.layers.UpSampling2D(size=(2,2))(x)
    u = tf.keras.layers.add([u, y])
    c = tf.keras.layers.Conv2D(filters_c1, (1, 1), activation=actifu,  kernel_initializer=kernel_ini, padding='same')(u)
    c = tf.keras.layers.BatchNormalization()(c)
    
    c = tf.keras.layers.Conv2D(filters_c1, (ks, ks), activation=actifu,  kernel_initializer=kernel_ini, padding='same')(c)
    c = tf.keras.layers.Conv2D(filters_c2, (1, 1), activation=actifu, kernel_initializer=kernel_ini, padding='same')(c)
    if dropoutrate > 0:
        c = tf.keras.layers.Dropout(dropoutrate)(c)
    c = tf.keras.layers.BatchNormalization()(c)

    return c


def get_unet(i_height, i_width, i_channels, num_outputmasks, alpha, actifu, actifuout, ks=3, kernel_ini='he_normal', dropout_rate_encoder=0, dropout_rate_decoder=0, dropout_rate_bottleneck=0): 
    inputs = tf.keras.layers.Input(shape=(i_height, i_width, i_channels))

    c1 = input_block(inputs, int(16*alpha), actifu, kernel_ini)
    
    c1, p1 = encoder_block(c1, int(16*alpha), int(16*alpha), ks, actifu, kernel_ini, dropout_rate_encoder)
    c2, p2 = encoder_block(p1, int(32*alpha), int(32*alpha), ks, actifu, kernel_ini, dropout_rate_encoder)
    c3, p3 = encoder_block(p2, int(64*alpha), int(64*alpha), ks, actifu, kernel_ini, dropout_rate_encoder)
    c4, p4 = encoder_block(p3, int(128*alpha), int(128*alpha), ks, actifu, kernel_ini, dropout_rate_encoder)

    c5 = bottleneck_block(p4, int(256*alpha), int(128*alpha), ks, actifu, kernel_ini, dropout_rate_bottleneck)

    c6 = decoder_block(c5, c4, int(128*alpha), int(64*alpha), ks, actifu, kernel_ini, dropout_rate_decoder)
    c7 = decoder_block(c6, c3, int(64*alpha), int(32*alpha), ks, actifu, kernel_ini, dropout_rate_decoder)
    c8 = decoder_block(c7, c2, int(32*alpha), int(16*alpha), ks, actifu, kernel_ini, dropout_rate_decoder)
    c9 = decoder_block(c8, c1, int(16*alpha), int(16*alpha), ks, actifu, kernel_ini, dropout_rate_decoder)
     
    output = tf.keras.layers.Conv2D(num_outputmasks, (1, 1), activation=actifuout, kernel_initializer=kernel_ini, name='out', dtype='float32')(c9)

    model = tf.keras.Model(inputs, output)
    
    return model
