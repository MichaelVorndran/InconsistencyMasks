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



# Add to unet.py

def get_unet_unimatch_parts(i_height, i_width, i_channels, num_outputmasks, alpha, actifu, actifuout, ks=3, kernel_ini='he_normal'): 
    """
    Returns (Encoder, Decoder) models for UniMatch.
    Encoder: Input -> [Bottleneck Features, Skip4, Skip3, Skip2, Skip1]
    Decoder: [Bottleneck, Skip4, Skip3, Skip2, Skip1] -> Output
    """
    # --- Encoder ---
    inputs = tf.keras.layers.Input(shape=(i_height, i_width, i_channels), name='input_img')

    c1 = input_block(inputs, int(16*alpha), actifu, kernel_ini)
    
    c1, p1 = encoder_block(c1, int(16*alpha), int(16*alpha), ks, actifu, kernel_ini, 0)
    c2, p2 = encoder_block(p1, int(32*alpha), int(32*alpha), ks, actifu, kernel_ini, 0)
    c3, p3 = encoder_block(p2, int(64*alpha), int(64*alpha), ks, actifu, kernel_ini, 0)
    c4, p4 = encoder_block(p3, int(128*alpha), int(128*alpha), ks, actifu, kernel_ini, 0)

    # Bottleneck
    c5 = bottleneck_block(p4, int(256*alpha), int(128*alpha), ks, actifu, kernel_ini, 0)
    
    # Encoder outputs the bottleneck AND all skip connections
    encoder_model = tf.keras.Model(inputs, [c5, c4, c3, c2, c1], name='encoder')

    # --- Decoder ---
    # Input shapes based on encoder outputs
    f5_shape = c5.shape[1:]
    f4_shape = c4.shape[1:]
    f3_shape = c3.shape[1:]
    f2_shape = c2.shape[1:]
    f1_shape = c1.shape[1:]

    inp_c5 = tf.keras.layers.Input(shape=f5_shape, name='feat_c5')
    inp_c4 = tf.keras.layers.Input(shape=f4_shape, name='feat_c4')
    inp_c3 = tf.keras.layers.Input(shape=f3_shape, name='feat_c3')
    inp_c2 = tf.keras.layers.Input(shape=f2_shape, name='feat_c2')
    inp_c1 = tf.keras.layers.Input(shape=f1_shape, name='feat_c1')

    # U-Net Skip Connections
    c6 = decoder_block(inp_c5, inp_c4, int(128*alpha), int(64*alpha), ks, actifu, kernel_ini, 0)
    c7 = decoder_block(c6, inp_c3, int(64*alpha), int(32*alpha), ks, actifu, kernel_ini, 0)
    c8 = decoder_block(c7, inp_c2, int(32*alpha), int(16*alpha), ks, actifu, kernel_ini, 0)
    c9 = decoder_block(c8, inp_c1, int(16*alpha), int(16*alpha), ks, actifu, kernel_ini, 0)
     
    output = tf.keras.layers.Conv2D(num_outputmasks, (1, 1), activation=actifuout, kernel_initializer=kernel_ini, name='out', dtype='float32')(c9)

    # Decoder takes a list of features [bottleneck, skip4, ..., skip1]
    decoder_model = tf.keras.Model(inputs=[inp_c5, inp_c4, inp_c3, inp_c2, inp_c1], outputs=output, name='decoder')
    
    return encoder_model, decoder_model

