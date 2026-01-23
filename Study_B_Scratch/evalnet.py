import tensorflow as tf


def input_block(x, filters, actifu, kernel_ini, normalize=True):
    if normalize:
        x = tf.keras.layers.Lambda(lambda x: x / 255)(x)

    x = tf.keras.layers.Conv2D(int(filters), (1, 1), activation=actifu,  kernel_initializer=kernel_ini, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x


def conv_block(x, filters, ks, actifu, kernel_ini, mp=True):
    x = tf.keras.layers.Conv2D(filters, (ks, ks), activation=actifu,  kernel_initializer=kernel_ini, padding='same')(x)
    x = tf.keras.layers.Conv2D(filters, (1, 1), activation=actifu, kernel_initializer=kernel_ini, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if mp == True:
        x = tf.keras.layers.MaxPooling2D((2,2))(x)

    return x


def get_evalnet(i_height, i_width, inputA_channels, inputB_channels, alpha=2, actifu='relu', ksi=3, kernel_ini='he_normal', normalize_A=True, normalize_B=True): 
    inputA = tf.keras.layers.Input(shape=(i_height, i_width, inputA_channels))
    inputB = tf.keras.layers.Input(shape=(i_height, i_width, inputB_channels))

    a = input_block(inputA, int(16*alpha), actifu, kernel_ini, normalize=normalize_A)
    a = conv_block(a, int(16*alpha), ksi, actifu, kernel_ini)
    b = input_block(inputB, int(16*alpha), actifu, kernel_ini, normalize=normalize_B)
    b = conv_block(b, int(16*alpha), ksi, actifu, kernel_ini)

    x = tf.keras.Model(inputs=inputA, outputs=a)
    y = tf.keras.Model(inputs=inputB, outputs=b)

    c = tf.keras.layers.concatenate([x.output, y.output])

    c = conv_block(c, int(16*alpha), ksi, actifu, kernel_ini)
    c = conv_block(c, int(32*alpha), ksi, actifu, kernel_ini)
    c = conv_block(c, int(64*alpha), ksi, actifu, kernel_ini)
    c = conv_block(c, int(128*alpha), ksi, actifu, kernel_ini)
    c = conv_block(c, int(256*alpha), ksi, actifu, kernel_ini)

    c = tf.keras.layers.GlobalAvgPool2D()(c)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(c)

    return tf.keras.Model(inputs=[x.input, y.input], outputs=output)

def get_evalnet_miou(i_height, i_width, inputA_channels, inputB_channels, alpha=2, actifu='relu', ksi=3, kernel_ini='he_normal', normalize_A=True, normalize_B=False): 
    inputA = tf.keras.layers.Input(shape=(i_height, i_width, inputA_channels))
    inputB = tf.keras.layers.Input(shape=(i_height, i_width, inputB_channels))

    a = input_block(inputA, int(16*alpha), actifu, kernel_ini, normalize=normalize_A)
    a = conv_block(a, int(16*alpha), ksi, actifu, kernel_ini)
    b = input_block(inputB, int(16*alpha), actifu, kernel_ini, normalize=normalize_B)
    b = conv_block(b, int(16*alpha), ksi, actifu, kernel_ini)

    x = tf.keras.Model(inputs=inputA, outputs=a)
    y = tf.keras.Model(inputs=inputB, outputs=b)

    c = tf.keras.layers.concatenate([x.output, y.output])

    c = conv_block(c, int(16*alpha), ksi, actifu, kernel_ini)
    c = conv_block(c, int(32*alpha), ksi, actifu, kernel_ini)
    c = conv_block(c, int(64*alpha), ksi, actifu, kernel_ini)
    c = conv_block(c, int(128*alpha), ksi, actifu, kernel_ini)
    c = conv_block(c, int(256*alpha), ksi, actifu, kernel_ini)

    c = tf.keras.layers.GlobalAvgPool2D()(c)
    o_iou = tf.keras.layers.Dense(inputB_channels, activation='sigmoid', name='iou')(c)
    o_conf = tf.keras.layers.Dense(inputB_channels, activation='sigmoid', name='detection')(c)

    return tf.keras.Model(inputs=[x.input, y.input], outputs=[o_iou, o_conf])


def get_evalnet_miou_v2(i_height, i_width, inputA_channels, inputB_channels, alpha=2, actifu='relu', ksi=3, kernel_ini='he_normal', normalize_A=True, normalize_B=False): 
    inputA = tf.keras.layers.Input(shape=(i_height, i_width, inputA_channels))
    inputB = tf.keras.layers.Input(shape=(i_height, i_width, inputB_channels))

    a = input_block(inputA, int(16*alpha), actifu, kernel_ini, normalize=normalize_A)
    a = conv_block(a, int(16*alpha), ksi, actifu, kernel_ini)
    a = conv_block(a, int(32*alpha), ksi, actifu, kernel_ini)
    a = conv_block(a, int(64*alpha), ksi, actifu, kernel_ini)
    a = conv_block(a, int(128*alpha), ksi, actifu, kernel_ini)

    b = input_block(inputB, int(16*alpha), actifu, kernel_ini, normalize=normalize_B)
    b = conv_block(b, int(16*alpha), ksi, actifu, kernel_ini)
    b = conv_block(b, int(32*alpha), ksi, actifu, kernel_ini)
    b = conv_block(b, int(64*alpha), ksi, actifu, kernel_ini)
    b = conv_block(b, int(128*alpha), ksi, actifu, kernel_ini)

    x = tf.keras.Model(inputs=inputA, outputs=a)
    y = tf.keras.Model(inputs=inputB, outputs=b)

    #c = tf.keras.layers.concatenate([x.output, y.output])
    c = tf.keras.layers.add([x.output, y.output])

    c = conv_block(c, int(64*alpha), ksi, actifu, kernel_ini)
    c = conv_block(c, int(128*alpha), ksi, actifu, kernel_ini)
    c = conv_block(c, int(256*alpha), ksi, actifu, kernel_ini)

    c = tf.keras.layers.GlobalAvgPool2D()(c)
    o_iou = tf.keras.layers.Dense(inputB_channels, activation='sigmoid', name='iou')(c)
    o_conf = tf.keras.layers.Dense(inputB_channels, activation='sigmoid', name='detection')(c)

    return tf.keras.Model(inputs=[x.input, y.input], outputs=[o_iou, o_conf])
