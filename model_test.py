import tensorflow as tf
import train_argument as args

def encoder(previous):
    # 입력 = None, 5, 64, 64, 3 == Batch, Time, Height, Width, Channel
    conv1 = tf.keras.layers.Conv3D(filters=16, kernel_size=(3,3,3), padding = 'same', activation = None)(previous)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)
    maxpool1 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv1)

    conv2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding = 'same', activation = None)(maxpool1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)
    maxpool2 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv2)

    conv3 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding = 'same', activation = None)(maxpool2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)
    temporal = tf.keras.layers.MaxPool3D(pool_size=(conv3.shape[1],2,2), strides=2)(conv3) # temporal_max_pooling

    encoded = tf.keras.layers.Reshape(target_shape=(8,8,32))(temporal)

    return encoded

def decoder_middle_frame(previous):
    
    upsample1 = tf.keras.layers.Resizing(height = previous.shape[1]*2, width = previous.shape[2] * 2, interpolation='nearest')(previous)

    conv5 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation=None)(upsample1)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.ReLU()(conv5)

    upsample2 = tf.keras.layers.Resizing(height = conv5.shape[1]*2, width = conv5.shape[2] * 2, interpolation='nearest')(conv5)

    conv6 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation=None)(upsample2)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.ReLU()(conv6)

    upsample3 = tf.keras.layers.Resizing(height = conv6.shape[1]*2, width = conv6.shape[2] * 2, interpolation='nearest')(conv6)

    conv7 = tf.keras.layers.Conv2D(filters = 3, kernel_size = (3,3), padding = 'same', activation=None)(upsample3)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.ReLU()(conv7)

    return conv7

def decoder_fwd_bwd(previous):
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=None)(previous)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2, activation=None)(x)

    return x

def model():
    inputs_shape = [args.model_input]
    inputs_consecutive_shape = [args.model_consecutive]
    inputs_resnet_shape = [args.model_resnet]
    for i in args.input_size:
        inputs_shape.append(i)
        inputs_consecutive_shape.append(i)
        inputs_resnet_shape.append(i)
    
    # 입력 모양 3개
    inputs_ = tf.keras.Input(inputs_shape)
    inputs_consecutive = tf.keras.Input(inputs_consecutive_shape)
    inputs_resnet = tf.keras.Input(inputs_resnet_shape)

    encoded = encoder(inputs_)
    decoder_middle_ = decoder_middle_frame(encoded)
    logits_fwd_bwd = decoder_fwd_bwd(encoded)

    model = tf.keras.Model(inputs = inputs_, outputs=[decoder_middle_, logits_fwd_bwd])

    return model


a = model()
a.summary()