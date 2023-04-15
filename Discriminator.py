#ساخت مدل جداساز بصورت جمله ای و طبق تصویر مستندات

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[256, 256, 3],
                                    kernel_initializer=KERNEL_INITIALIZER))
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU(alpha=LEAK_RELU_APLPHA))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                    kernel_initializer=KERNEL_INITIALIZER))
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU(alpha=LEAK_RELU_APLPHA))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                                    kernel_initializer=KERNEL_INITIALIZER))
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.LeakyReLU(alpha=LEAK_RELU_APLPHA))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
