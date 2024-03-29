
def make_generator_model():
    #مدل بصورت جمله ای می باشد
    model = tf.keras.Sequential()
    #اضافه کردن لایه ها
    model.add(layers.Dense(16*16*1024, use_bias=False, input_shape=(NOISE_SIZE,)))
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.ReLU())   

    model.add(layers.Reshape((16, 16, 1024)))
    assert model.output_shape == (None, 16, 16, 1024) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                    kernel_initializer=KERNEL_INITIALIZER))
    assert model.output_shape == (None, 32, 32, 512)
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                    kernel_initializer=KERNEL_INITIALIZER))
    assert model.output_shape == (None, 64, 64, 256)
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                    kernel_initializer=KERNEL_INITIALIZER))
    assert model.output_shape == (None, 128, 128, 128)
    model.add(layers.BatchNormalization(epsilon=EPSILON))
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh',
                                    kernel_initializer=KERNEL_INITIALIZER))
    assert model.output_shape == (None, 256, 256, 3)

    return model
