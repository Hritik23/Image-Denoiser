import tensorflow as tf
def build_model(input_shape):
    i=tf.keras.Input(
        shape=input_shape,
        )

    x=tf.keras.layers.Conv2D(
        filters=32, kernel_size=(3,3), padding='same',
        activation='relu'
        )(i)

    x=tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), padding='same'
        )(x)

    x=tf.keras.layers.Conv2D(
        filters=32, kernel_size=(3,3), padding='same',
        activation='relu'
        )(x)

    encoded=tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), padding='same'
        )(x)

    x=tf.keras.layers.Conv2D(
        filters=32, kernel_size=(3,3), padding='same',
        activation='relu'
        )(encoded)

    x=tf.keras.layers.UpSampling2D(
        size=(2, 2)
        )(x)

    x=tf.keras.layers.Conv2D(
        filters=32, kernel_size=(3,3), padding='same',
        activation='relu'
        )(x)

    x=tf.keras.layers.UpSampling2D(
        size=(2, 2)
        )(x)

    decoded=tf.keras.layers.Conv2D(
        filters=1, kernel_size=(3,3), padding='same',
        activation='sigmoid'
        )(x)



    encoder = tf.keras.Model(inputs=i, outputs=encoded)
    autoencoder = tf.keras.Model(inputs=i, outputs=decoded)
    return (encoder, autoencoder)

