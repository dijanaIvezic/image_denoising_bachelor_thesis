import keras.layers as layers
import keras.models as models
import keras.callbacks as callbacks

def convolution(k, n_filters, kernel_size=(3,3), strides=1):
    def apply_convolution(k):
        h = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides, padding='same')(k)
        h = layers.BatchNormalization()(h)
        h = layers.ReLU()(h)
        return h
    
    h = apply_convolution(k)
    h = apply_convolution(h)
    return h

def deconvolution(k, n_filters, kernel_size=(2,2), strides=2):
    h = layers.Conv2DTranspose(filters=n_filters, kernel_size=kernel_size, strides=strides, padding='same')(k)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    return h

def pooling(k, pool_size=(2,2), strides=2):
    h = layers.MaxPooling2D(pool_size=pool_size, strides=strides)(k)
    return h

def autoencoder(first_n_filters, n_layers, shape=(32,32,3)):
    input_layer = layers.Input(shape=shape, name='input')
    n_filters = first_n_filters
    h = input_layer
    down = []

    for n in range(n_layers):
        h = convolution(h, n_filters)
        down.append(h)
        h = pooling(h)
        n_filters = n_filters * 2

    h = convolution(h, n_filters)
    down.reverse()

    for d in down:
        n_filters = n_filters // 2
        h = deconvolution(h, n_filters)
        h = layers.concatenate([h,d])
        h = convolution(h, n_filters)
    
    output_layer = layers.Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(h)

    return models.Model(inputs=input_layer, outputs=output_layer, name="denoising_autoencoder")