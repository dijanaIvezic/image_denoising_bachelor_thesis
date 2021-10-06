import keras.datasets as datasets
import keras.callbacks as callbacks
from autoencoder import autoencoder
from functions import *

(train_clean_mnist, _), (test_clean_mnist, _) = datasets.mnist.load_data()
(train_clean_cifar, _), (test_clean_cifar, _) = datasets.cifar10.load_data()
train_clean_mnist = train_clean_mnist.astype('float32') / 255
test_clean_mnist = test_clean_mnist.astype('float32') / 255
train_clean_cifar = train_clean_cifar.astype('float32') / 255
test_clean_cifar = test_clean_cifar.astype('float32') / 255

train_clean_mnist = reshape_MNIST(train_clean_mnist)
test_clean_mnist = reshape_MNIST(test_clean_mnist)

sigma = 50
p = 0.05

train_noisy_cifar = add_noise(train_clean_cifar, sigma, p)
test_noisy_cifar = add_noise(test_clean_cifar, sigma, p)
train_noisy_mnist = add_noise(train_clean_mnist, sigma, p)
test_noisy_mnist = add_noise(test_clean_mnist, sigma, p)

model = autoencoder(32, 4)
model.compile(loss='mse', optimizer='adam')

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=4)
checkpoint = callbacks.ModelCheckpoint(filepath='model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

train_clean = np.concatenate((train_clean_cifar, train_clean_mnist))
train_noisy = np.concatenate((train_noisy_cifar, train_noisy_mnist))
test_clean = np.concatenate((test_clean_cifar, test_clean_mnist))
test_noisy = np.concatenate((test_noisy_cifar, test_noisy_mnist))

#history = model.fit(train_noisy, train_clean, validation_data=(test_noisy, test_clean), epochs=20, batch_size=150, shuffle=True, callbacks=[early_stopping, checkpoint])

model.load_weights('model.hdf5')

test_denoised_cifar = model.predict(test_noisy_cifar)
test_denoised_mnist = model.predict(test_noisy_mnist)

idx = 10
show_results([test_clean_cifar, test_noisy_cifar, test_denoised_cifar], idx, sigma, p)
show_results([test_clean_mnist, test_noisy_mnist, test_denoised_mnist], idx, sigma, p)
