from slice_image import *
from functions import *
from autoencoder import autoencoder
import cv2

large = cv2.imread('i.jpg')
small = cv2.resize(large, (32, 32))
cv2.imwrite('j.jpg', small)

test_clean_image = [Image.open('j.jpg')]
test_clean_image = to_array(test_clean_image)

sigma = 30
p = 0.04

test_noisy_image = add_noise(test_clean_image, sigma, p)

model = autoencoder(32, 4)
model.compile(loss='mse', optimizer='adam')
model.load_weights('model.hdf5')

test_denoised = model.predict(test_noisy_image)

show_results([test_clean_image, test_noisy_image, test_denoised], 0, sigma, p)