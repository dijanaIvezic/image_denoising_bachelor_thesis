from slice_image import *
from functions import *
from autoencoder import autoencoder
from PIL import Image
import cv2

original = Image.open('i.jpg')
original_width, original_height = original.size
test_clean_image = slice_image(original, 32, 32)
test_clean_image = to_array(test_clean_image)

sigma = 30
p = 0.04

test_noisy_image = add_noise(test_clean_image, sigma, p)
test_noisy_image_Image = to_image(test_noisy_image)

complete_noisy_image = concatenate_image(test_noisy_image_Image, original_width, original_height)

model = autoencoder(32, 4)
model.compile(loss='mse', optimizer='adam')
model.load_weights('model.hdf5')

test_denoised = model.predict(test_noisy_image)
test_denoised_Image = to_image(test_denoised)

complete_denoised_image = concatenate_image(test_denoised_Image, original_width, original_height)
complete_denoised_image.save('denoised.png')
complete_noisy_image.save('noisy.png')

original = to_array([original])
noisy = to_array([complete_noisy_image])
denoised = to_array([complete_denoised_image])

show_results([original, noisy, denoised], 0, sigma, p)
