import numpy as np
import matplotlib.pyplot as plt
import cv2

def reshape_MNIST(mnist):
    l,_,_ = mnist.shape
    bordered = np.ndarray((l,32,32), 'float32')
    for i in range(l):
        bordered[i] = cv2.copyMakeBorder(src=mnist[i], top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_CONSTANT, value=0)
    reshaped = bordered.reshape((bordered.shape + (1,)))
    reshaped = reshaped.repeat(3,-1)
    return reshaped

def add_noise(dataset, noise_level_gauss = 50, noise_level_sp = 0.05, sp_distribution = 0.5):
    #gaussian noise
    noisy = dataset + np.random.normal(0, noise_level_gauss, dataset.shape) / 255

    #salt and pepper noise
    width, height, _ = dataset[0].shape
    size = width * height
    n_s = int(np.ceil(noise_level_sp * size * sp_distribution))
    n_p = int(np.ceil(noise_level_sp * size * (1-sp_distribution)))

    for picture in noisy:
        coordinates_s = [np.random.randint(0, i-1, n_s) for i in (width, height)]
        coordinates_p = [np.random.randint(0, i-1, n_p) for i in (width, height)]

        picture[tuple(coordinates_s)] = [1., 1., 1.]
        picture[tuple(coordinates_p)] = [0., 0., 0.]

    np.clip(noisy, 0., 1., noisy)
    noisy = noisy.astype('float32')

    return noisy

def show_results(dataset, idx, sigma, p):
    cols = len(dataset)
    dataset = [dataset[i][idx] for i in range(cols)]
    txt = "sigma={}, p={}, PSNR={} dB".format(sigma, p, round(cv2.PSNR(dataset[0], dataset[cols-1], 1.0), 2))
    print(round(cv2.PSNR(dataset[0], dataset[1], 1.0), 2))
    _, ax = plt.subplots(2,cols)
    for i in range(cols):
        ax[0,i].imshow(dataset[i])
        ax[0,i].set_axis_off()
        ax[1,i].set_axis_off()
    ax[1,0].text(0,1,txt, fontsize=14)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.1, hspace=0.1)
    plt.show()
