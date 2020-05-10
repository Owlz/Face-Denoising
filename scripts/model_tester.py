"""
This script test the model made by the script "model_trainer"
The steps are:
1) We load the model and some pictures
2) Denoise the pictures
3) Render an image of the results
"""

import keras
import sys
import h5py
import numpy as np
from skimage.transform import resize
from skimage import io
import matplotlib.pyplot as plt
import glob
import time
from skimage import img_as_ubyte


COLOR_LAYERS = (3,)
AS_GRAY=False
FILENAME = "2018_12_19_14_41_55win5_color"

def load_test_data(folder="file dataset/img_celeba", limit=5000, resize_data=(64, 64)):
    img_shape = resize_data + COLOR_LAYERS

    a_shape = (limit,) + img_shape
    cropped = np.empty(shape=a_shape)
    noisy = np.empty(shape=a_shape)

    cropped_files = glob.glob(f"{folder}/cropped/*.jpg")[30000:30000+limit]
    noisy_files = glob.glob(f"{folder}/noisy/*.jpg")[30000:30000+limit]

    total_size = len(cropped_files)
    start = time.time()
    for i, (crop, noise) in enumerate(zip(cropped_files, noisy_files)):
        image = io.imread(crop, as_gray=AS_GRAY)
        cropped[i] = resize(image, img_shape, mode="constant")

        image = io.imread(noise, as_gray=AS_GRAY)
        noisy[i] = resize(image, img_shape, mode="constant")

        print(f"{(((i+1) * 100) / total_size):0.4f}", end='\r', flush=True)


    print(f"{(((i+1) * 100) / total_size):0.4f} - Total loading time: {(time.time() - start):4.2f}", end='\n', flush=True)
    return (cropped, noisy)

model = keras.models.load_model(f"{FILENAME}.h5")

c_test, n_test = load_test_data(limit=500)
print(model.evaluate(c_test, n_test))


def showOrigDec(orig, noise, denoise, num=1):
    import matplotlib.pyplot as plt
    n = num
    plt.figure(figsize=(20, 6))

    for i in range(n):
        # display original
        size = (64, 64)
        color_map=None
        if not AS_GRAY:
            color_map="gray"
            size += COLOR_LAYERS
            
        ax = plt.subplot(3, n, i+1)
        plt.imshow(orig[i+7].reshape(size), cmap=color_map)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy image
        ax = plt.subplot(3, n, i +1 + n)
        plt.imshow(noise[i+7].reshape(size), cmap=color_map)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display denoised image
        ax = plt.subplot(3, n, i +1 + n + n)
        plt.imshow(denoise[i+7].reshape(size), cmap=color_map)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(f'{FILENAME}.png', bbox_inches='tight', dpi=1200)

##image = io.imread("test.jpg", as_gray=True)
##image = resize(image, (64,64,1), mode="constant")
out = model.predict(n_test)

showOrigDec(c_test, n_test, out)

##size = (64, 64)
##io.imsave("output.jpg", out[150].reshape(size))
##io.imsave("input_orig.jpg", c_test[150].reshape(size))
##
##io.imsave("output_noisy.jpg", n_test[150].reshape(size))
