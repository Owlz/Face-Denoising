"""
This script build the model and trains it
"""

import os
import glob
import time

import cv2
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage import io

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Flatten, Reshape
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import models, layers


PRINT_CHARTS = False
COLOR_LAYERS = (3,) #RBG => 3, GRAYSCALE => 1
AS_GRAY = False
LIMIT = 10000

def testing_model(): #inspired by Zhao's "Image Denoising with Deep Convolutional Neural Networks"
    m = models.Sequential()
    
    m.add(Conv2D(64, (7, 7), input_shape=(64, 64, 1), padding='same'))
    m.add(MaxPooling2D(2, 2))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    
    m.add(Conv2D(64, (7, 7), padding='same'))
    m.add(MaxPooling2D(2, 2))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    
    m.add(Conv2D(128, (7, 7), padding='same'))
    m.add(MaxPooling2D(2, 2))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    
    m.add(Conv2D(256, (7, 7), padding='same'))
    m.add(MaxPooling2D(2, 2))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    
    m.add(Conv2D(518, (7, 7), padding='same'))
    m.add(UpSampling2D((2, 2)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    
    m.add(Conv2D(256, (7, 7), padding='same'))
    m.add(UpSampling2D((2, 2)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    
    m.add(Conv2D(128, (7, 7), padding='same'))
    m.add(UpSampling2D((2, 2)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    
    m.add(Conv2D(64, (7, 7), padding='same'))
    m.add(UpSampling2D((2, 2)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    
    m.add(Conv2D(64, (7, 7), padding='same'))
    
    m.add(Conv2D(1, (7, 7), padding='same'))

    m.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])

    return m


def DNCNN(img_shape=(64, 64, 1)): #best one yet
    input_img = Input(shape=img_shape)
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = Activation('relu')(x)

    for i in range(15):
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2D(1, (3, 3), padding='same')(x)
    output_img = Activation('tanh')(x)
    
    model = Model(input_img, output_img)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
    return model

def DNCNN_color(): #testing
    input_img = Input(shape=(64, 64, 3))
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = Activation('relu')(x)

    for i in range(15):
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2D(3, (3, 3), padding='same')(x)
    output_img = Activation('tanh')(x)
    
    model = Model(input_img, output_img)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
    return model

def DNCNN_bigger(): 
    input_img = Input(shape=(128, 128, 1))
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = Activation('relu')(x)

    for i in range(15):
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2D(1, (3, 3), padding='same')(x)
    output_img = Activation('tanh')(x)
    
    model = Model(input_img, output_img)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
    return model


def win5(): #way too good for black and white images
    input_img = Input(shape=(64, 64, 1))
    x = Conv2D(64, (7, 7), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    for i in range(5):
        x = Conv2D(128, (7, 7), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
    x = Conv2D(1, (7, 7), padding='same')(x)
    x = BatchNormalization()(x)
    output_img = layers.add([x, input_img])
    
    model = Model(input_img, output_img)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
    return model

def win5_color(): #our colored champions 3644431 param
    input_img = Input(shape=(64, 64, 3))
    x = Conv2D(64, (7, 7), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    for i in range(5):
        x = Conv2D(128, (7, 7), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
    x = Conv2D(3, (7, 7), padding='same')(x)
    x = BatchNormalization()(x)
    output_img = layers.add([x, input_img])
    
    model = Model(input_img, output_img)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
    return model


def load_dataset(folder="file dataset/img_celeba", limit=None, resize_data=(64,64)):
    print("Loading dataset, limit is", limit)
    if limit is None:
        limit = os.listdir(f"{folder}/cropped")


    img_shape = resize_data + COLOR_LAYERS

    a_shape = (limit,) + img_shape
    cropped = np.empty(shape=a_shape)
    noisy = np.empty(shape=a_shape)

    cropped_files = glob.glob(f"{folder}/cropped/*.jpg")[:limit]
    noisy_files = glob.glob(f"{folder}/noisy/*.jpg")[:limit]

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

clean, noisy = load_dataset(limit=LIMIT)
print("Dataset Loaded")
print(f"Shapes of the images laoded:\nClean {clean.shape},\nNoisy {noisy.shape}")


def print_charts(input_data, labels, filename, num=2, custom_cmap=None):
    if PRINT_CHARTS:
        fig, axs = plt.subplots(1, num, figsize=(12, 6), sharey=True, sharex=True)
        for i in range(num):
            axs[i].imshow(input_data[i], cmap=custom_cmap)
            axs[i].set_title(labels[i])

        fig.savefig('charts/' + filename + '.png', bbox_inches='tight', dpi=1000)


def print_hist(input_data, labels, filename, num=2, bins=255):
    if PRINT_CHARTS:
        fig, axs = plt.subplots(1, num, figsize=(12, 6), sharey=True, sharex=True)
        for i in range(num):
            axs[i].hist(input_data[i].ravel(), bins)
            axs[i].set_title(labels[i])

        fig.savefig('charts/' + filename + '.png', bbox_inches='tight', dpi=1000)


##clean_mean = np.mean(clean, axis=0).astype('float32')
##noisy_mean = np.mean(noisy, axis=0).astype('float32')
##print_charts([clean_mean, noisy_mean], ["clean", "noisy"], "mean_of_data")
##
##
##clean_std = np.std(clean, axis=0).astype('float32')
##noisy_std = np.std(noisy, axis=0).astype('float32')
##print_charts([clean_std, noisy_std], ["clean", "noisy"], "std_of_data")
##
##clean_mean_std = np.mean(clean_std, axis=2).astype('float32')
##noisy_mean_std = np.mean(noisy_std, axis=2).astype('float32')
##print_charts([clean_mean_std, noisy_mean_std], ["clean", "noisy"], "mean_std_of_data", custom_cmap='hot')
##
##clean_flat = clean.ravel()
##noisy_flat = noisy.ravel()
##
##
##print_hist([clean_flat, noisy_flat], ["clean", "noisy"], "hist_with_255bins")
##
##print_hist([clean[0], clean_mean, clean[0] - clean_mean],
##           ["first img distribution", "clean mean distribution", "(img - mean) distribution"],
##           "hist_with_20bins", num=3,bins=20)
##
##print_hist([clean[0] - clean_mean, clean_std, ((clean[0] - clean_mean) / clean_std)],
##           ["(img - mean) distribution", "std deviation distribution", "((img - mean) / std_dev) distribution"],
##           "distribution", num=3, bins=20)
##   
def print_acc(hist, filename):
    plt.clf()
    epochs = range(1, len(hist['acc'])+1)
    plt.plot(epochs, hist['acc'], 'b', label='Training acc')
    plt.plot(epochs, hist['val_acc'], 'r', label='Validation acc')
    plt.title('Training and validation Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig(filename + ".png")

def print_loss(hist, filename):
    plt.clf()
    epochs = range(1, len(hist['loss'])+1)
    plt.plot(epochs, hist['loss'], 'b', label='Training loss')
    plt.plot(epochs, hist['val_loss'], 'r', label='Validation loss')
    plt.title('Training and validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename + ".png")


model = win5_color()
model.summary()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
out = model.fit(noisy, clean, epochs=92222, batch_size=64, callbacks=[early_stopping], validation_split=0.2)
timestr = time.strftime("%Y_%m_%d_%H_%M_%S") 
model.save(f'{timestr}win5_color.h5')

print_acc(out.history, f"{timestr}_win5_color_acc")
print_loss(out.history, f"{timestr}_win5_color_loss")
