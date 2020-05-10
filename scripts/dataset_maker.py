"""
This script build the dataset of for the model training
The steps are:
1) Take a picture from the celebA dataset
2) Crop the face
3) Make the cropped area noisy
4) Save the picture
"""

import math
import cv2
import dlib
from PIL import Image
import os
from signal import signal, SIGABRT, SIGINT, SIGTERM
import sys
from datetime import datetime
from skimage.util import random_noise
from skimage import img_as_float, img_as_ubyte

from threading import Thread

os.chdir("file dataset\img_align_celeba\noisy")

cropped_folder = "cropped"
noisy_folder = "noisy"
MODEL_5_LANDMARK = "../shape_predictor_5_face_landmarks.dat"

if not os.path.exists(cropped_folder):
    os.makedirs(cropped_folder)

if not os.path.exists(noisy_folder):
    os.makedirs(noisy_folder)


def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1),
                             max(y2 - img.shape[0], 0),
                             -min(0, x1),
                             max(x2 - img.shape[1], 0),
                             cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

class DenoisingThread(Thread):
    def __init__(self, filename):
        super().__init__()
        self.fn = filename

    def run(self):
        img_in = dlib.load_rgb_image(self.fn)
        dets = detector(img_in, 1)

        if len(dets) == 0:
            return

        shape = shape_preditor(img_in, dets[0])

        points = []

        for i in range(0, shape.num_parts):
            point = shape.part(i)
            points.append((point.x, point.y))

        eye_sx = points[1]
        eye_dx = points[3]

        dy = eye_dx[1] - eye_sx[1]
        dx = eye_dx[0] - eye_sx[0]
        angle = math.degrees(math.atan2(dy, dx))

        center = (dets[0].center().x, dets[0].center().y)
        h, w = img_in.shape[:2]
        M = cv2.getRotationMatrix2D(center, angle + 180, 1)
        img_in = cv2.warpAffine(img_in, M, (w, h))

        dets = detector(img_in, 1)
        if len(dets) == 0:
            return
        
        bbox = (dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom())
        img_out = cv2.resize(imcrop(img_in, bbox), (128, 128))
        
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{cropped_folder}/{self.fn}", img_out)

        img_out = cv2.resize(img_out, (64, 64))
        img_out = img_as_float(img_out)
        noisy_image = random_noise(img_out, mode="speckle", clip=True, var=0.005)

        noisy_image = img_as_ubyte(noisy_image)
        noisy_image = cv2.resize(noisy_image, (128, 128))
        
        cv2.imwrite(f"{noisy_folder}/{self.fn}", noisy_image)
        print("Thread done -", self.fn) 

    
detector = dlib.get_frontal_face_detector()
shape_preditor = dlib.shape_predictor(MODEL_5_LANDMARK)

for f in sorted(os.listdir())[0:]:
    if os.path.isdir(f):
        continue

    DenoisingThread(f).start()


print("end")
