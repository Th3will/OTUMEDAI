import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import KFold
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import segmentation_models as sm

from sklearn.model_selection import KFold
from skimage.transform import rescale, resize
from scipy.ndimage import rotate
print(os.listdir("../input/"))
import tensorflow as tf
from tensorflow.keras import layers

input_path = input("please enter the path to the image: ")

checkpoint_path = "./Checkpoint/"
checkpoint_dir = os.path.dirname(checkpoint_path)

my_model = "efficientnetb1"

model = sm.Unet(my_model, encoder_weights='imagenet', input_shape=( 256,256, 3), classes=3, activation='sigmoid')
model.load_weights(checkpoint_path)
model.compile(optimizer='Adam') # changed from original
x = cv2.imread(input_path, cv2.IMREAD_COLOR)
x = cv2.resize(x, (256,256), interpolation= cv2.INTER_LINEAR)
original_shape = x.shape
y_pred = model.predict(np.expand_dims(x,0))
_,y_pred_thr = cv2.threshold(y_pred[0,:,:,0]*255, 127, 255, cv2.THRESH_BINARY)
y_pred = (y_pred_thr/255).astype(int)
y_pred_original = cv2.resize(y_pred.astype(float), (original_shape[1],original_shape[0]), interpolation= cv2.INTER_LINEAR)
plt.imshow(x, 'gray', interpolation='none')
plt.imshow(y_pred_original, 'jet', interpolation='none', alpha=0.4)
plt.savefig("/home/wni1717/dev/OTUMEDAI/OTUAILIB/models/Segmentation/output/test.png", format = "png")