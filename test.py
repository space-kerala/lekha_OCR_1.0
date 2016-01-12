from __future__ import division
import cv2
import preprocess as pp
import numpy as np
import os
import glob
import shutil
import itertools
from random import shuffle
from matplotlib import pyplot as plt

img=cv2.imread('samp/5.png',0)
print img.shape
f=np.fft.fft2(img)
fshift=np.fft.fftshift(f)
print fshift.shape
print fshift
magnitude_spectrum=20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
