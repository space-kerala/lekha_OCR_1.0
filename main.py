# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:25:26 2015

Project Malayalam OCR - Lekha
-----------------------------
Sponsored by ICFOSS

contributers : james, jithin
"""
import cv2
import sys
import preprocess as pp
import training as train
import random

url = sys.argv[1] #geting file url from command prompt
img=cv2.imread(url,0)
if(img.data==None):
	print url+' does\'nt exist'
	exit()

img = pp.preprocess(img)
im=img
im,rot = pp.skew_correction(img)
# Add Layout analysis here
print pp.recognize_block(im)
