# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:26:15 2015

@author: james
"""
#feature extraction and Recognition
from __future__ import division
import cv2
import preprocess as pp
import numpy as np
import os
import glob
import shutil
import itertools
from random import shuffle
from path import *

classifier = cv2.SVM()
classifier.load(PATH_TO_MAIN+'svm_class.xml')

def find_feature(char):
	# htow_ratio(char)
	# q=pixel_intensity(char)
	# print q
	# return zonewise_hu5(char)+feature_hu2(char)
	# return feature_hu2(char)
	# print len(hog(char))
	# im=preprocess(char)
	# return zonewise_hu5(char)+[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]+hog(char)
	# return hog(char)
	# return zonewise_hu5(char)+zonewise_hu3(char)
	# find_vlines(char.copy())
	# print len(zonewise_hu5(char)+htow_ratio(char)+q)
	return zonewise_hu5(char)+htow_ratio(char)+find_blobs(char)

	# return zonewise_hu5(char)+zonewise_hu3(char)

def find_vlines(img):
	edges=cv2.Canny(img,50,150,apertureSize=3)
	minLineLength=10
	maxLineGap=15
	lines=cv2.HoughLinesP(edges,1,np.pi,10,minLineLength,maxLineGap)
	try:
		for x1,y1,x2,y2 in lines[0]:
			cv2.line(img,(x1,y1),(x2,y2),0,2)
		print len(lines[0])
	except:
		print 0
	
	cv2.imshow('ske',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def preprocess(img):
	cv2.imwrite('before_pp_thresholding.png',img)
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,243,50)
	cv2.imwrite('after_pp_thresholding.png',img)
	return img

def htow_ratio(im):
	h,w=im.shape
	q=0
	for i in range(h):
		for j in range(w):
			if im.item(i,j)==255:
				q+=1
	# print [h/w,(q/(h*w))]
	return [h/w,(q/(h*w))]
# def pixel_intensity(im):

def find_blobs(im):
	params=cv2.SimpleBlobDetector_Params()
	params.filterByArea=True
	params.minArea=10
	params.filterByConvexity=True
	params.minConvexity=0.87
	detector=cv2.SimpleBlobDetector(params)
	keypoints=detector.detect(im)
	# print len(keypoints)
	return [len(keypoints)]
	
def recognize(feature):
	a = classifier.predict(feature)
	# print (label_uni[int(a)]=='ഠ')
	# if(pp.previous_char==None):
	# 	print 'none in prev'
	if (label_uni[int(a)]=='0' or label_uni[int(a)]=='ഠ'):
		# print pp.previous_char
		if(pp.previous_char==None):
			return label_uni.index('0')
		if(label_uni[int(pp.previous_char.label)].isdigit()):
			return label_uni.index('0')
		# print pp.previous_char.height,pp.cur_char.height
		if(pp.cur_char.height<=(pp.previous_char.height*3/4)):
			return label_uni.index('ം')
		# print '0'
		return label_uni.index('ഠ')
# ===========================================my additions
	# if (label_uni[int(a)]=='\'' or label_uni[int(a)]==','):
	# 	# print pp.previous_char
	# 	if(pp.previous_char==None):
	# 		return label_uni.index('\'')
		
	# 	if(pp.cur_char.height<=(pp.previous_char.height*3/4)):
	# 		return label_uni.index(',')
	# 	# print '0'
	# 	return label_uni.index('\'')
# =====================================
	# if (pp.previous_char!= None):
	# 	if (label_uni[int(pp.previous_char.label)]=='ം'):
	# 		if(pp.cur_char.height*3/4<pp.previous_char.height):
	# 			pp.previous_char.label= label_uni.index('ഠ')
	# 			print 'ഠ'

	return a
label_uni = []
f = open(PATH_TO_MAIN+'/label','r')
for l in f:
	label_uni.append(l[:-1])
# label_uni.append('0')
def label_unicode():
# 	url='../samples/train_images/'
# 	for i in range(101,208):
# 		file=open(url+str(i)+'/utf8',"r")
# 		i_uni=file.read()
# 		i_uni=i_uni[:-1]
# 		label_uni.append(i_uni)
	return label_uni
# label_unicode()
label_uni.append('ം')
def hog(img):
	bin_n =16
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
#	print len(mag)
	# quantizing binvalues in (0...16)
	bins = np.int32(bin_n*ang/(2*np.pi))
	cut= len(bins)/2
	
	# Divide to 4 sub-squares
	bin_cells = bins[:cut,:cut], bins[cut:,:cut], bins[:cut,cut:], bins[cut:,cut:]
	mag_cells = mag[:cut,:cut], mag[cut:,:cut], mag[:cut,cut:], mag[cut:,cut:]
	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)
	return hist
def feature_hu2(img):
	contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	moments = [0,0,0,0,0,0,0,0,0,0,0,0]
	if(len(contours)==0):
		return moments
	X = [cv2.contourArea(C) for C in contours]
	t=[i for i in range (0,len(contours))]
	X,t = zip(*sorted(zip(X,t),reverse=True))
	list = []
	for i in range (0,2):
		try:
 #			print (i)
			cnt = contours[i]
			if(cv2.contourArea(cnt)<4):
				[list.append(0.0) for j in range(0,6)]
				continue
			mom = cv2.HuMoments(cv2.moments(cnt))
			moments=mom[:-1]
			[list.append(m[0]) for m in moments]
		except IndexError:
			[list.append(0.0) for j in range(0,6)]
	return list


#jithin-additions

def zonewise_hu2(img):
	# global ter
	contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	X = [cv2.contourArea(C) for C in contours]
	t=[i for i in range (0,len(contours))]
	X,t = zip(*sorted(zip(X,t),reverse=True))
	cnt = contours[t[0]]
	x,y,w,h=cv2.boundingRect(cnt)
	im = img[y-1:y+h+1,x-1:x+w+1]
	# cv2.imwrite('zq2nd'+str(ter)+'.png',im)
	height,width=im.shape
	box = img[0:1,0:1]
	box[0,0]=0
	box = cv2.resize(box,(width,height))
	img4=[]
	[img4.append(box.copy())for i in range(0,4)]
	i=0
	for i in range (0,height):
		j=(int)(i*width/height)
		for k in range(0,width):
			if(k<j):
				img4[0][i,k]=im[i,k]
				img4[0][height-i-1,k]=im[height-i-1,k]
			elif(k>width-j):
				img4[2][i,k]=im[i,k]
				img4[2][height-i-1,k]=im[height-i-1,k]
			else:	
				img4[1][i,k]=im[i,k]
				img4[3][height-i-1,k]=im[height-i-1,k]
		if (j>width/2):
			break
	# i=0
	# for img in img4:
	# 	cv2.imwrite('zq2nd'+str(ter)+'_'+str(i)+'.png',img)
	# 	i+=1
	# ter+=1
	feature = []
	for img in img4:
		feature = feature+list(itertools.chain(feature_hu2(img)))
	return feature

def zonewise_hu3(img):
	# global ter
	contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	X = [cv2.contourArea(C) for C in contours]
	t=[i for i in range (0,len(contours))]
	X,t = zip(*sorted(zip(X,t),reverse=True))
	cnt = contours[t[0]]
	x,y,w,h=cv2.boundingRect(cnt)


	im = img[y-1:y+h+1,x-1:x+w+1]

	# cv2.imshow("img",im)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# cv2.imwrite('./samples/samp'+str(i)+'.png',im)



	height,width=img.shape
	M = cv2.moments(cnt)
	try:
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
	except:
		return [0]*48
	img4=[]
	img4.append(img[0:cy,0:cx])
	img4.append(img[0:cy,cx:width])
	img4.append(img[cy:height,0:cx])
	img4.append(img[cy:height,cx:width])
	i=0
 #	for img in img4:
 #		cv2.imwrite('zq'+str(ter)+str(i)+'.png',img)
 #		i+=1
 #	ter+=1
	feature = []
	for img in img4:
		feature = feature+list(itertools.chain(feature_hu2(img)))
		# print str(len(feature))+'hai'
	return feature

def zonewise_hu5(img):#diagonal with more contours
	global ter
	contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	X = [cv2.contourArea(C) for C in contours]
	t=[i for i in range (0,len(contours))]
	try:
		X,t = zip(*sorted(zip(X,t),reverse=True))
	except ValueError:
		cv2.imwrite('error.png',img)
		print 'no countours'
		exit
	cnt = contours[t[0]]
	x,y,w,h=cv2.boundingRect(cnt)
	for i in range(x,x+w):
		for j in range(y,y+h):
			if(cv2.pointPolygonTest(cnt,(i,j),False)==-1):
				img[j,i]=0
	im = img[y-1:y+h+1,x-1:x+w+1]
	height,width=im.shape
	box = img[0:1,0:1]
	box[0,0]=0
	box = cv2.resize(box,(width,height))
	img4=[]
	[img4.append(box.copy())for i in range(0,4)]
	i=0
	for i in range (0,height):
		j=(int)(i*width/height)
		for k in range(0,width):
			if(k<j):
				img4[0][i,k]=im[i,k]
				img4[0][height-i-1,k]=im[height-i-1,k]
			elif(k>width-j):
				img4[2][i,k]=im[i,k]
				img4[2][height-i-1,k]=im[height-i-1,k]
			else:	
				img4[1][i,k]=im[i,k]
				img4[3][height-i-1,k]=im[height-i-1,k]
		if (j>width/2):
			break
	i=0
	feature = []
	for img in img4:
		feature = feature+feature_hu2(img)
	return feature




def load():
	classifier.load('svm_class.xml')
load()
def test():
	load()
	count,correct=0,0
	url='../samples/train_images/'
	for i in range(101,150):
		s_list=glob.glob(url+str(i)+'/*.png')
		for j in s_list:
			imgo=cv2.imread(j,0)
			img=pp.preprocess(imgo.copy())
			f = find_feature(img.copy())
			fu= np.array(f,np.float32)
			# print len(fu)
			t = classifier.predict(fu)
			print label_uni[i-100],label_uni[int(t)],int(i-100==t)
			if(i-100==t):
				correct+=1
			else:
				name = './zerr_'+str(i)+'_'+str(count)+'.png'
				print j
				print count
#				cv2.imwrite('./zerr_'+str(i)+'_'+str(count)+'z.png',img)
				shutil.copyfile(j,name)
			count+=1
	print 'accuracy :'+str(100.0*correct/count)+'%'
	print ('accurate recognition :'+str(correct))
	print ('total character tested :'+str(count))
