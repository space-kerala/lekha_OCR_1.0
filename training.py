# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:26:15 2015

@author: james
"""
#feature extraction and Recognition
import cv2
import preprocess as pp
import numpy as np
import glob
import shutil
from random import shuffle
classifier = cv2.SVM()
classifier.load('svm_class.xml')

def find_feature(char):
	return zonewise_hu5(char)
	# return hog(char)

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
		# print pp.previous_char.hight,pp.cur_char.hight
		if(pp.cur_char.hight<=(pp.previous_char.hight*2/3)):
			# print 'O not o  ം'
			# return 93
			return label_uni.index('ം')
		return label_uni.index('ഠ')
	return a
label_uni = []
f = open('label','r')
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
# label_uni.append('ം')
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
	hight,width=im.shape
	box = img[0:1,0:1]
	box[0,0]=0
	box = cv2.resize(box,(width,hight))
	img4=[]
	[img4.append(box.copy())for i in range(0,4)]
	i=0
	for i in range (0,hight):
		j=(int)(i*width/hight)
		for k in range(0,width):
			if(k<j):
				img4[0][i,k]=im[i,k]
				img4[0][hight-i-1,k]=im[hight-i-1,k]
			elif(k>width-j):
				img4[2][i,k]=im[i,k]
				img4[2][hight-i-1,k]=im[hight-i-1,k]
			else:	
				img4[1][i,k]=im[i,k]
				img4[3][hight-i-1,k]=im[hight-i-1,k]
		if (j>width/2):
			break
	i=0
	feature = []
	for img in img4:
		feature = feature+feature_hu2(img)
	return feature

def train():
	svm_params = dict( kernel_type = cv2.SVM_RBF,
	                    svm_type = cv2.SVM_C_SVC,
	                    C=19.34, gamma=25.68 )
	svm=cv2.SVM()
	url='../samples/train_images/'
	train_set = []
	for i in range(101,208):
		s_list=glob.glob(url+str(i)+'/*.png')
		print i,label_uni[i-100],len(s_list)
		for j in s_list:
			img=cv2.imread(j,0)
			img=pp.preprocess(img)
			f =find_feature(img.copy())
			# print len(f)
			s = [i-100,f]
			train_set.append(s)
	shuffle(train_set)
	f_list = []
	label = []
	for t in train_set:
		label.append(t[0])
		f_list.append(t[1])
#	np.savetxt('feature.txt',f_list)
#	np.savetxt('label.txt',label)
#	samples = np.loadtxt('feature.txt',np.float32)
#	responses = np.loadtxt('label.txt',np.float32)
#	responses = responses.reshape((responses.size,1))  
	samples = np.array(f_list,np.float32)
	responses = np.array(label,np.float32)
	print 'auto training initiated'
	print 'please wait.....'
	svm.train(samples,responses,params=svm_params)
	# svm.train_auto(samples,responses,None,None,params=svm_params)
	svm.save("svm_class.xml")
def load():
	classifier.load('svm_class.xml')

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
