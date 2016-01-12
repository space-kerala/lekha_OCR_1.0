#print 9
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:36:22 2015
Project Malayalam OCR - Lekha
@author: james
"""
import cv2
import glob
import os
import shutil
from random import shuffle
import numpy as np
import itertools
from skimage import morphology

url='../samples/a4_14pt_300dpi.png'
BOX_SIZE = 128
stroke_width = 0
#classifier = cv2.KNearest()
classifier = cv2.SVM()
def find_lines(img):
	cv2.imwrite('t_img_in.png',img)
	line_list =[]
	hight,width=img.shape
	hor_pix_den=[0 for i in range(0,hight)]
	for i in range(0,hight):
		for j in range(0,width):
			hor_pix_den[i]+=img[i,j]
		hor_pix_den[i]/=255
#	print hor_pix_den
	max_cuts = 30
	min_line_width = 10
	j,start=0,0
	for i in range (0,hight):
		j = i
		if(hor_pix_den[i]<max_cuts):
			if(min_line_width>j-start):
				start = j
			else :
#				print start,j
				line = Line(img[start:j,0:width])
				line_list.append(line)
				start = j
	print (line_list)
	return line_list
class Line:
	no_words = 0
	def __init__(self,img):
		self.data = img
		self.sw = find_sw(img)
		self.word_list=find_words(img)

def find_words(img):
	hight,width=img.shape
	word_list = []
	ver_pix_den=[0 for i in range(0,width)]
	for i in range(0,width):
		for j in range(0,hight):
			ver_pix_den[i]+=img[j,i]
		ver_pix_den[i]/=255
	max_cuts = 1
	min_word_sep = hight/2
	j,i,start=0,0,0
	while(i<width):
		if (ver_pix_den[i]<=max_cuts):
			j = i
			while(ver_pix_den[j]<=max_cuts and j<width-1):
				j+=1
			if(j-i>min_word_sep):
				if(i-start>min_word_sep):
					word = Word(img[0:hight,start:i])
					word_list.append(word)
				start=j
				i=j
			else:
				i=j
		i+=1
	if(i-start>min_word_sep):
		word = Word(img[0:hight,start:i])
		word_list.append(word)
	return word_list

class Word:
	no_letters = 0
	def __init__(self,img):
		self.data = img
		self.char_list = find_lettes(img)
		
def find_lettes(img):
	char_list = []
	cv2.imwrite('t_word.png',img)
	contours2, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	contours = []
	for cnt in contours2:
		print (cv2.contourArea(cnt))
		if(cv2.contourArea(cnt)>5):
			try:
				contours.append(cnt)
			except ValueError:
				print ('error')
				pass
	Mset = [cv2.moments(cnt) for cnt in contours]
	X = [int(M['m10']/M['m00']) for M in Mset]
#	X,contours = zip(*sorted(zip(X,contours)))
	for cnt in contours:
		box = center_box(img.copy(),cnt)
		letter = Letters(box)
		char_list.append(letter)
	return char_list
class Letters:
	ratio = 0
	def __init__(self,char):
		self.data=char
#		self.feature=feature_hu(self.data.copy())
#		self.label=recognize(self.feature)
def preprocess(img):
	cv2.imwrite('before_pp_thresholding.png',img)
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,243,50)
	cv2.imwrite('after_pp_thresholding.png',img)
#	kernel = np.ones((2,2),np.uint8)
#	opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#	cv2.imwrite('after_pp_opening.png',opening)
	return img
def character_box(img,cnt):
	hight,width=img.shape
	box = img[0:1,0:1]
	box[0,0]=255
	box = cv2.resize(box,(BOX_SIZE,BOX_SIZE))
	x,y,w,h=cv2.boundingRect(cnt)
	for i in range(x,x+w):
		for j in range(y,y+h):
			if(cv2.pointPolygonTest(cnt,(i,j),False)==-1):
				img[j,i]=255
	char=img[y:y+h,x:x+w]
	if(h>w):
		newh=BOX_SIZE
		neww= int(w*newh/h)
		newy=(BOX_SIZE-newh)/2
		newx = (BOX_SIZE-neww)/2
		char = cv2.resize(char,(neww,newh),interpolation=cv2.INTER_CUBIC)
		box[newy:newy+newh,newx:newx+neww] = char[0:newh,0:neww]
	else :
		neww = BOX_SIZE
		newh = (h*neww/w)
		newx = (BOX_SIZE-neww)/2
		newy = (BOX_SIZE-newh)/2
		char = cv2.resize(char,(neww,newh),interpolation=cv2.INTER_CUBIC)
		box[newy:newy+newh,newx:newx+neww] = char[0:newh,0:neww]
	cv2.imwrite('t_word_resize.png',box)
	return box
def center_box(img,cnt):
#	return character_box(img,cnt)
	x,y,w,h=cv2.boundingRect(cnt)
#	if (h>BOX_SIZE or w>BOX_SIZE):
#		print ('resizing...')
#		return character_box(img,cnt)
#	hight,width=img.shape
#	box = img[0:1,0:1]
#	box[0,0]=255
#	box = cv2.resize(box,(BOX_SIZE,BOX_SIZE))
#	for i in range(x,x+w):
#		for j in range(y,y+h):
#			if(cv2.pointPolygonTest(cnt,(i,j),False)==-1):
#				img[j,i]=0
	char=img[y:y+h,x:x+w]
#	newy = (BOX_SIZE-h)/2
#	newx = (BOX_SIZE-w)/2
#	box[newy:newy+h,newx:newx+w] = char[0:h,0:w]
#	cv2.imwrite('t_word2.png',box)
	return char
	
def find_sw(img):
	global stroke_width
#	if(stroke_width !=0):
#		return stroke_width
	hight,width = img.shape
	array = [0 for i in range(0,width/2)]
	for j in range(0,hight):
		count = 0
		for i in range (0,width):
			if(img[j,i]==255):
				count+=1
			else:
				array[count]+=1
				count = 0
	array[0]=0
	stroke_width = array.index(max(array))
	return stroke_width
def find_feature(char):
	return zonewise_hu5(char)	
#t_no = 0
def feature_hu(char):
	global kli
	contours, hierarchy = cv2.findContours(char,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	moments = [0,0,0,0,0,0]
	if(len(contours)==0):
		return moments
	X = [cv2.contourArea(C) for C in contours]
	t=[i for i in range (0,len(contours))]
	X,t = zip(*sorted(zip(X,t),reverse=True))
	cnt = contours[0]
	#	epsilon = 0.001*cv2.arcLength(cnt,True)
	#	approx = cv2.approxPolyDP(cnt,epsilon,True)
	#	cv2.drawContours(char, [approx], 0, (155), 1)
	#	global t_no
	#	t_no+=1
	#	cv2.imwrite(str(t_no)+'appr.png',char)
	mom = cv2.HuMoments(cv2.moments(cnt))
	moments=mom[:-1]
	list = [m[0]for m in moments]
	return list
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
			mom = cv2.HuMoments(cv2.moments(cnt))
			moments=mom[:-1]
			[list.append(m[0]) for m in moments]
		except IndexError:
			[list.append(0.0) for j in range(0,6)]
	return list

ter=0
def zonewise_hu(im):
	global ter
#	cv2.imwrite('ztest_'+str(ter)+'.png',im)
	ter+=1
	contours, hierarchy = cv2.findContours(im.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	X = [cv2.contourArea(C) for C in contours]
#	print X
	t=[i for i in range (0,len(contours))]
	X,t = zip(*sorted(zip(X,t),reverse=True))
	cnt = contours[t[0]]
	x,y,w,h=cv2.boundingRect(cnt)
	img = im[y-1:y+h+1,x-1:x+w+1]
	cv2.imwrite('zone_cen.png',img)
	hight,width=img.shape
	img4=[]
	img4.append(img[0:hight/2,0:width/2])
	img4.append(img[0:hight/2,width/2:width])
	img4.append(img[hight/2:hight,0:width/2])
	img4.append(img[hight/2:hight,width/2:width])
	i=0
#	for img in img4:
#		cv2.imwrite('zq'+str(ter)+str(i)+'.png',img)
#		i+=1
	feature = []
	for img in img4:
#		print img.shape
		feature = feature+list(itertools.chain(feature_hu(img)))
#	print feature
	return feature

def zonewise_hu2(img):
	global ter
	contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	X = [cv2.contourArea(C) for C in contours]
	t=[i for i in range (0,len(contours))]
	X,t = zip(*sorted(zip(X,t),reverse=True))
	cnt = contours[t[0]]
	x,y,w,h=cv2.boundingRect(cnt)
	im = img[y-1:y+h+1,x-1:x+w+1]
	cv2.imwrite('zq2nd'+str(ter)+'.png',im)
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
	for img in img4:
		cv2.imwrite('zq2nd'+str(ter)+'_'+str(i)+'.png',img)
		i+=1
	ter+=1
	feature = []
	for img in img4:
		feature = feature+list(itertools.chain(feature_hu(img)))
	return feature
def zonewise_hu3(img):
	global ter
	contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	X = [cv2.contourArea(C) for C in contours]
	t=[i for i in range (0,len(contours))]
	X,t = zip(*sorted(zip(X,t),reverse=True))
	cnt = contours[t[0]]
	x,y,w,h=cv2.boundingRect(cnt)
#	im = img[y-1:y+h+1,x-1:x+w+1]
	hight,width=img.shape
	M = cv2.moments(cnt)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	img4=[]
	img4.append(img[0:cy,0:cx])
	img4.append(img[0:cy,cx:width])
	img4.append(img[cy:hight,0:cx])
	img4.append(img[cy:hight,cx:width])
	i=0
#	for img in img4:
#		cv2.imwrite('zq'+str(ter)+str(i)+'.png',img)
#		i+=1
#	ter+=1
	feature = []
	for img in img4:
		feature = feature+list(itertools.chain(feature_hu(img)))
	return feature

#junction=[[0 for i in range(124)] for j in range(124)]
#end=[[0 for i in range(124)] for j in range(124)]
def isjunction(img,i,j):
    count=-1
    for k in range(i-3,i+4):
        for l in range(j-3,j+4):
            if(junction[k][l]==1):
                return 0
    for k in range(i-1,i+2):
        for l in range(j-1,j+2):
            if(img[k,l]==255):
                count+=1
    if (count>2):
#        print 'jun',i,j,count
        junction[i][j]=1
        return 1
    else:
        return 0
def isendpoint(img,i,j):
    count=-1
    for k in range(i-3,i+3):
        for l in range(j-3,j+3):
            if(junction[k][l]==1):
                return 0
    for k in range(i-1,i+2):
        for l in range(j-1,j+2):
#            print k,l,img[k,l],i,j
            if(img[k,l]==255):
                count+=1;
    if (count==1):
#        print 'end',i,j,count
        end[i][j]=1
        return 1
    else:
        return 0
def zonewise_hu4(img):
	cv2.imwrite('skeltm.png',img)
	im = morphology.skeletonize(img>0)
	hight,width=img.shape
	imgn=im.view(np.uint8)
	for i in range (0,hight-1):
		for j in range (0,width-1):
			if(imgn[i,j]==1):
				imgn[i,j]=255
			else:
				imgn[i,j]=0
	img4=[]
	img4.append(imgn[0:hight/2,0:width/2])
	img4.append(imgn[0:hight/2,width/2:width])
	img4.append(imgn[hight/2:hight,0:width/2])
	img4.append(imgn[hight/2:hight,width/2:width])
	i=0
#	for img in img4:
#		cv2.imwrite('zhis'+str(ter)+str(i)+'.png',img)
#		i+=1
	feature = []
	for img in img4:
#		print img.shape
		count = 0
		for i in img:
			for j in i:
				if (j==255):
					count+=1
		h,w=img.shape
		value =[(float (count))/(h*w)]
#		print value
#		break
		feature = feature+value
#	print feature
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
	cv2.imwrite('zq2nd'+str(ter)+'.png',im)
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
#	for img in img4:
#		cv2.imwrite('zq2nd'+str(ter)+'_'+str(i)+'.png',img)
#		i+=1
#	ter+=1
#	feature = feature_hu2(img4[0])#+feature_hu2(img4[1])+feature_hu2(img4[2])+feature_hu2(img4[3])
	feature = []
	for img in img4:
		feature = feature+feature_hu2(img)
#	print len(feature_hu2(img4[0]))
#	print len(feature)
	return feature

def zonewise_hu6(img):
	hight,width=img.shape
	img4=[]
	img4.append(img[0:hight/2,0:width/2])
	img4.append(img[0:hight/2,width/2:width])
	img4.append(img[hight/2:hight,0:width/2])
	img4.append(img[hight/2:hight,width/2:width])
	feature = []
	for im in img4:
		k=0
		for i in im:
			for j in i:
				if (j==255):
					k+=1
		h,w=im.shape
#		print k,h,w
		feature.append(k/(float)(h*w))
#	print feature
	return feature
#i=1
#g=cv2.imread('t'+str(i)+'.png',0)
#print zonewise_hu6(preprocess(g))
def turning_points(img):
#	print find_sw(img)
	contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	X = [cv2.contourArea(C) for C in contours]
	t=[i for i in range (0,len(contours))]
	X,t = zip(*sorted(zip(X,t),reverse=True))
	cnt = contours[t[0]]
	x,y,w,h=cv2.boundingRect(cnt)
	for i in range(x,x+w):
		for j in range(y,y+h):
			if(cv2.pointPolygonTest(cnt,(i,j),False)==-1):
				img[j,i]=0
	contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	feature=[]
	feature.append(len(contours))
#	cv2.imwrite('skeltm.png',img)
#	im = morphology.skeletonize(img>0)
#	hight,width=img.shape
#	imgn=im.view(np.uint8)
#	for i in range (0,hight-1):
#		for j in range (0,width-1):
#			if(imgn[i,j]==1):
#				imgn[i,j]=255
#			else:
#				imgn[i,j]=0
#	c1=c2=c3=c4=0
#	for i in range(0,hight):
#		for j in range(0,width):
#			if(imgn[i,j]==255):
#				if(isjunction(imgn,i,j)):# or isendpoint(imgn,i,j)):
##					print 'junction',i,j
#					if(i<hight/2):
#						c1+=1
#					else:
#						c2+=1
#					if(j<width/2):
#						c3+=1
#					else:
#						c4+=1
#				elif(isendpoint(imgn,i,j)):
##					print 'end',i,j
#					if(i<hight/2):
#						c1+=1
#					else:
#						c2+=1
#					if(j<width/2):
#						c3+=1
#					else:
#						c4+=1
#	feature.append(c1+c2+c3+c4)
#	feature.append(c1)
#	feature.append(c2)
#	feature.append(c3)
#	feature.append(c4)
#	kernel = np.ones((2,2),np.uint8)
#	imgn = cv2.dilate(imgn,kernel,iterations =2)
#	cv2.imwrite('skelt.png',imgn)
	return feature
classifier.load('svm_class.xml')
def recognize(feature):
#	print feature
	a = classifier.predict(feature)
#	f = np.reshape(feature,(1,6))
#	retval, results, neigh_resp, dists = classifier.find_nearest(f, k = 1)
#	a = results[0][0]
	return a

label_uni = []
label_uni.append('0')
def label_unicode():
	url='../samples/train_images/'
	for i in range(1,56):
		file=open(url+str(i)+'/utf8',"r")
		i_uni=file.read()
		i_uni=i_uni[:-1]
		label_uni.append(i_uni)
	return label_uni
#class SVM():
#    def __init__(self, C = 1, gamma = 0.5):
#        self.model = cv2.ml.SVM_create()
#        self.model.setGamma(gamma)
#        self.model.setC(C)
#        self.model.setKernel(cv2.ml.SVM_RBF)
#        self.model.setType(cv2.ml.SVM_C_SVC)
#
#    def train(self, samples, responses):
#        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
#
#    def predict(self, samples):
#        return self.model.predict(samples)[1][0].ravel()
#    def load(self, fn):
#        self.model.load(fn)
#    def save(self, fn):
#        self.model.save(fn)
def train():
	svm_params = dict( kernel_type = 2,
	                    svm_type = 100,
	                    C=2.67, gamma=5.383 )
	svm=cv2.SVM()
	url='../samples/train_images/'
	train_set = []
	for i in range(1,5):
		s_list=glob.glob(url+str(i)+'/*.png')
		print i,label_uni[i],len(s_list)
		for j in s_list:
			img=cv2.imread(j,0)
#			cv2.imwrite('zone_cen_bfprepo.png',img)
			img=preprocess(img)
#			cv2.imwrite('zone_cen_afprepo.png',img)
#			tu = turning_points(img.copy())
#			print j
			f =find_feature(img.copy())#+feature_hu(img.copy())+zonewise_hu2(img.copy())+zonewise_hu(img.copy())
#			print len(f)
#			f= np.array(f,np.float32)
#			print fu
#			f.append(float(k) for k in tu)
#			print f
			s = [i,f]
			train_set.append(s)
#	print len(train_set)
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
#	print f_list
	samples = np.array(f_list,np.float32)
	responses = np.array(label,np.float32)
	svm.train_auto(samples,responses,[],[],params=svm_params)
	svm.save("svm_class.xml")
#	classifier.train(samples,responses)
#	print classifier.predict(samples[5])
#	classifier.save("svm_class.xml")
def test():
#	svm = cv2.ml.SVM_create()
#	classifier = cv2.SVM()
#	global classifier
#	classifier.load('svm_class.xml')
	count,correct=0,0
	url='../samples/test_image/'
	for i in range(1,56):
		s_list=glob.glob(url+str(i)+'/*.png')
		for j in s_list:
			imgo=cv2.imread(j,0)
			img=preprocess(imgo.copy())
#			print j
#			contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#			X = [cv2.contourArea(C) for C in contours]
#			t=[i for i in range (0,len(contours))]
#			X,t = zip(*sorted(zip(X,t),reverse=True))
#			cnt = contours[t[0]]
#			x,y,w,h=cv2.boundingRect(cnt)
#			im = imgo[y-1:y+h+1,x-1:x+w+1]
#			cv2.imwrite(j,im)
			f = find_feature(img.copy())#turning_points(img.copy())+zonewise_hu5(img.copy())#+feature_hu(img.copy())+zonewise_hu2(img.copy())+zonewise_hu(img.copy())
#			print len(f)
			fu= np.array(f,np.float32)
			t = classifier.predict(fu)
			print label_uni[i],label_uni[int(t)],int(i==t)
			if(i==t):
				correct+=1
			else:
				name = './zerr_'+str(i)+'_'+str(count)+'.png'
				print j
				print count
				shutil.copyfile(j,name)
			count+=1
	print ('accurate recognition :'+str(correct))
	print ('total character tested :'+str(count))
def gen_train_sample(im):
	classifier.load('svm_class.xml')
	img = preprocess(im.copy())
	contours2, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	contours = []
	for cnt in contours2:
#		print (cv2.contourArea(cnt))
		if(cv2.contourArea(cnt)>20):
			contours.append(cnt)
	X = [cv2.contourArea(C) for C in contours]
#	print len(contours),len(X)
	t=[i for i in range (0,len(contours))]
	X,t = zip(*sorted(zip(X,t)))
	i=0
	for j in t:
		x,y,w,h=cv2.boundingRect(contours[j])
		box = im[y-1:y+h+1,x-1:x+w+1]
		char = preprocess(box.copy())
		try:
			f = zonewise_hu5(char)
			fu= np.array(f,np.float32)
			t = classifier.predict(fu)
		except IndexError:
			t = 0
		cv2.imwrite('zsamp21_'+str(t)+str(i)+'.png',box)
		i+=1
def temp_feature(img):
	hight,width=img.shape
	ver_pix_den=[0 for i in range(0,width)]
	for i in range(0,width):
		for j in range(0,hight):
			ver_pix_den[i]+=img[j,i]
		ver_pix_den[i]/=255
	hor_pix_den=[0 for i in range(0,width)]
	for i in range(0,hight):
		for j in range(0,width):
			hor_pix_den[i]+=img[i,j]
		hor_pix_den[i]/=255
def temp_fucn():
	url='../samples/train_images/'
	s_list=os.listdir(url)
	i=1
	for j in s_list:
		os.rename(url+j,url+str(i))
		i+=1
		print (j)
def temp_inv_fucn():
	url='../samples/test_image/'
#	url2='../samples/train_images/'
	s_list=os.listdir(url)
	for j in s_list:
		file=open(url+j+'/utf8',"r")
#		file2=open(url2+j+'/utf8',"r")
		i_uni=file.read()
#		i_uni2=file2.read()
		i_uni=i_uni[:-1]
#		i_uni2=i_uni2[:-1]
#		print i_uni,i_uni2
		os.rename(url+j,url+i_uni)
#temp_inv_fucn()
url='../samples/Scans/scew2.jpg'
img=cv2.imread(url,0)
hight,width=img.shape
if(hight>1200 and width>1200):
	box = img[hight/2-600:hight/2+600,width/2-600:width/2+600]
	hight,width=box.shape
else:
	box = img
im=preprocess(box)
cv2.imwrite('afterpp.png',im)
edges = cv2.Canny(im,50,150,apertureSize = 3)
cv2.imwrite('canny.png',im)
lines = cv2.HoughLines(edges,1,np.pi/360,width/5,width/2,hight/10)
print lines
count=0
rotation=[]
for rho,theta in lines[0]:
	angle = theta*180.0/np.pi
	if (angle<60 or angle >120):
		continue
	rotation.append(angle)
	count+=1
	print angle
	if(count == 50):
		break
#	a = np.cos(theta)
#	b = np.sin(theta)
#	x0 = a*rho
#	y0 = b*rho
#	x1 = int(x0 + 3000*(-b))
#	y1 = int(y0 + 3000*(a))
#	x2 = int(x0 - 3000*(-b))
#	y2 = int(y0 - 3000*(a))
#	cv2.line(img,(x1,y1),(x2,y2),200,2)
rotation.sort()
rot=rotation[len(rotation)/2]
print rot
hight,width=img.shape
M = cv2.getRotationMatrix2D((hight/2,width/2),rot-90,1)
dst = cv2.warpAffine(img,M,(width,hight),flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS, fillval=(255))
cv2.imwrite('houghlines3.png',dst)
###cv2.imwrite('news_paper_300ppi.png',img)
#gen_train_sample(img)
#label_unicode()
#train()
#test()
##test 1 BOX
#i = 1


#url=url+str(i)+'.png'
#img=cv2.imread(url)
#im=img
#img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
#imgray = img.copy()
#contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#Mset = [cv2.moments(cnt) for cnt in contours]
#X = [int(M['m10']/M['m00']) for M in Mset]
#X,contours = zip(*sorted(zip(X,contours)))
#i = 0
#while(i!=-1):
#	cnt=contours[i]
#	cv2.drawContours(im, [cnt], 0, (0,255,0), 1)
#	char = character_box(imgray,cnt)
#	cv2.imwrite('test'+str(i)+'.png',char)
#	t=feature_hu(char)	
#	i+=1
##	if(i==28):../samples/news_paper.png'
#img = cv2.imread(url,0)
#img = preprocess(img)
##kernel = np.ones((1,1),np.uint8)
##im = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#char_list=find_lettes(img)
#i=0
#for let in char_list:
##	print i
#	im = (255-let.data)
#	cv2.imwrite('tt'+str(i)+'.png',im)
#	i+=1

#		break
#cv2.imwrite('t.png',im)



##test 2 line segmentation
#url='../samples/a4_14pt_300dpi.png'cv2.drawContours(im, [cnt], 0, (0,255,0), 1)
#img=cv2.imread(url,0)
#im = preprocess(img)
#cv2.imwrite('inv.png',im)
#line_list = find_lines(im)
#i = 1
#for l in line_list:
#	cv2.imwrite(str(i)+'line.png',l.data)
#	i+=1
#	


###test 3 word segmentation
#url='../samples/line1.png'
#img=cv2.imread(url,0)
#im = preprocess(img)
#line = Line(im)
#print line.word_list
#cv2.imwrite('word.png',line.word_list[0].data)


##test 4 resize interpolation
#url='../samples/line1.png'
#img=cv2.imread(url,0)
#im = preprocess(img)
#char = cv2.resize(im,(400,50),interpolation=cv2.INTER_CUBIC)
#cv2.imwrite('resize.png',char)

##test 5 letter segmentation
#url='../samples/line1.png'
#img=cv2.imread(url,0)
#im = preprocess(img)
#line = Line(im)
#cv2.imwrite('char.png',line.word_list[2].char_list[1].data)
#url='../samples/line1.png'
#img = cv2.imread(url,0)
#im = preprocess(img)
#sw = find_sw(im)
#print sw
#kernel = np.ones((2,2),np.uint16)
#im = cv2.erode(im,kernel,iterations = 2)
#im = cv2.dilate(im,kernel,iterations = 2)
#cv2.imwrite('line.png',im)
#print find_sw(im)


###generating training data
#url='../samples/train_images/'
#train_set = []
#for i in range(1,15):
#	for j in range(1,6):
#		img=cv2.imread(url+str(i)+'/'+str(j)+'.png',0)
#		img=preprocess(img)
#		f_t = feature_hu(img)
#		f = [f_t[0],f_t[1], f_t[2],f_t[3],f_t[5]]		
#		s = [i,f]
#		train_set.append(s)
#shuffle(train_set)
#f_list = []
#label = []
#for t in train_set:
#	label.append(t[0])
#	f_list.append(t[1])
#np.savetxt('feature.txt',f_list)
#np.savetxt('label.txt',label)
#train()
#
##testing
#
#samples = np.loadtxt('feature.txt',np.float32)
#responses = np.loadtxt('label.txt',np.float32)
#i=0
#count = 0
#correct = 0
#for s in samples:
#	result = recognize(s)
#	if (responses[i]==result):
#		correct+=1
#	print responses[i],result
#	count+=1
#	i+=1
#print correct,count


#sample generation
#url='../samples/news_paper3.png'
#img = cv2.imread(url,0)
#img = preprocess(img)
#kernel = np.ones((2,2),np.uint8)
##img = cv2.erode(img,kernel,iterations = 2)
##img = cv2.dilate(im,kernel,iterations = 1)
##print img
#line_list = find_lines(img)
#i = 1
#for l in line_list:
#	l_list = l.word_list
#	for w in l_list:
#		w_list = w.char_list
#		for let in w_list:
#			print i
#			im = (255-let.data)
##			im = preprocess(im.copy())
#			cv2.imwrite('a'+str(i)+'.png',im)
#			i+=1

#url='../samples/news_paper.png'
#img = cv2.imread(url,0)
#img = preprocess(img)
##kernel = np.ones((1,1),np.uint8)
##im = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#char_list=find_lettes(img)
#i=0
#for let in char_list:
##	print i
#	im = (255-let.data)
#	cv2.imwrite('tt'+str(i)+'.png',im)
#	i+=1

##testing validity of humoments
#im = cv2.imread('moment_test3.png',0)
#img = preprocess(im.copy())
#contours, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
##X = [cv2.contourArea(C) for C in contours]
#Mse = [(cv2.moments(cnt)) for cnt in contours]
#X = [int(M['m10']/M['m00']) for M in Mse]
#X,contours = zip(*sorted(zip(X,contours)))
#Box = [center_box(im,cnt) for cnt in contours]
##[cv2.imwrite(str(i)+'tbox.png',Box[i]) for i in range(0,11)]
#
##print zonewise_hu(Box[0])
#Mset = [zonewise_hu(preprocess(box.copy())) for box in Box]
#f=open("t5.txt","w")
##f.write('This is a test\n')
#i = 0
#for it in Mset: 
#	f.write(str(X[i]))
#	i+=1
#	for jt in it:
#		f.write(' '+str(jt[0]))
#	f.write('\n')
#f.close
