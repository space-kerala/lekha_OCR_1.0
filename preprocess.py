# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:22:39 2015

@author: james
"""
#preprocess and Segmentation
import cv2
import numpy as np
import training as train

# current_line = 0
# current_word = 0
previous_char = None
cur_char = None
BOX_SIZE = 128
def preprocess(img):#eliptical kernel
	cv2.imwrite('before_pp_thresholding.png',img)
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,243,50)
	cv2.imwrite('after_pp_thresholding.png',img)
	return img
def skew_correction(img):
	hight,width=img.shape
	if(hight>1200 and width>1200):
		box = img[hight/2-600:hight/2+600,width/2-600:width/2+600]
		hight,width=box.shape
	else:
		box = img
	edges = cv2.Canny(box,50,150,apertureSize = 3)
	lines = cv2.HoughLines(edges,1,np.pi/360,width/5,width/2,hight/10)
#	print lineskernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	count=0
	rotation=[]
	for rho,theta in lines[0]:
		angle = theta*180.0/np.pi
		if (angle<60 or angle >120):
			continue
		rotation.append(angle)
		count+=1
#		print angle
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
	# print rot
	hight,width=img.shape
	M = cv2.getRotationMatrix2D((hight/2,width/2),rot-90,1)
	dst = cv2.warpAffine(img,M,(width,hight))
	return dst,rot
def center_box(img,cnt):
	x,y,w,h=cv2.boundingRect(cnt)
	char=img[y-1:y+h+1,x-1:x+w+1]
	return char
	
def find_blocks(img):
	kernel = np.ones((6,4),np.uint16)
	kernel2 = np.ones((4,4),np.uint16)
	im = cv2.erode(img,kernel2,iterations = 8)
	im = cv2.dilate(img,kernel,iterations = 4)
	im = cv2.erode(img,kernel2,iterations = 4)
	im = cv2.dilate(img,kernel2,iterations = 6)
#	im = cv2.erode(img,kernel,iterations = 2)
	cv2.imwrite('block.png',im)
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
	max_cuts = 8
	min_line_width = 20
	j,start=0,0
	for i in range (0,hight):
		j = i
		if(hor_pix_den[i]<max_cuts):
			if(min_line_width>j-start):
				start = j
			else :
				# print start,j
				line = Line(img[start-1:j+1,0:width])
				line_list.append(line)
				start = j
#	print line_list
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
	min_word_sep = hight/5
	j,i,start=0,0,0
	while(i<width):
		if (ver_pix_den[i]<=max_cuts):
			j = i
			while(ver_pix_den[j]<=max_cuts and j<width-1):
				j+=1
			if(j-i>min_word_sep):
				if(i-start>min_word_sep):
					previous_char=None
					word = Word(img[0:hight,start:i+1])
					word_list.append(word)
				start=j
				i=j
			else:
				i=j
		i+=1
	if(i-start>min_word_sep):
		word = Word(img[0:hight,start-1:i+1])
		word_list.append(word)
	return word_list

class Word:
	no_letters = 0
	def __init__(self,img):
		self.data = img
		self.char_list = find_lettes(img)
		
def find_lettes(img):
	global previous_char
	char_list = []
	cv2.imwrite('t_word.png',img)
	contours2, hierarchy = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	if(len(contours2)==0):
		return char_list
	contours = []
	for cnt in contours2:
#		print (cv2.contourArea(cnt))
		if(cv2.contourArea(cnt)>6):
			try:
				contours.append(cnt)
			except ValueError:
				print ('error')
				pass
	Mset = [cv2.moments(cnt) for cnt in contours]
	X = [int(M['m10']/M['m00']) for M in Mset]
	index = [i for i in range(0,len(contours2))]
	X,index = zip(*sorted(zip(X,index)))
	for i in index:
		cnt = contours[i]
		box = center_box(img.copy(),cnt)
		letter = Letters(box)
		previous_char=letter
		char_list.append(letter)
	return char_list

class Letters:
	ratio = 0
	def __init__(self,char):
		global cur_char 
		cur_char = self
		self.hight,t=char.shape
		self.data=char
		self.feature=np.array(train.find_feature(self.data.copy()),np.float32)
		self.label=train.recognize(self.feature)
	# def hight(self):
	# 	h,w=self.data.shape
	# 	return h
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