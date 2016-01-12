# -*- coding: utf-8 -*-
from __future__ import division
import cv2
import numpy as np
import glob
import shutil
from random import shuffle
import os
import preprocess as pp
import training as train
import json
import re
#import myapp as app
def temp_fucn():
	url='../samples/train_images/'
	s_list=sorted(os.listdir(url))
	i=101
	for j in s_list:
		os.rename(url+j,url+str(i))
		i+=1
		print (j)
def temp_inv_fucn():
	url='../samples/train_temp/'
	print url
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
def train_svm():
	svm_params = dict( kernel_type = cv2.SVM_RBF,
	                    svm_type = cv2.SVM_C_SVC,
	                    C=9.34, gamma=15.68 )
	svm=cv2.SVM()
	label_list=[]
	label_list.append('a')
	url='train_images/'
	train_set = []
	s_list=sorted(os.listdir(url))
	label = 0
	for i in s_list:
		s_list=glob.glob(url+i+'/*.png')
		if(len(s_list)>25):
			file=open(url+i+'/utf8',"r")
			i_uni=file.read()
			i_uni=i_uni[:-1]
			label_list.append(i_uni)
			label+=1
		else:
			continue
		print str(label),i,label_list[label],len(s_list)
		for j in s_list:
			img=cv2.imread(j,0)
			img=pp.preprocess(img)
			f =train.find_feature(img.copy())
			# print len(f)
			s = [label,f]
			train_set.append(s)
	f=open('label','w')
	for l in label_list:
		f.write(l+'\n')
	f.close()

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
def gen_train_sample(im):
#	train.classifierclassifier.load('svm_class.xml')
	img = pp.preprocess(im.copy())
	# img,rot = pp.skew_correction(img)
	hight,width=im.shape
	# M = cv2.getRotationMatrix2D((hight/2,width/2),rot-90,1)
	# im = cv2.warpAffine(im,M,(width,hight))
	# cv2.imwrite('skew correct.png',im)
	contours2, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	contours = []
	for cnt in contours2:
		print (cv2.contourArea(cnt))
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
		char = pp.preprocess(box.copy())
		try:
			f = train.find_feature(char)
			fu= train.np.array(f,train.np.float32)
			# print len(fu)
			t = train.classifier.predict(fu)
			print t
		except IndexError:
			t = 0
		cv2.imwrite('samp/zsamp47_8_'+str(int(t))+'_'+str(i)+'.png',box)
		# cv2.imwrite('./samp/'+str(i)+'.png',box)
		i+=1
# train.test()
# train_svm()


# url='/home/jithin/lekha_OCR_1.0/samples/t8.png'
# img=cv2.imread(url,0)
# # im,rot = pp.skew_correction(img)
# # # cv2.imshow("img",img)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()
# gen_train_sample(img)



# url='../samples/Scans/test2.png'
# temp_fucn()
def make_compare_file():
	f=open('./Samp_22/compare_list.txt','w')
	g=open('./Samp_22/output_file.txt','w')
	# img=cv2.imread('./Example/dc_books_page.png',0)
	path='Samp_22/*.jpeg'
	url=glob.glob(path)
	img=cv2.imread(url[0],0)
	# img=cv2.imread('./Samp_3/samp3.png',0)
	if(img==None):
		print 'image does\'nt exist'
		exit()
	img = pp.preprocess(img)
	# im=img
	# im,rot = pp.skew_correction(img)

	line = pp.find_lines(img.copy())
	# print len(linene)
	label_list=train.label_unicode()
	i=0
	num=[]
	for l in line:
		for w in l.word_list:
			for c in w.char_list:
				# num.append((str(i),label_list[int(c.label)]))
				tup=label_list[int(c.label)]
				f.write(tup+'\n')
				g.write(tup)
				cv2.imwrite('./Samp_22/samp/'+str(i)+'.png',c.data)
				i+=1
			g.write(' ')
		g.write('\n')
	f.close()
	g.close()
		
# make_compare_file()	

	# for tup in num:
	# 	# f.write(tup[0]+' '+tup[1]+'\n')
	# 	f.write(tup[1]+'\n')
	# f.close()
def calculate_accuracy():
	f=open('./compare_list.txt','r')
	g=open('./preprocessed.txt','r')
	h=open('./result_compare.txt','w')
	l1=f.readlines()
	l2=g.readlines()
	# list1=[231,376,435,623,876,892,952,961,1002,1034,1036,1100,1155]
	j=0
	k=0
	for j in range(len(l1)):
		# for line2 in l2:
		a1=str(l1[j][:-1])
		a2=str(l2[j][:-1])
		# print a1+':'+a2
		if a1==a2:
			continue
		else:
			# print str(j)+':'+a1+'and'+a2
			# if j in list1:
			# 	print str(j)+':'+a1+'and'+a2
			h.write(str(j)+' '+a2+' '+a1+'\n')
			k+=1
	print 'ERRORS:',k
	print 'TOTAL:',j+1
	print 'ACCURACY:',100-((k/(j+1))*100)

# calculate_accuracy()

def find_vlines(img):
	edges = cv2.Canny(img,50,150,apertureSize = 3)
	h,w=img.shape
	# print h,w
	minLineLength=int(h*0.6)
	maxGap=int(w*0.3)
	maxLineGap =w*0.5
	lines = cv2.HoughLinesP(edges,1,np.pi,minLineLength,minLineLength,maxLineGap)
	try:
		x=sorted([x1 for x1,y1,x2,y2 in lines[0]])
		i=0
		c=[]
		while(i<len(x)):
			j=i+1
			while(j<len(x)):
				if (abs(x[i]-x[j])<maxGap or abs(x[i]-x[j])<15):
					c.append(x[j])
					del x[j]
					i+=1
					break
				else:
					j+=1
			i+=1
					
		# print c
		q=0
		for x1,y1,x2,y2 in lines[0]:
			if x1 in c:
				continue
			else:
				cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
				q+=1
		    
		return q
	except:
		# print 0
		return 0




def mytrain():
	svm_params = dict( kernel_type = cv2.SVM_RBF,
	                    svm_type = cv2.SVM_C_SVC,
	                    C=9.34, gamma=15.68 )
	svm=cv2.SVM()
	label_list=[]
	label_list.append('a')
	url='train_images/'
	train_set = []

	s_list=sorted(os.listdir(url))
	fr=open('newlabel.txt','w')
	
	label = 0
	for i in s_list:
		s_list=glob.glob(url+i+'/*.png')
		if(len(s_list)>25):
			file=open(url+i+'/utf8',"r")
			i_uni=file.read()
			i_uni=i_uni[:-1]
			label_list.append(i_uni)
			label+=1
		else:
			# for i in s_list1:
			fr.write(i+'\n')
			continue
		print str(label),i,label_list[label],len(s_list)
		for j in s_list:
			img=cv2.imread(j,0)

			# w=find_vlines(img.copy())

			img=pp.preprocess(img)
			f =train.find_feature(img.copy())
			# f+=[w]
			# print len(f)
			s = [label,f]
			train_set.append(s)
	f=open('label','w')
	for l in label_list:
		f.write(l+'\n')
	f.close()

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

# -------purify training set ----------

def get_labellist():
# load classifier

	label_list=[]
	label_list.append('a')
	url='train_images/'
	train_set = []
	s_list=sorted(os.listdir(url))
	label = 0
	for i in s_list:
		s_list=glob.glob(url+i+'/*.png')
		if(len(s_list)>25):
			file=open(url+i+'/utf8',"r")
			i_uni=file.read()
			i_uni=i_uni[:-1]
			label_list.append(i_uni)
			label+=1
		else:
			# for i in s_list1:
			continue
		# print str(label),i,label_list[label],len(s_list)
	return label_list

def purify_train():
	classifier = cv2.SVM()
	classifier.load('svm_class.xml')
	g=[]
	label_list=get_labellist()
	# label_list.append('a')
	url='train_images/'
	v=open('purify.txt','w')
	train_set = []
	s_list=sorted(os.listdir(url))
	label = 0
	for i in s_list:
		s_list=glob.glob(url+i+'/*.png')
		if(len(s_list)>25):
			# file=open(url+i+'/utf8',"r")
			# i_uni=file.read()
			# i_uni=i_uni[:-1]
			# label_list.append(i_uni)
			label+=1
		else:
			# for i in s_list1:
			continue
		print str(label),i,label_list[label],len(s_list)
		for j in s_list:
			img=cv2.imread(j,0)

			# w=find_vlines(img.copy())

			img=pp.preprocess(img)
			f =train.find_feature(img.copy())


			feature = np.array(f,np.float32)
			a = classifier.predict(feature)


			if a !=label:
				q=j.split('/')
				print a,label,int(a)
				# print label_list[int(a)],i

				v.write(q[2]+'\t'+label_list[int(a)]+' '+str(a)+'\t'+str(label)+' '+i+'\n')
				# cv2.imwrite('train_im/'+i+'/'+q[2],im)
				# os.rename(j,'train_im/'+i+'/'+q[2])


			feature=list(feature)
			
			g.append((feature,label))
	# print g

	with open('data.txt','w') as outfile:
		json.dump(g,outfile)


	# return f,g	
# -----------end of purification function---

def make_modified_file():
	f=open('./compare_list.txt','r')
	g=open('./output_file.txt','w')
	img=cv2.imread('./Example/dc_books_page.png',0)

	if(img==None):
		print url+' does\'nt exist'
		exit()
	img = pp.preprocess(img)
	im,rot = pp.skew_correction(img)

	line = pp.find_lines(im.copy())
	# print len(linene)
	label_list=train.label_unicode()

	q=f.readlines()
	i=0
	num=[]
	for l in line:
		for w in l.word_list:
			for c in w.char_list:
				# num.append((str(i),label_list[int(c.label)]))
				tup=label_list[int(c.label)]
				if(q[i][:-1]!=tup):
					print tup
				# f.write(tup+'\n')
				g.write(tup)
				# cv2.imwrite('samp/'+str(i)+'.png',c.data)
				i+=1
			g.write(' ')
		g.write('\n')
	f.close()
	g.close()



# def callapp():
# 	aap = app.simpleapp_tk(None)
# 	aap.title('My App')
# 	aap.mainloop()
# 	aap.OnButtonClick()

def recognize_block(im):
	line = pp.find_lines(im)
	# print len(linene)
	label_list=train.label_unicode()
	i=0
	string=''
	for l in line:
		# cv2.imwrite('zline_'+str(i)+'.png',l.data)
		# string=string+'\n'
		j=0
		for w in l.word_list:
	#		cv2.imwrite('zword_'+str(i)+'_word_'+str(j)+'.png',w.data)
			string=string+' '
			j+=1
			k=0
			c=0
			while(c<len(w.char_list)):
				char= w.char_list[c]
				try:
					if(label_list[int(char.label)]in ['\'',',']):
						char2=w.char_list[c+1]
						if(label_list[int(char2.label)]in ['\'',',']):
							string=string+'\"'
							c+=1
						else:
							string=string+label_list[int(char.label)]
					elif(label_list[int(char.label)]in ['െ','േ','്ര']):
						char2=w.char_list[c+1]
						if(label_list[int(char2.label)]in ['െ','്ര']):
							char3=w.char_list[c+2]
							string=string+label_list[int(char3.label)]
							c+=1
						string=string+label_list[int(char2.label)]
						string=string+label_list[int(char.label)]
						c+=1
					else:
						string=string+label_list[int(char.label)]
				except IndexError:
					string=string+label_list[int(char.label)]
				cv2.imwrite('output/zcline_'+str(i)+'_word_'+str(j)+'_c_'+str(k)+str(int(w.char_list[c].label))+'.png',w.char_list[c].data)
				k+=1
				c+=1
		i+=1
	return string

def process_lines(url):
	img=cv2.imread(url,0)
	# cv2.imshow("img",img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	filename=url[9:16]
	print filename
	img = pp.preprocess(img)
	# im,rot = pp.skew_correction(img)
	# cv2.imshow("img",im)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	string=recognize_block(img)
	with open('book/0001/'+filename+'.txt','w') as f:
		f.write(string+'\n')
	print string


def read_lines():
	path='book/0001/*.png'
	files=glob.glob(path)
	for i in files:
		# print i
		process_lines(i)
	# return files

	# print '=======phase1========'


# def visualize():
# 		for j in range(len(g)):
# 		if g[j]==83:
# 			print g[j]
# 			g[j]=0
# 		else:
# 			g[j]=1

# 	print g

# visualize()
# read_lines()

# img=cv2.imread('Example/pt2.png',0)
# cv2.imshow("img",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = pp.preprocess(img)
# im,rot = pp.skew_correction(img)
# # cv2.imshow("img",im)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# print recognize_block(img)


# print recognize_block(im.copy())
# mytrain()
# train_svm()
# make_compare_file()
# calculate_accuracy()

# purify_train()


# callapp()

# make_modified_file()
