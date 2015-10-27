# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:25:26 2015

Project Malayalam OCR - Lekha
-----------------------------

@author: james
"""
import cv2
import os
import preprocess as pp
import training as train

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
def gen_train_sample(im):
#	train.classifierclassifier.load('svm_class.xml')
	img = pp.preprocess(im.copy())
	img,rot = pp.skew_correction(img)
	hight,width=im.shape
	M = cv2.getRotationMatrix2D((hight/2,width/2),rot-90,1)
	im = cv2.warpAffine(im,M,(width,hight))
	cv2.imwrite('skew correct.png',im)
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
		char = pp.preprocess(box.copy())
		try:
			f = train.find_feature(char)
			fu= train.np.array(f,train.np.float32)
			# print len(fu)
			t = train.classifier.predict(fu)
			print t
		except IndexError:
			t = 0
		cv2.imwrite('samp/zsamp31_'+str(int(t))+'_'+str(i)+'.png',box)
		i+=1
def recognize_block(im):
	line = pp.find_lines(im)
	print len(line)
	label_list=train.label_unicode()
	i=0
	string='word:'
	for l in line:
		cv2.imwrite('zline_'+str(i)+'.png',l.data)
		string=string+'\n'
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
					if(label_list[int(char.label)]in ['\'',',1',',2']):
						char2=w.char_list[c+1]
						if(label_list[int(char2.label)]in ['\'',',1',',2']):
							string=string+'\"'
							c+=1
						elif(label_list[int(char.label)]in [',1',',2']):
							string=string+','
						else:
							string=string+label_list[int(char.label)]
					elif(label_list[int(char.label)]in ['ൾ2','ൾ']):
						string=string+'ൾ'
					elif(label_list[int(char.label)]in ['െ','േ','്ര']):
						char2=w.char_list[c+1]
						if(label_list[int(char2.label)]in ['െ','േ','്ര']):
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
# file=open('./result',"r")
# t=file.read()
# string =''
# c=0
# while(c<len(t)):
# 	char= t[c]#[-1]
# 	print char=='േ'
# 	if(char=='െ' or char=='േ'):
# 		try:
# 			char2=t[c+1]#[-1]
# 			string=string+char2
# 			string=string+char
# 			print 'e'#+char2+char
# 			c+=1
# 		except IndexError:
# 			print 'e'
# 			string=string+char
# 	else:
# 		string=string+char
# 	c+=1
# print string
# train.test()
# train.train()
# url='../samples/Scans/news_paper30.png'
url='../samples/Scans/test2.png'
# temp_fucn()


# url='../samples/Scans/news_paper31.png'
img=cv2.imread(url,0)
img = pp.preprocess(img)
# # word = pp.Word(img)
# # for c in word.char_list:
# # 	print train.label_uni[int(c.label)]
im,rot = pp.skew_correction(img)
# # cv2.imshow("img",im)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
print recognize_block(im)


# img = pp.preprocess(img)
# print train.recognize(train.np.array(train.find_feature(img),train.np.float32))
# #train.np.savetxt('t.txt',train.hog(img))
# train.load()
# train.test()
# gen_train_sample(img)
# #img = cv2.blur(img,(11,11))
# #ret, im = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# #cv2.imwrite('after_thresholding.png',im)

# #kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))
# #im = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel3)
# #im = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)
# #cv2.imwrite('opening.png',im)

# #cv2.imwrite('houghlines3.png',im)
# #pp.find_blocks(im)