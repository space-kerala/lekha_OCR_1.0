from __future__ import division
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
# train.test()
# train.train()
# url='../samples/Scans/news_paper30.png'
# url='../samples/Scans/test2.png'
# temp_fucn()
def make_compare_file():
	f=open('./compare_list.txt','w')
	img=cv2.imread('./Example/dc_books_page.png',0)
	if(img==None):
		print url+' does\'nt exist'
		exit()
	img = pp.preprocess(img)
	im,rot = pp.skew_correction(img)

	line = pp.find_lines(im.copy())
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
				# cv2.imwrite('samp/'+str(i)+'.png',c.data)
				i+=1
	f.close()
		
	

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
			print str(j)+':'+a1+'and'+a2
			h.write(str(j)+':'+a2+' as '+a1+'\n')
			k+=1
	print 'ERRORS:',k
	print 'TOTAL:',j+1
	print 'ACCURACY:',100-((k/(j+1))*100)

			
	




# make_compare_file()
calculate_accuracy()
	