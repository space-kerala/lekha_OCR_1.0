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
