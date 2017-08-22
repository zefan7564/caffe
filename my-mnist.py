#coding=utf-8
#caffe and opencv test mnist
#test by yuzefan
import os
import caffe
import numpy as np 
import cv2
import sys
caffe_root='/home/ubuntu/caffe-master/'
sys.path.insert(0,caffe_root+'python') #add this python path
os.chdir(caffe_root)
MODEL_FILE=caffe_root+'mytest/my-mnist/classificat_net.prototxt'
WEIGTHS=caffe_root+'mytest/my-mnist/lenet_iter_10000.caffemodel'
net=caffe.Classifier(MODEL_FILE,WEIGTHS)
caffe.set_mode_gpu()
IMAGE_PATH=caffe_root+'mytest/smy-mnist/'
font = cv2.FONT_HERSHEY_SIMPLEX #normal size sans-serif font
for i in range(0,9):
  # astype() is a method provided by numpy to convert numpy dtype.
  input_image=cv2.imread(IMAGE_PATH+'{}.png'.format(i),cv2.IMREAD_GRAYSCALE).astype(np.float32)
  #resize Image to improve vision effect.
  resized=cv2.resize(input_image,(280,280),None,0,0,cv2.INTER_AREA)
  input_image = input_image[:, :, np.newaxis] # input_image.shape is (28, 28, 1), with dtype float32
  # The previous two lines(exclude resized line) is the same as what caffe.io.load_iamge() do.
    # According to the source code, caffe load_image uses skiamge library to load image from disk.

    # for debug
    # print type(input_image), input_image.shape, input_image.dtype
    # print input_image

  prediction = net.predict([input_image], oversample=False)
  cv2.putText(resized, str(prediction[0].argmax()), (200, 280), font, 4, (255,), 2)
  cv2.imshow("Prediction", resized)
  print 'predicted class:', prediction[0].argmax()
  keycode = cv2.waitKey(0) & 0xFF
  if keycode == 27:
    break