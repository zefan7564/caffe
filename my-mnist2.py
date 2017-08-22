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
MEAN_FILE=caffe_root+'mytest/my-mnist/mean.binaryproto'
print('Params loaded!')
cv2.waitKey(1000)
caffe.set_mode_gpu()
net=caffe.Net(MODEL_FILE,WEIGTHS,caffe.TEST)
mean_blob=caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(open(MEAN_FILE, 'rb').read())
mean_npy = caffe.io.blobproto_to_array(mean_blob)
a=mean_npy[0, :, 0, 0]
print(net.blobs['data'].data.shape)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2, 0, 1))
##transformer.set_raw_scale('data', 255)
#transformer.set_channel_swap('data', (2, 1, 0))
for i in range(0,10):
  IMAGE_PATH=caffe_root+'mytest/my-mnist/{}.png'.format(i)

  #img = caffe.io.load_image(IMAGE_PATH)
  input_image=cv2.imread(IMAGE_PATH,cv2.IMREAD_GRAYSCALE).astype(np.float32)
  resized=cv2.resize(input_image,(280,280),None,0,0,cv2.INTER_AREA)
  net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
  predict = net.forward()
  names = []
  with open('/home/ubuntu/caffe-master/mytest/my-mnist/words.txt', 'r+') as f:
    for l in f.readlines():
        names.append(l.split(' ')[1].strip())

  print(names)
  prob = net.blobs['prob'].data[0].flatten()
  print('prob: ', prob)
  print('class: ', names[np.argmax(prob)])
  cv2.imshow("Prediction", resized)
  keycode = cv2.waitKey(0) & 0xFF
  if keycode == 27:
    break
