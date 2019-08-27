from mtcnn.mtcnn import MTCNN
import cv2
import urllib.request
import matplotlib.pyplot as plt
import math 
import numpy as np
import mxnet as mx
import sklearn
from sklearn.preprocessing import Normalizer
import json
from mxnet.contrib.onnx.onnx2mx.import_model import import_model


def rotateImage(image, angle): 
    (w,h) = image.shape[:2]    
    center=(h/2,w/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (h, w))
    return new_image

def get_model(ctx, model):
    image_size = (112,112)
    # Import ONNX model
    sym, arg_params, aux_params = import_model(model)
    # Define and binds parameters to the network
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


def init():
    for i in range(4):
        mx.test_utils.download(dirname='mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}-0001.params'.format(i+1))
        mx.test_utils.download(dirname='mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}-symbol.json'.format(i+1))
        mx.test_utils.download(dirname='mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}.caffemodel'.format(i+1))
        mx.test_utils.download(dirname='mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}.prototxt'.format(i+1))
    # Determine and set context
    if len(mx.test_utils.list_gpus())==0:
        ctx = mx.cpu()
        print('use CPU')
    else:
        ctx = mx.gpu(0)
        print('use GPU')
    # Download onnx model
    mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100.onnx')
    # Path to ONNX model
    model_name = 'resnet100.onnx'
    # Load ONNX model
    model = get_model(ctx , model_name)
    return model
    

def get_feature(model,input_blob): 
    (count,channels,h,w)=input_blob.shape
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    modelOutputs = model.get_outputs()[0];
    embeddings =modelOutputs.asnumpy().tolist()
    i=0
    while i < len(embeddings):
        embeddings[i]=sklearn.preprocessing.normalize([embeddings[i]]).flatten().tolist()
        i =i+1
   
    return embeddings

def getFaceAligment (url):
    inputBlob=np.empty((0, 3, 112, 112))
    urllib.request.urlretrieve (url, "img.jpg")
    img = cv2.cvtColor(cv2.imread("img.jpg"), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    result = detector.detect_faces(img)
    i=0
    outEmb =[]
    for resus in result:
      if resus['confidence'] <0.7:
        continue
    
      # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
      bounding_box = resus['box']
      keypoints = resus['keypoints']
      expandCoeff=0
      (x,y,h,w)=(bounding_box[0]-expandCoeff,bounding_box[1]-expandCoeff,bounding_box[3]+expandCoeff*2,bounding_box[2]+expandCoeff*2)
  
      if h*w <30*30:
        continue
        
      if x<0:
        x=0
      if y<0:
        y=0
  
      cropCentrY = int(y+h/2)
      cropCentrX = int(x+w/2)
      cropSize=int(max(w,h)/2*1.5)
      cropY=cropCentrY-cropSize
      if cropY<0:
        cropY=0
    
      cropX=cropCentrX-cropSize
      if cropX<0:
        cropX=0     
      crop_img =img[(cropY):(cropCentrY+cropSize), (cropX):(cropCentrX+cropSize)]
  
      (w1,h1) = crop_img.shape[:2]
      if w1<=0 or h1 <=0:
        continue

      i=i+1
      #fig.add_subplot(14, 10, i)  
      (kp1X,kp1Y)=keypoints['right_eye']
      (kp2X,kp2Y)=keypoints['left_eye']
      dX=kp1X-kp2X
      dY=kp1Y-kp2Y
      angile=0
      if abs(dY)>abs(dX):
          if dY>0:
                angile=180-math.degrees(math.asin(dX/dY))-90
          else:
              angile=180+math.degrees(math.asin(dX/-dY))+90
      else:
              angile=math.degrees(math.asin(dY/dX))
      crop_img=rotateImage(crop_img,angile)  
      
      resizeS=int(112*1.3)

      crop_img=cv2.resize(crop_img, (resizeS, resizeS))

      cxy=int(resizeS/2)

      crop_img=crop_img[(cxy-56):(cxy+56),(cxy-56):(cxy+56)]
      crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)      
      crop_img = np.transpose(crop_img, (2,0,1))
      inputBlob = np.append(inputBlob,[crop_img],axis=0)
      
    return inputBlob


def getFaceFeatures(url):
    faces=getFaceAligment(url)
    model=init()
    featuresOut=get_feature(model, faces)
    resultus = {'url':url,
        'features':featuresOut}
    #jsonStr=json.dumps(resultus)
    #print(jsonStr)
    return resultus
