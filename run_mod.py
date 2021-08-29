#!/usr/bin/env python
import argparse
import logging
import sys
import os
import time
import json
import glob
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials,db,firestore, storage
from getmac import get_mac_address as gma
from boto3.session import Session
import boto3
from datetime import datetime


if not firebase_admin._apps:
  cred=credentials.Certificate('/app/firebasecredential.json')
  firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://demoplayer-ecc96.firebaseio.com',
    'storageBucket': 'demoplayer-ecc96.appspot.com'
  })

ref = db.reference('progress')


logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

progress=0.0
oldTime=time.time()
newTime=time.time()
oldProgress=progress
newProgress=progress


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--postID', type=str, default='none')
    parser.add_argument('--userID', type=str, default='none')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=432x368, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()
    
    ref.child(args.postID).set({'object':{'progress':0}})
    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    

    try: 
      os.mkdir('/openPose/images_'+args.postID) 
    except OSError as error: 
      print(error)   


    try: 
      os.mkdir('/openPose/output_'+args.postID) 
    except OSError as error: 
      print(error) 
    
   #video 1 split into frames

    src = cv2.VideoCapture('/app/'+args.postID+'_input1.mp4')
    fps = src.get(cv2.CAP_PROP_FPS)

    frame_num = 0
    while(frame_num< int(src.get(cv2.CAP_PROP_FRAME_COUNT))):
      # Capture frame-by-frame
      ret, frame = src.read()
      if ret == False:
        break
      # Saves image of the current frame in jpg file
      name = '/openPose/images_'+args.postID+'/f1rame_' + str(frame_num) + '.png'
      #print ('Creating...' + name)
      cv2.imwrite(name, frame)

      # To stop duplicate images
      frame_num += 1

      # When everything done, release the capture
    src.release()
    cv2.destroyAllWindows()



   #video 2 split into frames

    src = cv2.VideoCapture('/app/'+args.postID+'_input2.mp4')
    fps = src.get(cv2.CAP_PROP_FPS)

    frame_num = 0
    while(frame_num< int(src.get(cv2.CAP_PROP_FRAME_COUNT))):
      # Capture frame-by-frame
      ret, frame = src.read()
      if ret == False:
        break
      # Saves image of the current frame in jpg file
      name = '/openPose/images_'+args.postID+'/f2rame_' + str(frame_num) + '.png'
      #print ('Creating...' + name)
      cv2.imwrite(name, frame)

      # To stop duplicate images
      frame_num += 1

      # When everything done, release the capture
    src.release()
    cv2.destroyAllWindows()


    f1Count = len(glob.glob1('/openPose/images_'+args.postID,"f1rame_*.png"))
    f2Count = len(glob.glob1('/openPose/images_'+args.postID,"f2rame_*.png"))

    fCount = f1Count if f1Count < f2Count else f2Count 
    # processing first video
    data1 = {}
    data1['parts'] = []

    data = {}
    data['frames'] = []

    for i in range(0,fCount): 
      newTime=time.time()
      progress=80.0*float(i)/(2*fCount)
      newProgress=progress  
      if newProgress-oldProgress>5.0:
        oldProgress=newProgress
        try:
          ref.child(args.postID).set({'object':{'progress':progress}})
          print('progress is:',str(progress))
        #  logger.info('progress is %s' % str(progress))
          
        except:
          print("File write exception from run_mod: ",sys.exc_info()[0]) 
            

      # estimate human poses from a single image !
      image = common.read_imgfile('/openPose/images_'+args.postID+'/f1rame_'+str(i)+'.png', None, None)
      if image is None:
          logger.error('Image can not be read')
          sys.exit(-1)
      t = time.time()
      humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
      elapsed = time.time() - t
      data1 = {}
      data1['parts'] = []
      measure1=np.zeros((len(humans),))
      kk=0
      for human in humans:
            # draw point
            for ii in range(common.CocoPart.Background.value):
                if ii not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[ii]
                
                if 10 in human.body_parts.keys() and 1 in human.body_parts.keys():
               # center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                  x1= human.body_parts[10].x
                  y1=human.body_parts[10].y
                  x2= human.body_parts[1].x
                  y2=human.body_parts[1].y
                  measure1[kk] = (x1-x2)**2+(y1-y2)**2
            kk=kk+1
            
      logger.info('inference image f1rame_: %s in %.4f seconds.' % (str(i), elapsed))
      #print('inference f1_rame_'str(i),' is 'elapsed, 'seconds')
      try:
       # image = TfPoseEstimator.draw_humans(image, humans[np.argmax(measure1):np.argmax(measure1)+1], imgcopy=False)
        human1=humans[np.argmax(measure1)]
        for ii in range(common.CocoPart.Background.value):
                if ii not in human1.body_parts.keys():
                    continue

                body_part = human1.body_parts[ii]
                data1['parts'].append({
                'id': ii,
                'x': body_part.x,
                'y': body_part.y
                })        
      except:  
          continue
      #  image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
     # cv2.imwrite('/openPose/output_'+args.postID+'/f1rame_'+str(i)+'.png',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
     
      data['frames'].append({
        'num': i,
        'array':data1}
      )

    with open('/openPose/output_'+args.postID+'/data1.json', 'w') as outfile:
      json.dump(data, outfile)


    # processing second video
    data1 = {}
    data1['parts'] = []

    data = {}
    data['frames'] = []

    for i in range(0,fCount): 
      newTime=time.time()
      progress=40.0+80.0*float(i)/(2*fCount)
      newProgress=progress  
      if newProgress-oldProgress>5.0:
        oldProgress=newProgress
        try:
          ref.child(args.postID).set({'object':{'progress':progress}})
        #  print('progress is:',str(progress))
          logger.info('progress is %s'% str(progress))
        except:
          print("File write exception from run_mod :",sys.exc_info()[0]) 
      
      # estimate human poses from a single image !
      image = common.read_imgfile('/openPose/images_'+args.postID+'/f2rame_'+str(i)+'.png', None, None)
      if image is None:
          logger.error('Image can not be read')
          sys.exit(-1)
      t = time.time()
      humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
      elapsed = time.time() - t
      data1 = {}
      data1['parts'] = []
      measure2=np.zeros((len(humans),))
      kk=0
      for human in humans:
            # draw point
            for ii in range(common.CocoPart.Background.value):
                if ii not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[ii]
               # center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                if 10 in human.body_parts.keys() and 1 in human.body_parts.keys():
                  x1= human.body_parts[10].x
                  y1=human.body_parts[10].y
                  x2= human.body_parts[1].x
                  y2=human.body_parts[1].y
                  measure2[kk] = (x1-x2)**2+(y1-y2)**2               
            kk=kk+1        

      #logger.info('inference image f2rame_: %s in %.4f seconds.' % (str(i), elapsed))
      try: 
       # image = TfPoseEstimator.draw_humans(image, humans[np.argmax(measure2):np.argmax(measure2)+1], imgcopy=False)
        human2=humans[np.argmax(measure2)]
        for ii in range(common.CocoPart.Background.value):
                if ii not in human2.body_parts.keys():
                    continue

                body_part = human2.body_parts[ii]
                data1['parts'].append({
                'id': ii,
                'x': body_part.x,
                'y': body_part.y
                }) 


      except:
          continue
     #   image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)  
     # cv2.imwrite('/openPose/output_'+args.postID+'/f2rame_'+str(i)+'.png',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
     
      data['frames'].append({
        'num': i,
        'array':data1}
      )

    with open('/openPose/output_'+args.postID+'/data2.json', 'w') as outfile:
      json.dump(data, outfile)  

    #os.system('ffmpeg -i /openPose/output/f1rame_%d.png -y -start_number 1 -vf scale=400:800 -c:v libx264 -pix_fmt yuv420p /openPose/output/out1.mp4')
    #os.system('ffmpeg -i /openPose/output/f2rame_%d.png -y -start_number 1 -vf scale=400:800 -c:v libx264 -pix_fmt yuv420p /openPose/output/out2.mp4')

    
      
   
    
   
    def getScore(x):
      if x<10:
        return 100
      if x<=40 and x>=10:
        return 100-2*x
      if x>40:
        return 0


#     xline1_2=np.zeros((2,))
#     xline1_5=np.zeros((2,))
#     xline2_3=np.zeros((2,))
#     xline3_4=np.zeros((2,))
#     xline5_6=np.zeros((2,))
#     xline6_7=np.zeros((2,))
#     xline1_11=np.zeros((2,))
#     xline1_8=np.zeros((2,))
#     xline11_12=np.zeros((2,))
#     xline12_13=np.zeros((2,))
#     xline8_9=np.zeros((2,))
#     xline9_10=np.zeros((2,))
#     xline0_1=np.zeros((2,))

#     yline1_2=np.zeros((2,))
#     yline1_5=np.zeros((2,))
#     yline2_3=np.zeros((2,))
#     yline3_4=np.zeros((2,))
#     yline5_6=np.zeros((2,))
#     yline6_7=np.zeros((2,))
#     yline1_11=np.zeros((2,))
#     yline1_8=np.zeros((2,))
#     yline11_12=np.zeros((2,))
#     yline12_13=np.zeros((2,))
#     yline8_9=np.zeros((2,))
#     yline9_10=np.zeros((2,))
#     yline0_1=np.zeros((2,))

    v1line1_2=np.zeros((2,))
    v1line1_5=np.zeros((2,))
    v1line2_3=np.zeros((2,))
    v1line3_4=np.zeros((2,))
    v1line5_6=np.zeros((2,))
    v1line6_7=np.zeros((2,))
    v1line1_11=np.zeros((2,))
    v1line1_8=np.zeros((2,))
    v1line11_12=np.zeros((2,))
    v1line12_13=np.zeros((2,))
    v1line8_9=np.zeros((2,))
    v1line9_10=np.zeros((2,))
    v1line0_1=np.zeros((2,))

    v2line1_2=np.zeros((2,))
    v2line1_5=np.zeros((2,))
    v2line2_3=np.zeros((2,))
    v2line3_4=np.zeros((2,))
    v2line5_6=np.zeros((2,))
    v2line6_7=np.zeros((2,))
    v2line1_11=np.zeros((2,))
    v2line1_8=np.zeros((2,))
    v2line11_12=np.zeros((2,))
    v2line12_13=np.zeros((2,))
    v2line8_9=np.zeros((2,))
    v2line9_10=np.zeros((2,))
    v2line0_1=np.zeros((2,))
    theta=np.zeros((13,))

    with open('/openPose/output_'+args.postID+'/data1.json') as f1:
      frame_array1= json.load(f1)

    f1Count=  len(frame_array1['frames'])

    with open('/openPose/output_'+args.postID+'/data2.json') as f2:
      frame_array2= json.load(f2)

    f2Count=  len(frame_array2['frames']) 
    pose2=np.zeros((18*2*f2Count,))
    #print('frame:1:',f1Count,'frame 2:',f2Count)
    minFrames=f1Count if f1Count < f2Count else f2Count
    if minFrames <1:
      minFrames=1
    x1Array=np.zeros((18,minFrames))
    y1Array=np.zeros((18,minFrames))
    x2Array=np.zeros((18,minFrames))
    y2Array=np.zeros((18,minFrames))

    k=0 
    netSim=0
    simArr=np.zeros((minFrames,))
    simArr[:]=np.NAN
    for frame in frame_array1['frames']:
      frameList=frame['num']
      partList=frame['array']
      parts=partList['parts']
      k1=0
      if k<minFrames:
        for part in parts:  
            x=part['x']
            y=part['y']
            x1Array[k1][k]=x
            y1Array[k1][k]=y
            k1=k1+1
      k=k+1

    k=0 
    for frame in frame_array2['frames']:
      frameList=frame['num']
      partList=frame['array']
      parts=partList['parts']
      k1=0
      if k<minFrames:
        for part in parts:  
            x=part['x']
            y=part['y']
            x2Array[k1][k]=x
            y2Array[k1][k]=y
            k1=k1+1
      k=k+1


    #print(x1Array)
    
    for i in range(0,minFrames):
#       xline1_2[0]=x1Array[1][i]
#       xline1_2[1]=x1Array[2][i]
#       yline1_2[0]=-1*y1Array[1][i]
#       yline1_2[1]=-1*y1Array[2][i]
      v1line1_2[0]=x1Array[2][i]-x1Array[1][i]
      v1line1_2[1]=y1Array[2][i]-y1Array[1][i]


#       xline1_5[0]=x1Array[1][i]
#       xline1_5[1]=x1Array[5][i]
#       yline1_5[0]=-1*y1Array[1][i]
#       yline1_5[1]=-1*y1Array[5][i]
      v1line1_5[0]=x1Array[5][i]-x1Array[1][i]
      v1line1_5[1]=y1Array[5][i]-y1Array[1][i]



#       xline2_3[0]=x1Array[2][i]
#       xline2_3[1]=x1Array[3][i]
#       yline2_3[0]=-1*y1Array[2][i]
#       yline2_3[1]=-1*y1Array[3][i]
      v1line2_3[0]=x1Array[3][i]-x1Array[2][i]
      v1line2_3[1]=y1Array[3][i]-y1Array[2][i]


#       xline3_4[0]=x1Array[3][i]
#       xline3_4[1]=x1Array[4][i]
#       yline3_4[0]=-1*y1Array[3][i]
#       yline3_4[1]=-1*y1Array[4][i]
      v1line3_4[0]=x1Array[4][i]-x1Array[3][i]
      v1line3_4[1]=y1Array[4][i]-y1Array[3][i]


#       xline5_6[0]=x1Array[5][i]
#       xline5_6[1]=x1Array[6][i]
#       yline5_6[0]=-1*y1Array[5][i]
#       yline5_6[1]=-1*y1Array[6][i]
      v1line5_6[0]=x1Array[6][i]-x1Array[5][i]
      v1line5_6[1]=y1Array[6][i]-y1Array[5][i]

#       xline6_7[0]=x1Array[6][i]
#       xline6_7[1]=x1Array[7][i]
#       yline6_7[0]=-1*y1Array[6][i]
#       yline6_7[1]=-1*y1Array[7][i]
      v1line6_7[0]=x1Array[7][i]-x1Array[6][i]
      v1line6_7[1]=y1Array[7][i]-y1Array[6][i]


#       xline1_11[0]=x1Array[1][i]
#       xline1_11[1]=x1Array[11][i]
#       yline1_11[0]=-1*y1Array[1][i]
#       yline1_11[1]=-1*y1Array[11][i]
      v1line1_11[0]=x1Array[11][i]-x1Array[1][i]
      v1line1_11[1]=y1Array[11][i]-y1Array[1][i]


#       xline1_8[0]=x1Array[1][i]
#       xline1_8[1]=x1Array[8][i]
#       yline1_8[0]=-1*y1Array[1][i]
#       yline1_8[1]=-1*y1Array[8][i]
      v1line1_8[0]=x1Array[8][i]-x1Array[1][i]
      v1line1_8[1]=y1Array[8][i]-y1Array[1][i]


#       xline11_12[0]=x1Array[11][i]
#       xline11_12[1]=x1Array[12][i]
#       yline11_12[0]=-1*y1Array[11][i]
#       yline11_12[1]=-1*y1Array[12][i]
      v1line11_12[0]=x1Array[12][i]-x1Array[11][i]
      v1line11_12[1]=y1Array[12][i]-y1Array[11][i]


#       xline12_13[0]=x1Array[12][i]
#       xline12_13[1]=x1Array[13][i]
#       yline12_13[0]=-1*y1Array[12][i]
#       yline12_13[1]=-1*y1Array[13][i]
      v1line12_13[0]=x1Array[13][i]-x1Array[12][i]
      v1line12_13[1]=y1Array[13][i]-y1Array[12][i]

#       xline8_9[0]=x1Array[8][i]
#       xline8_9[1]=x1Array[9][i]
#       yline8_9[0]=-1*y1Array[8][i]
#       yline8_9[1]=-1*y1Array[9][i]
      v1line8_9[0]=x1Array[9][i]-x1Array[8][i]
      v1line8_9[1]=y1Array[9][i]-y1Array[8][i]

#       xline9_10[0]=x1Array[9][i]
#       xline9_10[1]=x1Array[10][i]
#       yline9_10[0]=-1*y1Array[9][i]
#       yline9_10[1]=-1*y1Array[10][i]
      v1line9_10[0]=x1Array[10][i]-x1Array[9][i]
      v1line9_10[1]=y1Array[10][i]-y1Array[9][i]

#       xline0_1[0]=x1Array[0][i]
#       xline0_1[1]=x1Array[1][i]
#       yline0_1[0]=-1*y1Array[0][i]
#       yline0_1[1]=-1*y1Array[1][i]
      v1line0_1[0]=x1Array[1][i]-x1Array[0][i]
      v1line0_1[1]=y1Array[1][i]-y1Array[0][i]



#       plt.plot(xline1_2,yline1_2,color='red')
#       plt.plot(xline1_5,yline1_5,color='red')
#       plt.plot(xline2_3,yline2_3,color='red')
#       plt.plot(xline3_4,yline3_4,color='red')
#       plt.plot(xline5_6,yline5_6,color='red')
#       plt.plot(xline6_7,yline6_7,color='red')
#       plt.plot(xline1_11,yline1_11,color='red')
#       plt.plot(xline1_8,yline1_8,color='red')
#       plt.plot(xline11_12,yline11_12,color='red')
#       plt.plot(xline12_13,yline12_13,color='red')
#       plt.plot(xline8_9,yline8_9,color='red')
#       plt.plot(xline9_10,yline9_10,color='red')
#       plt.plot(xline0_1,yline0_1,color='red')

#       xline1_2[0]=x2Array[1][i]
#       xline1_2[1]=x2Array[2][i]
#       yline1_2[0]=-1*y2Array[1][i]
#       yline1_2[1]=-1*y2Array[2][i]
      v2line1_2[0]=x2Array[2][i]-x2Array[1][i]
      v2line1_2[1]=y2Array[2][i]-y2Array[1][i]

#       xline1_5[0]=x2Array[1][i]
#       xline1_5[1]=x2Array[5][i]
#       yline1_5[0]=-1*y2Array[1][i]
#       yline1_5[1]=-1*y2Array[5][i]
      v2line1_5[0]=x2Array[5][i]-x2Array[1][i]
      v2line1_5[1]=y2Array[5][i]-y2Array[1][i]


#       xline2_3[0]=x2Array[2][i]
#       xline2_3[1]=x2Array[3][i]
#       yline2_3[0]=-1*y2Array[2][i]
#       yline2_3[1]=-1*y2Array[3][i]
      v2line2_3[0]=x2Array[3][i]-x2Array[2][i]
      v2line2_3[1]=y2Array[3][i]-y2Array[2][i]

#       xline3_4[0]=x2Array[3][i]
#       xline3_4[1]=x2Array[4][i]
#       yline3_4[0]=-1*y2Array[3][i]
#       yline3_4[1]=-1*y2Array[4][i]
      v2line3_4[0]=x2Array[4][i]-x2Array[3][i]
      v2line3_4[1]=y2Array[4][i]-y2Array[3][i]


#       xline5_6[0]=x2Array[5][i]
#       xline5_6[1]=x2Array[6][i]
#       yline5_6[0]=-1*y2Array[5][i]
#       yline5_6[1]=-1*y2Array[6][i]
      v2line5_6[0]=x2Array[6][i]-x2Array[5][i]
      v2line5_6[1]=y2Array[6][i]-y2Array[5][i]


#       xline6_7[0]=x2Array[6][i]
#       xline6_7[1]=x2Array[7][i]
#       yline6_7[0]=-1*y2Array[6][i]
#       yline6_7[1]=-1*y2Array[7][i]
      v2line6_7[0]=x2Array[7][i]-x2Array[6][i]
      v2line6_7[1]=y2Array[7][i]-y2Array[6][i]


#       xline1_11[0]=x2Array[1][i]
#       xline1_11[1]=x2Array[11][i]
#       yline1_11[0]=-1*y2Array[1][i]
#       yline1_11[1]=-1*y2Array[11][i]
      v2line1_11[0]=x2Array[11][i]-x2Array[1][i]
      v2line1_11[1]=y2Array[11][i]-y2Array[1][i]


#       xline1_8[0]=x2Array[1][i]
#       xline1_8[1]=x2Array[8][i]
#       yline1_8[0]=-1*y2Array[1][i]
#       yline1_8[1]=-1*y2Array[8][i]
      v2line1_8[0]=x2Array[8][i]-x2Array[1][i]
      v2line1_8[1]=y2Array[8][i]-y2Array[1][i]


#       xline11_12[0]=x2Array[11][i]
#       xline11_12[1]=x2Array[12][i]
#       yline11_12[0]=-1*y2Array[11][i]
#       yline11_12[1]=-1*y2Array[12][i]
      v2line11_12[0]=x2Array[12][i]-x2Array[11][i]
      v2line11_12[1]=y2Array[12][i]-y2Array[11][i]

#       xline12_13[0]=x2Array[12][i]
#       xline12_13[1]=x2Array[13][i]
#       yline12_13[0]=-1*y2Array[12][i]
#       yline12_13[1]=-1*y2Array[13][i]
      v2line12_13[0]=x2Array[13][i]-x2Array[12][i]
      v2line12_13[1]=y2Array[13][i]-y2Array[12][i]

#       xline8_9[0]=x2Array[8][i]
#       xline8_9[1]=x2Array[9][i]
#       yline8_9[0]=-1*y2Array[8][i]
#       yline8_9[1]=-1*y2Array[9][i]
      v2line8_9[0]=x2Array[9][i]-x2Array[8][i]
      v2line8_9[1]=y2Array[9][i]-y2Array[8][i]

#       xline9_10[0]=x2Array[9][i]
#       xline9_10[1]=x2Array[10][i]
#       yline9_10[0]=-1*y2Array[9][i]
#       yline9_10[1]=-1*y2Array[10][i]
      v2line9_10[0]=x2Array[10][i]-x2Array[9][i]
      v2line9_10[1]=y2Array[10][i]-y2Array[9][i]

#       xline0_1[0]=x2Array[0][i]
#       xline0_1[1]=x2Array[1][i]
#       yline0_1[0]=-1*y2Array[0][i]
#       yline0_1[1]=-1*y2Array[1][i]
      v2line0_1[0]=x2Array[1][i]-x2Array[0][i]
      v2line0_1[1]=y2Array[1][i]-y2Array[0][i]


      #print(x1Array[1][i])

#       plt.plot(xline1_2,yline1_2,color='black')
#       plt.plot(xline1_5,yline1_5,color='black')
#       plt.plot(xline2_3,yline2_3,color='black')
#       plt.plot(xline3_4,yline3_4,color='black')
#       plt.plot(xline5_6,yline5_6,color='black')
#       plt.plot(xline6_7,yline6_7,color='black')
#       plt.plot(xline1_11,yline1_11,color='black')
#       plt.plot(xline1_8,yline1_8,color='black')
#       plt.plot(xline11_12,yline11_12,color='black')
#       plt.plot(xline12_13,yline12_13,color='black')
#       plt.plot(xline8_9,yline8_9,color='black')
#       plt.plot(xline9_10,yline9_10,color='black')
#       plt.plot(xline0_1,yline0_1,color='black')

      try:
        theta[0]=180.0/3.14*math.acos( round( dot(v1line0_1, v2line0_1)/(norm(v1line0_1)*norm(v2line0_1)),3))
        theta[1]=180.0/3.14*math.acos(round( dot(v1line1_2, v2line1_2)/(norm(v1line1_2)*norm(v2line1_2)),3))
        theta[2]=180.0/3.14*math.acos(round( dot(v1line1_5, v2line1_5)/(norm(v1line1_5)*norm(v2line1_5)),3))
        theta[3]=180.0/3.14*math.acos(round( dot(v1line2_3, v2line2_3)/(norm(v1line2_3)*norm(v2line2_3)),3))
        theta[4]=180.0/3.14*math.acos(round( dot(v1line5_6, v2line5_6)/(norm(v1line5_6)*norm(v2line5_6)),3))
        theta[5]=180.0/3.14*math.acos( round(dot(v1line3_4, v2line3_4)/(norm(v1line3_4)*norm(v2line3_4)),3))
        theta[6]=180.0/3.14*math.acos(round( dot(v1line6_7, v2line6_7)/(norm(v1line6_7)*norm(v2line6_7)),3))
        theta[7]=180.0/3.14*math.acos(round( dot(v1line1_8, v2line1_8)/(norm(v1line1_8)*norm(v2line1_8)),3))
        theta[8]=180.0/3.14*math.acos( round(dot(v1line1_11, v2line1_11)/(norm(v1line1_11)*norm(v2line1_11)),3))
        theta[9]=180.0/3.14*math.acos(round( dot(v1line8_9, v2line8_9)/(norm(v1line8_9)*norm(v2line8_9)),3))
        theta[10]=180.0/3.14*math.acos( round(dot(v1line11_12, v2line11_12)/(norm(v1line11_12)*norm(v2line11_12)),3))
        theta[11]=180.0/3.14*math.acos(round(dot(v1line9_10, v2line9_10)/(norm(v1line9_10)*norm(v2line9_10)),3))
        theta[12]=180.0/3.14*math.acos(round( dot(v1line12_13, v2line12_13)/(norm(v1line12_13)*norm(v2line12_13)),3))
        #print('line23  ',theta )

      
#         plt.text(xline0_1[0],yline0_1[0], int(theta[0]), size=15, color='purple')
#         plt.text(xline1_2[0]-0.03,yline1_2[0], int(theta[1]), size=15, color='purple')
#         plt.text(xline1_5[0]+0.03,yline1_5[0], int(theta[2]), size=15, color='purple')
#         plt.text(xline2_3[0],yline2_3[0], int(theta[3]), size=15, color='purple')
#         plt.text(xline5_6[0],yline5_6[0], int(theta[4]), size=15, color='purple')
#         plt.text(xline3_4[0],yline3_4[0], int(theta[5]), size=15, color='purple')
#         plt.text(xline6_7[0],yline6_7[0], int(theta[6]), size=15, color='purple')
#         plt.text(xline1_8[0],yline1_8[0]+0.03, int(theta[7]), size=15, color='purple')
#         plt.text(xline1_11[0],yline1_11[0]-0.03, int(theta[8]), size=15, color='purple')
#         plt.text(xline8_9[0],yline8_9[0], int(theta[9]), size=15, color='purple')
#         plt.text(xline11_12[0],yline11_12[0], int(theta[10]), size=15, color='purple')
#         plt.text(xline9_10[0],yline9_10[0], int(theta[11]), size=15, color='purple')
#         plt.text(xline12_13[0],yline12_13[0], int(theta[12]), size=15, color='purple')
      except:
        print("An exception occurred") 

      #plt.show()
      sim=0

      newTime=time.time()
      progress=90.0
      if newTime-oldTime>5.0:
        oldTime=newTime
        try:
          ref.child(args.postID).set({'object':{'progress':progress}})
        except:
          print("File write exception from run_mod") 


      for angle in theta: 
        if math.isnan(angle) == False:
          sim=sim+getScore(angle)

      sim=sim/13
      simArr[i]=sim
      netSim=netSim+sim
    #  print('Score is ',sim)
    #  plt.savefig('/openPose/output/output_'+str(i)+'.png')

    #  plt.cla()
    netSim=netSim/minFrames  
    maxSim=np.argmax(simArr)
    minSim=np.argmin(simArr) 

    #print('Net Similarity:',netSim)
    db1 = firestore.client()
    result=db1.collection('copy_objects').document(args.postID).update({'score':netSim})
 
    font = cv2.FONT_HERSHEY_SIMPLEX
    top=0
    bottom=0
    for x in simArr:
      if x>80:
        top=top+x*30
        bottom=bottom+30
      else:
        top=top+x
        bottom=bottom+1
    
    if bottom>0:
      netSim=top/bottom 
    
    h1,w1= cv2.imread('/openPose/images_'+args.postID+'/f1rame_'+str(1)+'.png').shape[:2]
    h2,w2= cv2.imread('/openPose/images_'+args.postID+'/f2rame_'+str(1)+'.png').shape[:2]
    
    small_h=480
    small_w= 270
    big_h=1920
    big_w=1080
    orientation='vertical'
    x_score=100
    y_score=700
    if h1>w1:
      small_h=480
      small_w=270
    else:
      small_h=270
      small_w=480

    if h2>w2:
      big_h=1920
      big_w=1080
      orientation='vertical'
    else:
      big_h=1080
      big_w=1920
      x_score=350
      y_score=700
      orientation='horizontal'

    for i in range(0,fCount):
      s_img =cv2.resize(cv2.imread('/openPose/images_'+args.postID+'/f1rame_'+str(i)+'.png'),(small_w,small_h))
      l_img = cv2.resize(cv2.imread('/openPose/images_'+args.postID+'/f2rame_'+str(i)+'.png'),(big_w,big_h) )
      x_offset=y_offset=0
      l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
      cv2.putText(l_img, 'Score', (l_img.shape[1]-200,100), font, 2, (0,0,255), 2, cv2.LINE_AA)
      if len(simArr)>=fCount:
        cv2.putText(l_img, str(round(simArr[i],1)), (l_img.shape[1]-200,200), font, 2, (0,0, 255), 2, cv2.LINE_AA)
      cv2.imwrite('/openPose/output_'+args.postID+'/combo_'+str(i)+'.png',l_img)
    os.system('ffmpeg -i /openPose/output_'+args.postID+'/combo_%d.png -y -start_number 1 -c:v libx264 -pix_fmt yuv420p -y /openPose/output_'+args.postID+'/output_main.mp4')
    
    if netSim>=80:
      im = cv2.imread('/openPose/templates/3_star_'+orientation+'.png', 1)  
    elif netSim>=50 and netSim<80:
      im = cv2.imread('/openPose/templates/2_star_'+orientation+'.png', 1) 
    elif netSim>=20 and netSim<50:
      im = cv2.imread('/openPose/templates/1_star_'+orientation+'.png', 1)    
    else:
      im = cv2.imread('/openPose/templates/0_star_'+orientation+'.png', 1)
    if netSim>=0:
      cv2.putText(im, 'Score: '+str(round(netSim,1)), (x_score,y_score), font, 4, (255, 0, 0), 2, cv2.LINE_AA)
    else:
      cv2.putText(im, 'Score: Invalid Video', (10,300), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite('/openPose/output_'+args.postID+'/score.png', im)

    conf=firestore.client().collection('confidential').document('doc1').get().to_dict()
    ACCESS_KEY = conf['accessKeyId']
    SECRET_KEY = conf['secretAccessKey']
    session = Session(aws_access_key_id=ACCESS_KEY,
              aws_secret_access_key=SECRET_KEY)
    s3 = session.resource('s3')
    buck = s3.Bucket('mooplaystorage')

    doc3=firestore.client().collection('copy_objects').document(args.postID).get().to_dict()
    parentID=doc3['parentID'] 
    docList=[]
    docs = db1.collection(u'copy_objects').where(u'parentID', u'==', parentID).where(u'score', u'>=', 0).stream()

    for doc in docs:
 #   print(f'{doc.id} => {doc.to_dict()}')
      docList.append(doc.to_dict())

    rankList=[]
    sortedList=sorted(docList, key = lambda i: i['score'],reverse=True) 
    post_rank = next((index for (index, d) in enumerate(sortedList) if d["postID"] == args.postID), None)
    totalPosts=len(sortedList)
    if(len(sortedList)<=10):
        print("Case:1 total count<=10") 
        rankList=[] 
        count=1
        for i in sortedList:
            doc1={}
            doc1['rank']=count
            doc1['userName']=i['userName']
            doc1['userID']=i['userID']
            doc1['picUrl']=i['userPicUrl']
            doc1['score']=i['score']
            rankList.append(doc1)
            count=count+1
    elif(post_rank<=7):
        print("Case:2 post rank<8") 
        rankList=[]
        count=1
        for i in sortedList[0:10]:
            doc1={}
            doc1['rank']=count
            doc1['userName']=i['userName']
            doc1['userID']=i['userID']
            doc1['picUrl']=i['userPicUrl']
            doc1['score']=i['score']
            rankList.append(doc1)
            count=count+1
    else:
        rankList=[]
        count=1
        print("Case:3 total count>10 and post rank>7") 
        for i in sortedList[0:3]:
            doc1={}
            doc1['rank']=count
            doc1['userName']=i['userName']
            doc1['userID']=i['userID']
            doc1['picUrl']=i['userPicUrl']
            doc1['score']=i['score']
            rankList.append(doc1)
            count=count+1
        if(totalPosts-post_rank<=3):
            print("Case:3a post rank within final 3") 
            count=totalPosts-6
            for i in sortedList[totalPosts-7:totalPosts]:
                doc1={}
                doc1['rank']=count
                doc1['userName']=i['userName']
                doc1['userID']=i['userID']
                doc1['picUrl']=i['userPicUrl']
                doc1['score']=i['score']
                rankList.append(doc1)
                count=count+1
        else:
            print("Case:3b post rank not within final 3") 
            count=post_rank-2
            for i in sortedList[post_rank-3 :post_rank+4]:
                doc1={}
                doc1['rank']=count
                doc1['userName']=i['userName']
                doc1['userID']=i['userID']
                doc1['picUrl']=i['userPicUrl']
                doc1['score']=i['score']
                rankList.append(doc1)
                count=count+1
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)
    thickness = 5

    if h2>w2:
        #   Vertical case
        image = cv2.imread('/openPose/templates/blank_vertical.png')

        count=0
        
        for xx in rankList:
            fileName='{}'.format(xx['rank'])+'.jpg'
            buck.download_file( 'ProfilePics/ProfilePic-'+ xx['userID']+'.jpg',fileName)
            s_img =cv2.resize(cv2.imread(fileName),(100,100))

            mask = np.zeros(s_img.shape, dtype=np.uint8)
            x,y = 50, 50
            cv2.circle(mask, (x,y), 50, (255,255,255), -1)

            # Bitwise-and for ROI
            ROI = cv2.bitwise_and(s_img, mask)

            # Crop mask and turn background white
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            x,y,w,h = cv2.boundingRect(mask)
            result = ROI[y:y+h,x:x+w]
            mask = mask[y:y+h,x:x+w]
            result[mask==0] = (255,255,255)
            fileName2='{}'.format(xx['rank'])+'_cropped.jpg'
            cv2.imwrite(fileName2,result)
            y=174*(count+2)
            image[y-96-50:y-96+50,200:300]=result
            cv2.putText(image, '{}'.format(xx['rank']), (50,y-96), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, xx['userName'], (350,y-96), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, '{:.1f}'.format(xx['score']), (800,y-96), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.line(image, (0,y), (1080,y), color, thickness)
            count=count+1
            y=260*(1)
            cv2.putText(image, 'Rank', (40,y-96), font, 2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image,  'UserName', (320,y-96), font, 2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image,  'Score', (750,y-96), font, 2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image,  datetime.now().strftime("%d/%m/%Y %H:%M:%S"), (100,y-96-70), font, 1, (0, 0, 0), 4, cv2.LINE_AA)

        cv2.imwrite('/openPose/output_'+args.postID+'/table.png',image) 
    else:
        
        #   horizontal case
        image = cv2.imread('/openPose/templates/blank_horizontal.png')
        count=0
        for xx in rankList:
            fileName='{}'.format(xx['rank'])+'.jpg'
            buck.download_file( 'ProfilePics/ProfilePic-'+ xx['userID']+'.jpg',fileName)
            s_img =cv2.resize(cv2.imread(fileName),(60,60))

            mask = np.zeros(s_img.shape, dtype=np.uint8)
            x,y = 30, 30
            cv2.circle(mask, (x,y), 30, (255,255,255), -1)

            # Bitwise-and for ROI
            ROI = cv2.bitwise_and(s_img, mask)

            # Crop mask and turn background white
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            x,y,w,h = cv2.boundingRect(mask)
            result = ROI[y:y+h,x:x+w]
            mask = mask[y:y+h,x:x+w]
            result[mask==0] = (255,255,255)
            fileName2='{}'.format(xx['rank'])+'_cropped.jpg'
            cv2.imwrite(fileName2,result)
            y=98*(count+2)
            image[y-54-30:y-54+30,800:860]=result
            cv2.putText(image, '{}'.format(xx['rank']), (50,y-40), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, xx['userName'], (900,y-40), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, '{:.1f}'.format(xx['score']), (1600,y-40), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.line(image, (0,y), (1920,y), color, thickness)
            count=count+1
            y=140*(1)
            cv2.putText(image, 'Rank', (40,y-54), font, 2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image,  'UserName', (850,y-54), font, 2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image,  'Score', (1550,y-54), font, 2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image,  datetime.now().strftime("%d/%m/%Y %H:%M:%S"), (400,y-54-50), font, 1, (0, 0, 0), 4, cv2.LINE_AA)

        cv2.imwrite('/openPose/output_'+args.postID+'/table.png',image)    



    os.system('ffmpeg -loop 1 -i /openPose/output_'+args.postID+'/score.png -c:v libx264 -t 5 -pix_fmt yuv420p -y /openPose/output_'+args.postID+'/score.mp4')
    os.system('ffmpeg -loop 1 -i /openPose/templates/best_moments_'+orientation+'.png -c:v libx264 -t 2 -pix_fmt yuv420p -y /openPose/output_'+args.postID+'/best_moments.mp4')
    os.system('ffmpeg -loop 1 -i /openPose/templates/poor_moments_'+orientation+'.png -c:v libx264 -t 2 -pix_fmt yuv420p -y /openPose/output_'+args.postID+'/poor_moments.mp4')
    os.system('ffmpeg -loop 1 -i /openPose/output_'+args.postID+'/combo_'+str(minSim)+'.png -c:v libx264 -t 3 -pix_fmt yuv420p -y /openPose/output_'+args.postID+'/poor_moments1.mp4')
    os.system('ffmpeg -loop 1 -i /openPose/output_'+args.postID+'/combo_'+str(maxSim)+'.png -c:v libx264 -t 3 -pix_fmt yuv420p -y /openPose/output_'+args.postID+'/best_moments1.mp4')
    
    os.system('ffmpeg -loop 1 -i /openPose/output_'+args.postID+'/table.png -c:v libx264 -t 5 -pix_fmt yuv420p -y /openPose/output_'+args.postID+'/table.mp4')

    fClip = open('/openPose/output_'+args.postID+'/clipList.txt', "w")
    fClip.write('file \'/openPose/output_'+args.postID+'/output_main.mp4\'\n' )
    fClip.write('file \'/openPose/output_'+args.postID+'/score.mp4\'\n' )
    fClip.write('file \'/openPose/output_'+args.postID+'/best_moments.mp4\'\n' )
    fClip.write('file \'/openPose/output_'+args.postID+'/best_moments1.mp4\'\n' )
    fClip.write('file \'/openPose/output_'+args.postID+'/poor_moments.mp4\'\n' )
    fClip.write('file \'/openPose/output_'+args.postID+'/poor_moments1.mp4\'\n')
    fClip.write('file \'/openPose/output_'+args.postID+'/table.mp4\'' )
    fClip.close()


    os.system('ffmpeg -f concat -safe 0 -i /openPose/output_'+args.postID+'/clipList.txt -c copy /openPose/output_'+args.postID+'/output_full1.mp4 -y')

    os.system('ffmpeg -i /app/'+args.postID+'_input1.mp4 -q:a 0 -map a /openPose/output_'+args.postID+'/audio1.mp3 -y') 
    os.system('ffmpeg -i "concat:/openPose/output_' +args.postID+'/audio1.mp3|/openPose/templates/mooplay_theme.mp3" -acodec copy /openPose/output_'+args.postID+'/audio.mp3 -y' ) 
    os.system('ffmpeg -i /openPose/output_'+args.postID+'/output_full1.mp4 -i /openPose/output_'+args.postID+'/audio.mp3 -c:v copy -c:a aac /openPose/output_'+args.postID+'/output_full.mp4 -y')
    #file upload and firestore update
    videoName='Video-'+args.postID+'.mp4' 




    
    
    buck.upload_file('/openPose/output_'+args.postID+'/output_full.mp4','ComparisonVideos/'+videoName,ExtraArgs={'ACL':'public-read'})
    logger.info('Comparison video uploaded to AWS')
    com_url="https://mooplaystorage.s3.ap-south-1.amazonaws.com/"+'ComparisonVideos/'+videoName
    db1 = firestore.client()
    result=db1.collection('copy_objects').document(args.postID).update({'score':netSim})
    result=db1.collection('copy_objects').document(args.postID).update({'comparison_video_url':com_url})
   # print(result)
    logger.info('Comparison URl updated in Firebase result  is %s' % str(result))
    
    progress=100.0
    try:
      ref.child(args.postID).set({'object':{'progress':progress}})
      logger.info('progress is %s' % str(progress))  
     # os.system('rm -r /openPose/images_'+args.postID)
     # os.system('rm -r /openPose/output_'+args.postID)
    except:
      print("File write exception from run_mod: ",sys.exc_info()[0]) 
    
    try:
      ref1=db.reference('serverStatus/'+str(gma())+'/runningJobs')
      ref1.set(ref1.get()-1)
    except:
      print("Firebase write exception from run_mod: ",sys.exc_info()[0]) 

