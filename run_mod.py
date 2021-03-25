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
    parser.add_argument('--imagePath', type=str, default='./images/')
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
    
    os.system('rm -r /openPose/images')

    try: 
      os.mkdir('/openPose/images') 
    except OSError as error: 
      print(error)   

    os.system('rm -r /openPose/output')

    try: 
      os.mkdir('/openPose/output') 
    except OSError as error: 
      print(error) 
    
   #video 1 split into frames

    src = cv2.VideoCapture('/app/input1.mp4')
    fps = src.get(cv2.CAP_PROP_FPS)

    frame_num = 0
    while(frame_num< int(src.get(cv2.CAP_PROP_FRAME_COUNT))):
      # Capture frame-by-frame
      ret, frame = src.read()

      # Saves image of the current frame in jpg file
      name = '/openPose/images/f1rame_' + str(frame_num) + '.png'
      print ('Creating...' + name)
      cv2.imwrite(name, frame)

      # To stop duplicate images
      frame_num += 1

      # When everything done, release the capture
    src.release()
    cv2.destroyAllWindows()



   #video 2 split into frames

    src = cv2.VideoCapture('/app/input2.mp4')
    fps = src.get(cv2.CAP_PROP_FPS)

    frame_num = 0
    while(frame_num< int(src.get(cv2.CAP_PROP_FRAME_COUNT))):
      # Capture frame-by-frame
      ret, frame = src.read()

      # Saves image of the current frame in jpg file
      name = '/openPose/images/f2rame_' + str(frame_num) + '.png'
      print ('Creating...' + name)
      cv2.imwrite(name, frame)

      # To stop duplicate images
      frame_num += 1

      # When everything done, release the capture
    src.release()
    cv2.destroyAllWindows()


    f1Count = len(glob.glob1('/openPose/images',"f1rame_*.png"))
    f2Count = len(glob.glob1('/openPose/images',"f2rame_*.png"))

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
      image = common.read_imgfile('/openPose/images/f1rame_'+str(i)+'.png', None, None)
      if image is None:
          logger.error('Image can not be read, path=%s' % args.imagePath)
          sys.exit(-1)
      t = time.time()
      humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
      elapsed = time.time() - t
      data1 = {}
      data1['parts'] = []
      for human in humans:
            # draw point
            for ii in range(common.CocoPart.Background.value):
                if ii not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[ii]
               # center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                
                
                data1['parts'].append({
                'id': ii,
                'x': body_part.x,
                'y': body_part.y
                })
            break    

      #logger.info('inference image f1rame_: %s in %.4f seconds.' % (str(i), elapsed))
      #print('inference f1_rame_'str(i),' is 'elapsed, 'seconds')
      image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
      cv2.imwrite('/openPose/output/f1rame_'+str(i)+'.png',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
     
      data['frames'].append({
        'num': i,
        'array':data1}
      )

    with open('/openPose/output/data1.json', 'w') as outfile:
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
      image = common.read_imgfile('/openPose/images/f2rame_'+str(i)+'.png', None, None)
      if image is None:
          logger.error('Image can not be read, path=%s' % args.imagePath)
          sys.exit(-1)
      t = time.time()
      humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
      elapsed = time.time() - t
      data1 = {}
      data1['parts'] = []
      for human in humans:
            # draw point
            for ii in range(common.CocoPart.Background.value):
                if ii not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[ii]
               # center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                
                
                data1['parts'].append({
                'id': ii,
                'x': body_part.x,
                'y': body_part.y
                })
            break    

      #logger.info('inference image f2rame_: %s in %.4f seconds.' % (str(i), elapsed))

      image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
      cv2.imwrite('/openPose/output/f2rame_'+str(i)+'.png',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
     
      data['frames'].append({
        'num': i,
        'array':data1}
      )

    with open('/openPose/output/data2.json', 'w') as outfile:
      json.dump(data, outfile)  

    #os.system('ffmpeg -i /openPose/output/f1rame_%d.png -y -start_number 1 -vf scale=400:800 -c:v libx264 -pix_fmt yuv420p /openPose/output/out1.mp4')
    #os.system('ffmpeg -i /openPose/output/f2rame_%d.png -y -start_number 1 -vf scale=400:800 -c:v libx264 -pix_fmt yuv420p /openPose/output/out2.mp4')

    
      
    for i in range(0,fCount):
      s_img =cv2.resize(cv2.imread('/openPose/output/f1rame_'+str(i)+'.png'),(270,480))
      l_img = cv2.resize(cv2.imread('/openPose/output/f2rame_'+str(i)+'.png'),(1080,1920) )
      x_offset=y_offset=0
      l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
      cv2.imwrite('/openPose/output/combo_'+str(i)+'.png',cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB))
    os.system('ffmpeg -i /openPose/output/combo_%d.png -y -start_number 1 -vf scale=400:800 -c:v libx264 -pix_fmt yuv420p -y /openPose/output/output_main.mp4')
    
   
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

    with open('/openPose/output/data1.json') as f1:
      frame_array1= json.load(f1)

    f1Count=  len(frame_array1['frames'])

    with open('/openPose/output/data2.json') as f2:
      frame_array2= json.load(f2)

    f2Count=  len(frame_array2['frames']) 
    pose2=np.zeros((18*2*f2Count,))
    print('frame:1:',f1Count,'frame 2:',f2Count)
    minFrames=f1Count if f1Count < f2Count else f2Count
    
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
        theta[0]=180.0/3.14*math.acos( dot(v1line0_1, v2line0_1)/(norm(v1line0_1)*norm(v2line0_1)))
        theta[1]=180.0/3.14*math.acos( dot(v1line1_2, v2line1_2)/(norm(v1line1_2)*norm(v2line1_2)))
        theta[2]=180.0/3.14*math.acos( dot(v1line1_5, v2line1_5)/(norm(v1line1_5)*norm(v2line1_5)))
        theta[3]=180.0/3.14*math.acos( dot(v1line2_3, v2line2_3)/(norm(v1line2_3)*norm(v2line2_3)))
        theta[4]=180.0/3.14*math.acos( dot(v1line5_6, v2line5_6)/(norm(v1line5_6)*norm(v2line5_6)))
        theta[5]=180.0/3.14*math.acos( dot(v1line3_4, v2line3_4)/(norm(v1line3_4)*norm(v2line3_4)))
        theta[6]=180.0/3.14*math.acos( dot(v1line6_7, v2line6_7)/(norm(v1line6_7)*norm(v2line6_7)))
        theta[7]=180.0/3.14*math.acos( dot(v1line1_8, v2line1_8)/(norm(v1line1_8)*norm(v2line1_8)))
        theta[8]=180.0/3.14*math.acos( dot(v1line1_11, v2line1_11)/(norm(v1line1_11)*norm(v2line1_11)))
        theta[9]=180.0/3.14*math.acos( dot(v1line8_9, v2line8_9)/(norm(v1line8_9)*norm(v2line8_9)))
        theta[10]=180.0/3.14*math.acos( dot(v1line11_12, v2line11_12)/(norm(v1line11_12)*norm(v2line11_12)))
        theta[11]=180.0/3.14*math.acos( dot(v1line9_10, v2line9_10)/(norm(v1line9_10)*norm(v2line9_10)))
        theta[12]=180.0/3.14*math.acos( dot(v1line12_13, v2line12_13)/(norm(v1line12_13)*norm(v2line12_13)))
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
      print('Score is ',sim)
    #  plt.savefig('/openPose/output/output_'+str(i)+'.png')

    #  plt.cla()
    netSim=netSim/minFrames  
    maxSim=np.argmax(simArr)
    minSim=np.argmin(simArr) 

    print('Net Similarity:',netSim)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if netSim>=80:
      im = cv2.imread('/openPose/templates/1_star.png', 1)  
    if netSim>=50 and netSim<80:
      im = cv2.imread('/openPose/templates/2_star.png', 1) 
    else:
      im = cv2.imread('/openPose/templates/1_star.png', 1)

    cv2.putText(im, 'Score: '+str(netSim), (10,300), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite('/openPose/output/score.png', im)
    os.system('ffmpeg -loop 1 -i /openPose/output/score.png -c:v libx264 -t 5 -pix_fmt yuv420p -y /openPose/output/score.mp4')
    os.system('ffmpeg -loop 1 -i /openPose/templates/best_moments.png -c:v libx264 -t 5 -pix_fmt yuv420p -y /openPose/output/best_moments.mp4')
    os.system('ffmpeg -loop 1 -i /openPose/templates/poor_moments.png -c:v libx264 -t 5 -pix_fmt yuv420p -y /openPose/output/poor_moments.mp4')
    os.system('ffmpeg -loop 1 -i /openPose/output/combo_'+str(minSim)+'.png -c:v libx264 -t 5 -pix_fmt yuv420p -y /openPose/output/poor_moments1.mp4')
    os.system('ffmpeg -loop 1 -i /openPose/output/combo_'+str(maxSim)+'.png -c:v libx264 -t 5 -pix_fmt yuv420p -y /openPose/output/best_moments1.mp4')
    os.system('ffmpeg -f concat -safe 0 -i /openPose/clipList.txt -c copy /openPose/output/output_full.mp4 -y')
    
    #file upload and firestore update
    videoName='Video-'+args.postID+'.mp4' 
    bucket = storage.bucket()
    blob = bucket.blob('ComparisonVideos/'+videoName)
    outfile='/openPose/output/output_full.mp4'
    with open(outfile, 'rb') as my_file:
      blob.upload_from_file(my_file)
    db = firestore.client()
    result=db.collection('copy_objects').document('1eNfmDW05yOZNdGTB7hx').update({'comparison_video_url':blob.public_url,'score':netSim})
   # print(result)
   # logger.info('upload and update result  is %s' % str(result))
    
    progress=100.0
    try:
      ref.child(args.postID).set({'object':{'progress':progress}})
      print('progress is:',str(progress))
      logger.info('progress is %s' % str(progress))
    except:
      print("File write exception from run_mod: ",sys.exc_info()[0]) 
    
