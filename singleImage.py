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

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--imagePath', type=str, default='./images/')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=432x368, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

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

    f1Count = len(glob.glob1('/openPose/images',"f1rame_*.png"))
    f2Count = len(glob.glob1('/openPose/images',"f2rame_*.png"))

    # processing first video
    data1 = {}
    data1['parts'] = []

    data = {}
    data['frames'] = []

    for i in range(1,f1Count): 
      # estimate human poses from a single image !
      image = common.read_imgfile('/openPose/images/f1rame_'+str(i)+'.png', None, None)
      if image is None:
          logger.error('Image can not be read, path=%s' % args.imagePath)
          sys.exit(-1)
      t = time.time()
      humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
      elapsed = time.time() - t
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

      logger.info('inference image f1rame_: %s in %.4f seconds.' % (str(i), elapsed))

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

    for i in range(1,f2Count): 
      # estimate human poses from a single image !
      image = common.read_imgfile('/openPose/images/f2rame_'+str(i)+'.png', None, None)
      if image is None:
          logger.error('Image can not be read, path=%s' % args.imagePath)
          sys.exit(-1)
      t = time.time()
      humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
      elapsed = time.time() - t
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

      logger.info('inference image f2rame_: %s in %.4f seconds.' % (str(i), elapsed))

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

    fCount = f1Count if f1Count < f2Count else f2Count 
      
    for i in range(1,290):
      s_img =cv2.resize(cv2.imread('/openPose/output/f1rame_'+str(i)+'.png'),(270,480))
      l_img = cv2.resize(cv2.imread('/openPose/output/f2rame_'+str(i)+'.png'),(1080,1920) )
      x_offset=y_offset=0
      l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
      cv2.imwrite('/openPose/output/combo_'+str(i)+'.png',cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB))

    os.system('ffmpeg -i /openPose/output/combo_%d.png -y -start_number 1 -vf scale=400:800 -c:v libx264 -pix_fmt yuv420p /openPose/output/out.mp4')
  