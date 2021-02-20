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
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    
    os.sysytem('rm -r /openPose/images')

    try: 
      os.mkdir('/openPose/images') 
    except OSError as error: 
      print(error)   

   #video 1 split into frames

    src = cv2.VideoCapture('./videos/input1.mp4')
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

    src = cv2.VideoCapture('./videos/input2.mp4')
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

    # processing first video
    data1 = {}
    data1['parts'] = []

    data = {}
    data['frames'] = []

    for i in range(1,f1Count): 
      # estimate human poses from a single image !
      image = common.read_imgfile(args.imagePath+'f1rame_'+str(i)+'.png', None, None)
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
                'x': int(body_part.x),
                'y': int(body_part.y)
                })
            break    

      logger.info('inference image: %s in %.4f seconds.' % (args.imagePath, elapsed))

      image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
      cv2.imwrite('/openPose/output/f1rame_'+str(i)+'.png',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
     
      data['frames'].append({
        'num': i,
        'array':data1}
      )

    with open('data1.json', 'w') as outfile:
      json.dump(data, outfile)





    # processing second video
    data1 = {}
    data1['parts'] = []

    data = {}
    data['frames'] = []

    for i in range(1,f2Count): 
      # estimate human poses from a single image !
      image = common.read_imgfile(args.imagePath+'f2rame_'+str(i)+'.png', None, None)
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
                'x': int(body_part.x),
                'y': int(body_part.y)
                })
            break    

      logger.info('inference image: %s in %.4f seconds.' % (args.imagePath, elapsed))

      image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
      cv2.imwrite('/openPose/output/f2rame_'+str(i)+'.png',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
     
      data['frames'].append({
        'num': i,
        'array':data1}
      )

    with open('data2.json', 'w') as outfile:
      json.dump(data, outfile)      

      