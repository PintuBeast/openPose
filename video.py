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
   

       #video 1 split into frames

    try: 
      os.mkdir('content/openPose/images') 
    except OSError as error: 
      print(error)   


    try: 
      os.mkdir('content/openPose/output') 
    except OSError as error: 
      print(error) 

    src = cv2.VideoCapture('./videos/input1.mp4')
    fps = src.get(cv2.CAP_PROP_FPS)

    frame_num = 0
    while(frame_num< int(src.get(cv2.CAP_PROP_FRAME_COUNT))):
      # Capture frame-by-frame
      ret, frame = src.read()

      # Saves image of the current frame in jpg file
      name = './images/f1rame_' + str(frame_num) + '.png'
      #print ('Creating...' + name)
      cv2.imwrite(name, frame)

      # To stop duplicate images
      frame_num += 1

      # When everything done, release the capture
    src.release()
    cv2.destroyAllWindows()

    f1Count = len(glob.glob1('./images',"f1rame_*.png"))

    for i in range(0,f1Count): 
 
      # estimate human poses from a single image !
      image = common.read_imgfile('./images/f1rame_'+str(i)+'.png', None, None)
      if image is None:
          logger.error('Image can not be read')
          sys.exit(-1)
      t = time.time()
      humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
      
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
            
     
     
      image = TfPoseEstimator.draw_humans(image, humans[np.argmax(measure1):np.argmax(measure1)+1], imgcopy=False)
      cv2.imwrite('./output/f1rame_'+str(i)+'.png',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    os.system('ffmpeg -i ./output/f1rame_%d.png -y -start_number 1 -c:v libx264 -pix_fmt yuv420p -y ./output/output_main.mp4')



        
        
        
        
        
