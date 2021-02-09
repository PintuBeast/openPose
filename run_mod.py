import argparse
import logging
import sys
import time

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

    for i in range(1,10): 
      # estimate human poses from a single image !
      image = common.read_imgfile(args.imagePath+'frame_'+str(i)+'.png', None, None)
      if image is None:
          logger.error('Image can not be read, path=%s' % args.imagePath)
          sys.exit(-1)

      t = time.time()
      humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
      elapsed = time.time() - t

      logger.info('inference image: %s in %.4f seconds.' % (args.image.Path, elapsed))

      image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
      cv2.imwrite('./output/frame_'+str(i)+'.png',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

