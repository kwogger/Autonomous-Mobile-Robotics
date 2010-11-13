'''
Blob Tracking with Hue Segmentation.

Adapted from MatLab code by Stephen Waslander.

@version: 1.0
@author: Michael Kwan
'''

import cv
import numpy

def cv2array(im): 
  depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }
  a = numpy.fromstring(
         im.tostring(),
         dtype=depth2dtype[im.depth],
         count=im.width * im.height * im.nChannels) 
  a.shape = (im.height, im.width, im.nChannels) 
  return a 

def array2cv(a): 
  dtype2depth = { 
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    } 
  try: 
    nChannels = a.shape[2] 
  except: 
    nChannels = 1 
  cv_im = cv.CreateImageHeader((a.shape[1], a.shape[0]), dtype2depth[str(a.dtype)], nChannels)
  cv.SetData(cv_im, a.tostring(), a.dtype.itemsize * nChannels * a.shape[1])
  return cv_im 


if __name__ == '__main__':
  video = cv.CaptureFromFile('cones.avi')
  frame_cnt = cv.GetCaptureProperty(video, cv.CV_CAP_PROP_FRAME_COUNT)
  print 'frames: %i' % frame_cnt
  # TODO(mkwan): allocate memory for the processed video here
  
  hue = 10 / 256
  hue_window = 10 / 256
  hue_max = hue - hue_window
  hue_min = hue + hue_window
  
  frame_index = 0
  frame = cv.QueryFrame(video)
  while frame:
    # Process Frame
    hsv = cv.CreateImage(cv.GetSize(frame), frame.depth, frame.nChannels)
    cv.CvtColor(frame, hsv, cv.CV_RGB2HSV)
    hsv_ary = cv2array(hsv) / 255.0
    
    
    print "%i / %i" % (frame_index, frame_cnt)
    frame_index += 1
    frame = cv.QueryFrame(video)
