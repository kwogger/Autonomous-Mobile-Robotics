'''Extended Kalman Filter

Created on 2010-11-01

@author: Jamie Bragg

'''

import math
import roslib; roslib.load_manifest('grp6')
import rospy
import numpy as np
import scipy as sp


'''  according to the internets the lat/long of Waterloo is 
     42 28' 0'' N / 80 32' 0'' W 
'''

def ekf( x, u, S, gps, enc)
  '''
  Extended Kalman Filter for localisation using GPS data
  x is a dictionary of the current state
  u is the control input
  S is the prior covariance
  gps is the gps data
  enc is the encoder data

  returns mu: the best guess position and heading
  and S: covariance of this guess

  Need to define discrete motion model Ad (matrix)
  Measurement model for encoder and encoder + GPS
  R matrix (covariance of command error)
  Q matrix (covariance of GPS data error)

  '''
  #Motion Model
  Ad = 
  n=length(x)
  #Measurement Model
  
  
  #Prediction
  mup = np.dot(Ad,mu)
  Sp = np.dot(np.dot(Ad,S),Ad.T) + R
  
  #Linearization
  Ht = [(mup[1]/math.sqrt(mup[1]^2 + mup[3]^2)) 0 (mup[3]/math.sqrt(mup[1]^2 + mup[3]^2))]
  
  #Measurement
  K = np.dot(np.dot(Sp,Ht.T),sp.linalg.inv(np.dot(np.dot(Ht,Sp),Ht.T)+Q))
  #done in return
  #mu = mup + dot(K,
  S = np.dot(eye(n)-np.dot(K,Ht),Sp)
  
  #Store Results
  return{
    'mu':mup + dot(K,
    'S':S
    }

