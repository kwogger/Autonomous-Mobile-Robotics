#!/usr/bin/env python
'''
Extended Kalman Filter Function
For Lab 2 of ME 597

Created on 2010-11-03

@author: Jamie Bragg
'''

import math
import numpy as np
import roslib; roslib.load_manifest('grp6lab2')
import rospy

lat_orig = 43.472209667
long_orig = -80.5398025

'''Right now, this code will base 0,0 at the south corner of E5
'''

def ekf_gps(x, S, u, R, y, Q, t):
  '''General Extended Kalman Filter for use with either encoders, IPS or GPS.

  Args:
  x: current state
  S: covariance of the current state
  u: latest control input
  R: covariance of the command
  y: sensor reading
  Q: covariance of the sensor reading
  t: last time this function was called, for the purposes of dt

  Returns:
    mu, the best guess of current state (position x,y and heading)
    S, Covariance of this guess
  '''
  dt = rospy.get_time() - t
  Q = [[ 1.27563225105601E-23, 3.12522321280352E-23, 0],
       [ 3.12522321280352E-23, 8.13638415591719E-23, 0],
       [ 0, 0                   , 10.1834384245562]]
  R = [[ 0.14162522524774, 0.0135871726258, -0.01885776500978],
       [ 0.0135871726258, 0.00130502292838, -0.0018274115723],
       [ -0.01885776500978, -0.0018274115723, 0.0058360123874]]


  #Motion Model, linearized about current state
  G = [[1, 0, -u[1] * math.sin(x[3]) * dt],
       [0, 1, u[1] * math.sin(x[3]) * dt],
       [0, 0, u[1] * math.sec(u[2]) * dt]]

  #Prediction Step, using linearized model
  mup = [x[1] + u[1] * math.cos(x[3]) * dt,
         x[2] + u[1] * math.sin(x[3]) * dt,
         x[3] + ((u[1] * math.tan(u[2]) * dt) / L)]
  Sp = np.dot(np.dot(G, S), np.linalg.inv(G)) + R

  #Measurement Model Linearized about current prediction
  H = [[1 / (6365 * (mup[1] ^ 2 / 6365 ^ 2)), 0, 0],
       [0, 1 / (6365 * (mup[2] ^ 2 / 6365 ^ 2)), 0],
       [0, 0, 1]]
  C = [np.arcsin(mup[1] / 6365) + lat_orig, np.arcsin(mup[2] / 6365) + long_orig, mup[3]]

  #Calculate Kalman gain
  K = np.dot(Sp, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(Sp, H.T)) + Q)))

  return {
      'mu': mup + np.dot(K, (y - C)),
      'S': np.dot(np.eye(3) - np.dot(K, H), Sp),
      }


def ekf_enc(x, S, u, R, y, Q, t):
  '''No longer a generalized EKF. ;_;
  
  '''
  dt = rospy.get_time() - t
  Q = 0.0000380112014723476
  R = [[ 0.14162522524774, 0.0135871726258, -0.01885776500978],
       [ 0.0135871726258, 0.00130502292838, -0.0018274115723],
       [ -0.01885776500978, -0.0018274115723, 0.0058360123874]]
  
  #Motion Model, linearized about current state
  G = [[1, 0, -u[1] * math.sin(x[3]) * dt],
       [0, 1, u[1] * math.sin(x[3]) * dt],
       [0, 0, u[1] * math.sec(u[2]) * dt]]

  #Prediction Step, using linearized model
  mup = [x[1] + u[1] * math.cos(x[3]) * dt,
         x[2] + u[1] * math.sin(x[3]) * dt,
         x[3] + ((u[1] * math.tan(u[2]) * dt) / L)]
  Sp = np.dot(np.dot(G, S), np.linalg.inv(G)) + R

  #Measurement Model Linearized about current prediction
  H = [[1 / np.cos(mup[3]), 0, 0],
      [0, 1 / np.sin(mup[3]), 0],
      [0, 0, 1]]
  C = [mup[1] * np.arcsin(mup[3]), mup[2] * np.arccos(mup[3]), mup[3]]

  #Calculate Kalman Gain
  K = np.dot(Sp, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(Sp, H.T)) + Q)))

  return {
      'mu': mup + np.dot(K, (y - C)),
      'S': np.dot(np.eye(3) - np.dot(K, H), Sp),
      }

