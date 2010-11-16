'''
Created on 2010-11-15

@author: Michael Kwan
'''
import collections
import controller
import csv
import math
import numpy as np
from matplotlib import pyplot


VELOCITY_STICTION_OFFSET = 18
DRIVING_DISTANCE = 10.0
ROBOT_LENGTH = 0.238


# EKF Constants
EKF_CONSTS = {
#    'R': np.array([
#        [0.14162522524774, 0.0135871726258, -0.01885776500978],
#        [0.0135871726258, 0.00130502292838, -0.0018274115723],
#        [-0.01885776500978, -0.0018274115723, 0.0058360123874],
#        ]),
    'R': np.array([
        [0.14162522524774, 0, 0],
        [0, 0.00130502292838, 0],
        [0, 0, 0.0058360123874],
        ]),
    'G': lambda x, u, dt: np.array([
        [1, 0, -u[0] * math.sin(x[2]) * dt],
        [0, 1, u[0] * math.cos(x[2]) * dt],
        [0, 0, 1],
        ]),
    'mu_p': lambda x, u, dt: np.array([
        x[0] + u[0] * math.cos(x[2]) * dt,
        x[1] + u[0] * math.sin(x[2]) * dt,
        x[2] + ((u[0] * math.tan(u[1]) * dt) / 0.238)
        ]),
    }
EKF_CONSTS_GPS = {
    'Q': np.array([
        [4.22603233, 8.1302549, -0.05544],
        [8.13025, 16.192, -0.10088],
        [-0.05544, -0.10088, 0.003102*1e9],
        ]),
    'H': lambda mu_p, x: np.array([
        [0, 1 / 111101.911587005, 0],
        [1 / 80913.694760278, 0, 0],
        [0, 0, -1],
        ]),
    'h': lambda lat_orig, long_orig: lambda mu_p, x: np.array([
         mu_p[1] / 111101.911587005 + lat_orig,
         mu_p[0] / 80913.694760278 + long_orig,
         math.pi / 2 - mu_p[2],
         ])
    }
EKF_CONSTS_GPS.update(EKF_CONSTS)
EKF_CONSTS_ENC = {
    'Q': 0.0000380112014723476,
    'H': lambda mu_p, x: np.array([[
        x[0] + (mu_p[0] - x[0]) / np.sqrt((mu_p[0] - x[0]) ** 2 + (mu_p[1] - x[1]) ** 2),
        x[1] + (mu_p[1] - x[1]) / np.sqrt((mu_p[0] - x[0]) ** 2 + (mu_p[1] - x[1]) ** 2),
        0,
        ]]),
    'h': lambda mu_p, x: np.sqrt((mu_p[0] - x[0]) ** 2 + (mu_p[1] - x[1]) ** 2)
    }
EKF_CONSTS_ENC.update(EKF_CONSTS)


if __name__ == '__main__':
  # Load the data
  enc_reader = csv.reader(open('enc.csv', 'r'))
  gps_reader = csv.reader(open('gps.csv', 'r'))
  enc_data = collections.deque()
  gps_data = collections.deque()

  prev_tick = None
  for t, tick, vel_cmd, turn_cmd in enc_reader:
    if prev_tick is None:
      prev_tick = int(tick)
    else:
      enc_data.append((float(t) / 1e9, (int(tick) - prev_tick) * controller.METER_PER_TICK, float(turn_cmd)))
      prev_tick = int(tick)
  for t, lat, long, alt, track, err_track, speed, vel_cmd, turn_cmd in gps_reader:
    if not t == '0.0':
#      if float(track) == 0:
#        track = '60'
      gps_data.append((float(t) / 1e9, float(lat), float(long), float(track) / 180 * math.pi, float(turn_cmd)))
  sorted_data = collections.deque()
  while len(enc_data) > 0 and len(gps_data) > 0:
    if enc_data[0][0] > gps_data[0][0]:
      sorted_data.append(gps_data.popleft())
    else:
      sorted_data.append(enc_data.popleft())
  if len(enc_data) > 0:
    for data in enc_data:
      sorted_data.append(data)
  else:
    for data in gps_data:
      sorted_data.append(data)

  # Setup EKF
  ekf_data = {
      'mu': np.array([0, 0, 0.69800502]),
      'S': np.eye(3),
      }
#  u = np.array([0.4, 0])
  mup_data = [ekf_data['mu']]
  S_data = [ekf_data['S']]
  prev_t = 1289870797.29

  for data in sorted_data:
    if len(data) == 5:
#      continue
      ekf_data = controller.ekf(
          ekf_data['mu'],
          np.array([data[1], data[2], data[3]]),
          ekf_data['S'],
          EKF_CONSTS_GPS['Q'],
          np.array([0.4, data[4]]),
          EKF_CONSTS['R'],
          EKF_CONSTS['G'],
          EKF_CONSTS['mu_p'],
          EKF_CONSTS_GPS['H'],
          EKF_CONSTS_GPS['h'](43.4723815, -80.53945933),
          data[0],
          prev_t,
          )
    else:
#      continue
      ekf_data = controller.ekf(
          ekf_data['mu'],
          np.array([data[1]]),
          ekf_data['S'],
          EKF_CONSTS_ENC['Q'],
          np.array([0.4, data[2]]),
          EKF_CONSTS['R'],
          EKF_CONSTS['G'],
          EKF_CONSTS['mu_p'],
          EKF_CONSTS_ENC['H'],
          EKF_CONSTS_ENC['h'],
          data[0],
          prev_t,
          )
    prev_t = data[0]
    mup_data.append(ekf_data['mu'])
    S_data.append(ekf_data['S'])
  csv.writer(open('mup_est.csv', 'w')).writerows(mup_data)
  csv.writer(open('S_est.csv', 'w')).writerows(S_data)

  pyplot.figure(1)
  pyplot.clf()
  pyplot.hold(True)
  pyplot.plot(
      [r[0] for r in mup_data],
      [r[1] for r in mup_data],
      'rx', markersize=8 , linewidth=2)
  pyplot.axis('equal')
  pyplot.xlabel('X-Coordinate (m)')
  pyplot.ylabel('Y-Coordinate (m)')
  pyplot.show()
