'''
Created on 2010-11-22

@author: Michael Kwan
'''

from amr import draw
import csv
import math
from matplotlib import pyplot

if __name__ == '__main__':
  ekf_est_file = open('ekf_est.csv', 'r')
  gps_file = open('gps.csv', 'r')
  ekf_est_reader = csv.reader(ekf_est_file)
  gps_reader = csv.reader(gps_file)
  ekf_est_data = []
  gps_data = []

  for x, y, theta in ekf_est_reader:
    ekf_est_data.append((float(x), float(y), float(theta)))

  for t, lat, long, alt, track, err_track, speed, vel_cmd, turn_cmd in gps_reader:
    gps_data.append((float(t) / 1e9, float(lat), float(long), float(track) / 180 * math.pi, float(turn_cmd)))

  pyplot.figure(1)
  print 'PLOT EKF EST'
  print 'Plotting path'
  pyplot.plot([r[0] for r in ekf_est_data], [r[1] for r in ekf_est_data], 'g-')
  print 'Plotting cars'
  for x, y, theta in ekf_est_data[0:-1:10]:
    draw.drawcar(x, y, theta, .5 * 0.257, 1)
  
  pyplot.figure(2)
  print 'PLOT GPS'
  print 'Plotting path'
  pyplot.plot([r[1] for r in gps_data], [r[2] for r in gps_data], 'g-')
  print 'Plotting cars'
  for r in gps_data[0:-1:5]:
    draw.drawcar(r[1], r[2], r[4], .5 * 0.257 / 1e5, 2)
  pyplot.show()
