'''
Created on Dec 13, 2010

@author: Michael Kwan
'''
import math
import numpy as np
from amr import controller
from matplotlib import pyplot

if __name__ == '__main__':
  # Laser scanner inverse measurement model plot

  M = 50
  N = 60
  m = 0.5 * np.ones((M, N))  # map

  alpha = 1  # Distance about measurement to fill in
  beta = 0.01  # Angle beyond which to exclude 

  # Robot location
  x = 25
  y = 10
  theta = math.pi / 2
  rmax = 80

  # Measurements
  meas_phi = np.arange(-.4, .4, 0.01)  # heading
  meas_r = 40 * np.ones(meas_phi.shape)  # range

  m = controller.inversescanner(M, N, x, y, theta, meas_phi, meas_r, rmax, alpha, beta)

  pyplot.figure(2)
  pyplot.clf()
  pyplot.hold(True)
  pyplot.gray()
  pyplot.imshow(1 - m, interpolation='nearest')
  pyplot.plot(y, x, 'rx', markersize=8, linewidth=2)
  # Plot the circles at the ray's end
#  for i in xrange(len(meas_r)):
#    pyplot.plot(y + meas_r[i] * np.sin(meas_phi[i] + theta), x + meas_r[i] * np.cos(meas_phi[i] + theta), 'go')

  pyplot.axis((0, N, 0, M))
  pyplot.show()
