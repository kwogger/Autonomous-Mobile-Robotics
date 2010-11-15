'''
Robot trajectories.

Created on Nov 14, 2010

@author: Michael Kwan
'''
import math
import numpy as np
from matplotlib import pyplot
from amr import draw


if __name__ == '__main__':
  # Time
  Tmax = 10
  dt = 0.1
  T = np.arange(0, Tmax, dt)

  # Spiral
  x0 = np.mat([1, 1, 1]).T
  xddot = np.mat(np.zeros((3, len(T))))
  xd = np.mat(np.zeros((3, len(T) + 1)))
  xd[:, 0] = x0
  v = np.exp(-0.2 * T)
  w = np.ones(len(T))
  for t in xrange(len(T)):
    xddot[:, t] = [[v[t] * math.cos(xd[2, t])],
                   [v[t] * math.sin(xd[2, t])],
                   [w[t]]]
    xd[:, t + 1] = xd[:, t] + dt * xddot[:, t]

  pyplot.figure(1)
  pyplot.clf()
  pyplot.hold(True)
  pyplot.plot(T, xd[:, 0:-1].T)

  pyplot.figure(2)
  pyplot.clf()
  pyplot.hold(True)
  pyplot.plot(xd[0, :].T, xd[1, :].T)
  for t in xrange(0, len(T), 3):
    draw.drawcar(xd[0, t], xd[1, t], xd[2, t], .05, 2)
  pyplot.title('Desired Trajectory');
  pyplot.axis('equal')

  # Squiggle
  x0 = np.mat([1, 1, 1]).T
  xddot = np.mat(np.zeros((3, len(T))))
  xd = np.mat(np.zeros((3, len(T) + 1)))
  xd[:, 0] = x0
  for t in xrange(len(T)):
    xddot[:, t] = np.mat([[2 * np.cos(xd[2, t])],
                          [1 * np.sin(xd[2, t])],
                          [(xd[0, t])]])
    xd[:, t + 1] = xd[:, t] + dt * xddot[:, t]

  pyplot.figure(3)
  pyplot.clf()
  pyplot.hold(True)
  pyplot.plot(xd[0, :].T, xd[1, :].T)
  for t in xrange(0, len(T), 3):
    draw.drawcar(xd[0, t], xd[1, t], xd[2, t], .2, 3)
  pyplot.title('Desired Trajectory');
  pyplot.axis('equal')

  # Motions
  x0 = np.mat([1, 1, 1]).T
  xddot = np.mat(np.zeros((3, len(T))))
  xd = np.mat(np.zeros((3, len(T) + 1)))
  xd[:, 0] = x0
  v = 2 * np.mat(np.ones(len(T)))
  w = np.mat(np.zeros(len(T)))
  c = math.floor(len(w.T) / 8)

  pyplot.figure(4)
  pyplot.clf()
  pyplot.figure(5)
  pyplot.clf()
  for i in xrange(0, 10):
    w[0, 2 * c + 1:3 * c] = (-5 + i) / 4.0
    w[0, 3 * c + 1:4 * c] = -(-5 + i) / 4.0

    for t in xrange(len(T)):
      xddot[:, t] = np.mat([[v[0, t] * math.cos(xd[2, t])],
                            [v[0, t] * math.sin(xd[2, t])],
                            [w[0, t]]])
      xd[:, t + 1] = xd[:, t] + dt * xddot[:, t]
  
    pyplot.figure(4)
    pyplot.hold(True)
    pyplot.plot(T, xd[:, :-1].T)
  
    pyplot.figure(5)
    pyplot.hold(True)
    pyplot.plot(xd[0, :].T, xd[1, :].T)
#    for t in xrange(0, len(T), 5):
#      draw.drawcar(xd[0, t], xd[1, t], xd[2, t], .3, 5)
    pyplot.title('Desired Trajectory')
    pyplot.axis('equal')

  pyplot.show()
