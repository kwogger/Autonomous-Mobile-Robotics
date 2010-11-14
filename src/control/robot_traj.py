'''
Created on Nov 14, 2010

@author: Michael Kwan
'''
import math
import numpy as np
from matplotlib import pyplot


def drawcar(x, y, h=0, scale=1, fig=0):
  '''Plots a car at position x,y heading h and size scale on figure number fig.

  The default x,y = (0,0), h = 0 points to the right, and scale=1 plots a car
  with a body of radius 2 units.

  Args:
    x: The x-coordinate to draw the car
    y: The y-coordinate to draw the car
    h: The heading of the car
    scale: The scale to draw the car
    fig: The number of the figure to draw the car in.
  '''
  # Make a circle for the body
  t = np.mat(np.arange(0, 2 * math.pi, 0.01))
  length = len(t.T)
  bx = np.sin(t)
  by = np.cos(t)

  # Wheel locations on body
  wh1 = round(length / 4) - 1
  wh2 = round(3 * length / 4) - 1

  # Draw the wheels
  wwidth = 0.2
  wheight = 0.4
  w = np.mat([[0, -wheight],
              [wwidth, -wheight],
              [ wwidth, wheight],
              [ 0, wheight],
              [ 0, 0]])

  # Body top
  top = round(length / 2)
  # Top pointer
  pwidth = 0.1
  pheight = 0.2
  tp = np.mat([[pwidth / 2, 0],
               [ 0, -pheight],
               [ -pwidth / 2, 0],
               [ pwidth / 2, 0]])

  # Car outline
  car = np.bmat([[bx[0, :wh1].T, by[0, :wh1].T],
                [bx[0, wh1] + w[:, 0], by[0, wh1] + w[:, 1]],
                [bx[0, wh1:wh2].T, by[0, wh1:wh2].T],
                [bx[0, wh2] - w[:, 0], by[0, wh2] - w[:, 1]],
                [bx[0, wh2:].T, by[0, wh2:].T]])

  point = np.bmat([bx[0, top] + tp[:, 0], by[0, top] + tp[:, 1]])

  # Size scaling
  car = scale * car
  point = scale * point

  # Rotation matrix
  R = np.mat([[np.cos(h + math.pi / 2), -np.sin(h + math.pi / 2)],
              [np.sin(h + math.pi / 2), np.cos(h + math.pi / 2)]])
  car = (R * car.T).T
  point = (R * point.T).T

  # Centre
  car[:, 0] = car[:, 0] + x
  car[:, 1] = car[:, 1] + y
  point[:, 0] = point[:, 0] + x
  point[:, 1] = point[:, 1] + y

  # Plot
  pyplot.figure(fig)
  pyplot.plot(car[:, 0], car[:, 1], 'b')
  pyplot.plot(car[:, 0], car[:, 1], 'b')
  pyplot.plot(point[:, 0], point[:, 1], 'r')
  pyplot.axis('equal')


if __name__ == '__main__':
  # Robot trajectories

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
    drawcar(xd[0, t], xd[1, t], xd[2, t], .05, 2)
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
    drawcar(xd[0, t], xd[1, t], xd[2, t], .2, 3)
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
#      drawcar(xd[0, t], xd[1, t], xd[2, t], .3, 5)
    pyplot.title('Desired Trajectory')
    pyplot.axis('equal')

  pyplot.show()
