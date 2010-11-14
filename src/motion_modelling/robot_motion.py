'''
Two-wheeled robot motion model.

Created on 2010-11-13

@author: Michael Kwan
'''
import math
import numpy as np
from matplotlib import pyplot


if __name__ == '__main__':
  dt = 1  # Timestep
  x0 = np.mat([[0], [0], [0.1]])  # Initial State
  v = 1  # Speed
  w = 0.1  # Heading rate of change

  # Noise Model (speed and heading)
  R = np.mat([[0.001, 0],
              [0, 0.05]])
  RE, Re = np.linalg.eig(R)[0]

  n = 1000  # Samples
  x = np.mat(np.zeros((3, n)))
  # Generate Disturbances
  E = RE * np.sqrt(Re) * np.random.randn(1, n)
  for i in xrange(0, n):
    # Dynamics
    x[:, i] = (x0
               + np.mat([[dt * v * math.cos(x0[2])],
                         [dt * v * math.sin(x0[2])],
                         [E[0, i] + dt * w]])
               + np.dot(np.diag([0.05, 0.05, 0.1]), np.random.randn(3, 1))
               )
  # Disturbance free dynamics
  x1 = x0 + np.mat([[dt * v * math.cos(x0[2])],
                    [dt * v * math.sin(x0[2])],
                    [dt * w]])

  # Plot
  pyplot.figure(1)
  pyplot.clf()
  pyplot.hold(True)
  pyplot.plot(x0[0], x0[1], 'bo', markersize=20, linewidth=3)
  pyplot.plot(x1[0], x1[1], 'bo', markersize=20, linewidth=3)
  pyplot.plot([x0[0, 0], x1[0, 0]], [x0[1, 0], x1[1, 0]], 'b')
  pyplot.plot(x[0, :], x[1, :], 'm.', markersize=3)
  pyplot.title('Motion Model Distribution for two-wheeled robot')
  pyplot.xlabel('x (m)')
  pyplot.ylabel('y (m)')
  pyplot.axis('equal')
  pyplot.show()
