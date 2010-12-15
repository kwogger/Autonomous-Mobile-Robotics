'''
Example of error ellipse and Gaussian noise generation

Created on Dec 15, 2010

@author: Michael Kwan
'''
import numpy as np
from matplotlib import pyplot
from amr import draw

if __name__ == '__main__':
  # Define distribution
  mu = np.array([[1], [2]])
  S = np.array([[4, -1], [-1, 1]])

  # Find eigenstuff
  Se, SE = np.linalg.eig(S)

  # Generate samples
  samples = np.dot(np.dot(SE, np.sqrt(Se*np.eye(2))), np.random.randn(2, 10000))

  # Create ellipse plots
  pyplot.figure(1)
  pyplot.clf()
  pyplot.hold(True)
  pyplot.axis('equal')
  pyplot.plot(mu[0] + samples[0, :], mu[1] + samples[1, :], 'r.')
  draw.error_ellipse(S, mu, 0.5)
  draw.error_ellipse(S, mu, 0.99)
  pyplot.show()
