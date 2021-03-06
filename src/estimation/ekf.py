'''
Extended Kalman filter example.

Created on 2010-11-14

@author: Michael Kwan
'''
import math
import numpy as np
from matplotlib import pyplot
from amr import draw, controller


if __name__ == '__main__':
  # Discrete time step
  dt = 0.1

  # Initial State
  x0 = np.mat([20, -2, 3]).T

  # Prior
  mu = np.mat([22, -1.8, 3.5]).T  # mean (mu)
  S = np.mat(1 * np.eye(3))  # covariance (Sigma)

  # Discrete motion model
  Ad = np.mat([[1, dt, 0],
               [0, 1, 0],
               [0, 0, 1]])

  R = np.mat([[.0001, 0, 0],
              [0, .0001, 0],
              [0, 0, .0001]])
  Re, RE = np.linalg.eig(R)
  Re = Re * np.eye(len(Re))

  mu_p = lambda mu, u, dt: np.dot(Ad, mu)
  u = None
  G = lambda mu, u, dt: Ad

  # Measurement model defined below
  Q = np.mat([.0001])

  H = lambda mu_p, mu: np.mat([
      (mu_p[0, 0]) / (math.sqrt(mu_p[0, 0] ** 2 + mu_p[2, 0] ** 2)),
      0,
      (mu_p[2, 0]) / (math.sqrt(mu_p[0, 0] ** 2 + mu_p[2, 0] ** 2))])
  h = lambda mu_p, mu: math.sqrt(mu_p[0, 0] ** 2 + mu_p[2, 0] ** 2)

  # Simulation Initializations
  Tf = 10
  T = np.arange(0, Tf, dt)
  n = len(Ad)
  x = np.mat(np.zeros((n, len(T))))
  x[:, 0] = x0
  m = len(Q)
  y = np.mat(np.zeros((m, len(T))))
  mup_S = np.mat(np.zeros((n, len(T))))
  mu_S = np.mat(np.zeros((n, len(T))))
  K_S = np.mat(np.zeros((n, len(T))))


  ## Main loop
  for t in xrange(1, len(T)):
    ## Simulation
    # Select a motion disturbance
    e = np.dot(np.dot(RE, np.sqrt(Re)), np.random.randn(n, 1))
    # Update state
    x[:, t] = np.dot(Ad, x[:, t - 1]) + e

    # Take measurement
    # Select a motion disturbance
    d = np.dot(np.sqrt(Q), np.random.randn(m, 1))
    # Determine measurement
    y[:, t] = math.sqrt(x[0, t] ** 2 + x[2, t] ** 2) + d


    ## Extended Kalman Filter Estimation
    # Measurement update
    ekf_data = controller.ekf(mu, y[:, t], S, Q, u, R, G, mu_p, H, h, t, t - 1)
    K = ekf_data['K']
    mu = ekf_data['mu']
    mup = ekf_data['mu_p']
    S = ekf_data['S']

    # Store results
    mup_S[:, t] = mup
    mu_S[:, t] = mu
    K_S[:, t] = K


    ## Plot results
    pyplot.figure(1)
    pyplot.clf()
    pyplot.hold(True)
    pyplot.plot(0, 0, 'bx', markersize=6, linewidth=2)
    pyplot.plot([20, -1], [0, 0], 'b--')
    pyplot.plot(x[0, 1:t], x[2, 1:t], 'ro--')
    pyplot.plot(mu_S[0, 1:t], mu_S[2, 1:t], 'bx--')
    mu_pos = [mu[0], mu[2]]
    S_pos = np.mat([[S[0, 0], S[0, 2]],
                    [S[2, 0], S[2, 2]]])
    draw.error_ellipse(S_pos, mu_pos, 0.75)
    draw.error_ellipse(S_pos, mu_pos, 0.95)
    pyplot.title('True state and belief')
    pyplot.axis([-1, 20, -1, 10])
    pyplot.waitforbuttonpress(1e-9)
