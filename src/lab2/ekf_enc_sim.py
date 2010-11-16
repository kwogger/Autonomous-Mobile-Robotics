'''
Extended Kalman filter example.

Created on 2010-11-14

@author: Michael Kwan
'''
import math
import numpy as np
from matplotlib import pyplot
from amr import draw
from lab2 import controller


EKF_CONSTS = {
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
    'Q': 0.0000380112014723476,
    'H': lambda mu_p, x: np.array([[
        x[0] + (mu_p[0] - x[0]) / np.sqrt((mu_p[0] - x[0]) ** 2 + (mu_p[1] - x[1]) ** 2),
        x[1] + (mu_p[1] - x[1]) / np.sqrt((mu_p[0] - x[0]) ** 2 + (mu_p[1] - x[1]) ** 2),
        0,
        ]]),
    'h': lambda mu_p, x: np.sqrt((mu_p[0] - x[0]) ** 2 + (mu_p[1] - x[1]) ** 2)
    }



if __name__ == '__main__':
  # Discrete time step
  dt = 0.1

  # Initial State
  x0 = np.array([0, 0, math.pi / 4])

  # Prior
  mu = np.array([0, 0, math.pi / 4])  # mean (mu)
  S = 1 * np.eye(3)  # covariance (Sigma)
  u = np.array([0.4, 0])

  R = EKF_CONSTS['R']
  RE, Re, _ = np.linalg.eig(R)[0]

  # Measurement model defined below
  Q = EKF_CONSTS['Q']

  # Simulation Initializations
  Tf = 10
  T = np.arange(0, Tf, dt)
  n = 3
  x = np.zeros((n, len(T)))
  x[:, 0] = x0
  m = 1
  y = np.zeros((m, len(T)))
  mup_S = np.zeros((n, len(T)))
  mu_S = np.zeros((n, len(T)))
  K_S = np.zeros((n, len(T)))


  ## Main loop
  for t in xrange(1, len(T)):
    print x[:, t-1]
    print mu
    ## Simulation
    # Select a motion disturbance
    e = RE * math.sqrt(Re) * np.random.randn(n, 1)
    # Update state
    x[:, t] = EKF_CONSTS['mu_p'](x[:, t - 1], u, dt) + e.T

    # Take measurement
    # Select a motion disturbance
    d = np.sqrt(Q) * np.random.randn(m, 1)
    # Determine measurement
    y[:, t] = EKF_CONSTS['h'](mu, x[:, t - 1]) + d

    ## Extended Kalman Filter Estimation
    # Prediction update
    ekf_data = controller.ekf(
        x[:, t],
        y[:, t],
        S,
        Q,
        u,
        R,
        EKF_CONSTS['G'],
        EKF_CONSTS['mu_p'],
        EKF_CONSTS['H'],
        EKF_CONSTS['h'],
        t * dt,
        (t - 1) * dt)

    # Store results
    K = ekf_data['K']
    mu = ekf_data['mu']
    S = ekf_data['S']
#    mup_S[:, t] = mup
    mu_S[:, t] = mu.T
    K_S[:, t] = K.T


    ## Plot results
    pyplot.figure(1)
    pyplot.clf()
    pyplot.hold(True)
    pyplot.plot(x[0, 1:t], x[1, 1:t], 'ro--')
    pyplot.plot(mu_S[0, 1:t], mu_S[1, 1:t], 'bx--')
    mu_pos = [mu[0], mu[2]]
    S_pos = np.mat([[S[0, 0], S[0, 2]],
                    [S[2, 0], S[2, 2]]])
    draw.error_ellipse(S_pos, mu_pos, 0.75)
    draw.error_ellipse(S_pos, mu_pos, 0.95)
    pyplot.title('True state and belief')
    pyplot.xlabel('X-Coordinate (m)')
    pyplot.ylabel('Y-Coordinate (m)')
    pyplot.axis('equal')
    if pyplot.waitforbuttonpress(1e-9):
      break
  pyplot.show()
