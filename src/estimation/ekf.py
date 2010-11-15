'''
Extended Kalman filter example.

Created on 2010-11-14

@author: Michael Kwan
'''
import math
import numpy as np
from matplotlib import pyplot
from amr import draw


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
  RE, Re, _ = np.linalg.eig(R)[0]

  # Measurement model defined below
  Q = np.mat([.0001])

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
    e = RE * math.sqrt(Re) * np.random.randn(n, 1)
    # Update state
    x[:, t] = Ad * x[:, t - 1] + e

    # Take measurement
    # Select a motion disturbance
    d = np.sqrt(Q) * np.random.randn(m, 1)
    # Determine measurement
    y[:, t] = math.sqrt(x[0, t] ** 2 + x[2, t] ** 2) + d


    ## Extended Kalman Filter Estimation
    # Prediction update
    mup = Ad * mu
    Sp = Ad * S * Ad.T + R

    # Linearization
    Ht = np.mat([(mup[0, 0]) / (math.sqrt(mup[0, 0] ** 2 + mup[2, 0] ** 2)),
                 0,
                 (mup[2, 0]) / (math.sqrt(mup[0, 0] ** 2 + mup[2, 0] ** 2))])

    # Measurement update
    K = Sp * Ht.T * np.linalg.inv(Ht * Sp * Ht.T + Q)
    mu = mup + K * (y[:, t] - math.sqrt(mup[0] ** 2 + mup[2] ** 2))
    S = (np.eye(n) - K * Ht) * Sp

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
