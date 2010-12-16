'''
Extended Kalman Filter Localization example

Created on Dec 15, 2010

@author: Michael Kwan
'''
import math
import numpy as np
from amr import misc, draw
from matplotlib import pyplot


if __name__ == '__main__':
  # Time
  Tf = 20
  dt = 0.1
  T = np.arange(0, Tf, dt)

  # Initial State
  x0 = np.array([0, 0, 0])

  # Prior
  mu = np.array([0, 0, 0])  # mean (mu)
  S = 0.001 * np.eye(3)  # covariance (Sigma)

  # Control inputs
  u = np.ones((2, len(T)))
  u[1, :] = 0.3 * u[1, :]

  # Disturbance model
  R = np.array([[1e-4, 0, 0],
                [0, 1e-4, 0],
                [0, 0, 1e-6]])
  Re, RE = np.linalg.eig(R)
  Re = Re * np.eye(len(Re))

  # Measurement type and noise
  meas = 3; # 1 - range, 2 - bearing, 3 - both

  Q = {
      1: np.array([0.01]),
      2: np.array([0.01]),
      3: np.array([[0.001, 0],
                   [0, 0.001]]),
      }.get(meas)
  Qe, QE = np.linalg.eig(Q)
  Qe = Qe * np.eye(len(Qe))

  # Feature Map
  map = np.array([[5, 5],
                  [3, 1],
                  [-4, 5],
                  [-2, 3],
                  [0, 4]])

  # Simulation Initializations
  n = len(x0)
  x = np.zeros((n, len(T)))
  x[:, 0] = x0
  m = len(Q[:, 0])
  y = np.zeros((m, len(T)))
  mup_S = np.zeros((n, len(T)))
  mu_S = np.zeros((n, len(T)))
  mf = np.zeros((2, len(T)))

  ## Main loop
  for t in xrange(1, len(T)):
    ## Simulation
    # Select a motion disturbance
    e = np.dot(np.dot(RE, np.sqrt(Re)), np.random.randn(n, 1))
    # Update state
    x[:, t] = (np.array([[x[0, t - 1] + u[0, t] * math.cos(x[2, t - 1]) * dt],
                         [x[1, t - 1] + u[0, t] * math.sin(x[2, t - 1]) * dt],
                         [x[2, t - 1] + u[1, t] * dt]]) + e).flat

    mup = np.array([[mu[0] + u[0, t] * math.cos(mu[2]) * dt],
                    [mu[1] + u[0, t] * math.sin(mu[2]) * dt],
                    [mu[2] + u[1, t] * dt]]).flatten()


    # Take measurement
    # Pick feature
    mf[:, t] = misc.closest_feature(map, x[:, t])[0]
    # Select a motion disturbance
    d = np.dot(np.dot(QE, np.sqrt(Qe)), np.random.randn(m, 1))
    # Determine measurement
    y[:, t] = {
        1: np.array([np.sqrt(np.power(mf[0, t] - x[0, t], 2) + np.power(mf[1, t] - x[1, t], 2))]) + d,
        2: np.array([np.arctan2(mf[1, t] - x[1, t], mf[0, t] - x[0, t]) - x[2, t]]) + d,
        3: np.array([[np.sqrt(np.power(mf[0, t] - x[0, t], 2) + np.power(mf[1, t] - x[1, t], 2))],
                     [np.arctan2(mf[1, t] - x[1, t], mf[0, t] - x[0, t]) - x[2, t]]]) + d,
    }.get(meas).flat

    ## Extended Kalman Filter Estimation
    # Prediction update
    Gt = np.array([[1, 0, -u[0, t] * np.sin(mu[2]) * dt],
                   [0, 1, u[0, t] * np.cos(mu[2]) * dt],
                   [0, 0, 1]])

    Sp = np.dot(np.dot(Gt, S), Gt.T) + R

    # Linearization
    # Predicted range
    rp = np.sqrt((mf[0, t] - mup[0]) ** 2 + np.square(mf[1, t] - mup[1]) ** 2)
    Ht = {
        1: np.array([-(mf[0, t] - mup[0]) / rp,
                     - (mf[1, t] - mup[1]) / rp,
                     0]),
        2: np.array([(mf[1, t] - mup[1]) / rp ** 2,
                     - (mf[0, t] - mup[0]) / rp ** 2,
                     - 1]),
        3: np.array([[-(mf[0, t] - mup[0]) / rp,
                      - (mf[1, t] - mup[1]) / rp,
                      0],
                     [(mf[1, t] - mup[1]) / rp ** 2,
                      - (mf[0, t] - mup[0]) / rp ** 2,
                      - 1]])
        }.get(meas)

    # Measurement update
    K = np.dot(np.dot(Sp, Ht.T), np.linalg.inv(np.dot(np.dot(Ht, Sp), Ht.T) + Q))
    I = {
        1: y[:, t] - np.sqrt(np.square(mf[0, t] - mup[0], 2) + np.square(mf[1, t] - mup[1], 2)),
        2: y[:, t] - (np.arctan2(mf[1, t] - mup[1], mf[0, t] - mup[0]) - mup[2]),
        3: y[:, t] - np.array([
            [np.sqrt(np.square(mf[0, t] - mup[0], 2) + np.square(mf[1, t] - mup[1], 2))],
            [(np.arctan2(mf[1, t] - mup[1], mf[0, t] - mup[0]) - mup[2])]])
    }.get(meas)
    mu = mup + np.dot(K, I)
    S = np.dot(np.eye(n) - np.dot(K, Ht), Sp)

    # Store results
    mup_S[:, t] = mup
    mu_S[:, t] = mu


    ## Plot results
    pyplot.figure(1)
    pyplot.clf()
    pyplot.hold(True)
    pyplot.plot(map[:, 0], map[:, 1], 'go', markersize=10, linewidth=2)
    pyplot.plot(mf[0, t], mf[1, t], 'mx', markersize=10, linewidth=2)
    pyplot.plot(x[0, :t], x[1, :t], 'ro--')
#    if meas == 1:
#      pyplot.circle(1, x(1:2, t), y(1, t))
    if meas == 2:
      pyplot.plot([x[0, t], x[0, t] + 10 * np.cos(y[0, t] + x[2, t])],
                  [ x[1, t], x[1, t] + 10 * np.sin(y[0, t] + x[2, t])],
                  'c')
    elif meas == 3:
      pyplot.plot([x[0, t], x[0, t] + y[0, t] * np.cos(y[1, t] + x[2, t])],
                  [x[1, t], x[1, t] + y[0, t] * np.sin(y[1, t] + x[2, t])],
                  'c')
    pyplot.plot(mu_S[0, :t], mu_S[1, :t], 'bx--')
    pyplot.plot(mup_S[0, :t - 1], mup_S[1, :t - 1], 'go--')
    pyplot.axis('equal')
    pyplot.axis([-4, 6, -1, 7])
    mu_pos = np.array([mu[0], mu[1]])
    S_pos = np.array([[S[0, 0], S[0, 1]], [S[1, 0], S[1, 1]]])
    draw.error_ellipse(S_pos, mu_pos, 0.95)
    draw.error_ellipse(S_pos, mu_pos, 0.999)
    pyplot.title('Range & Bearing Measurements for Localization')
    pyplot.waitforbuttonpress(1e-9)

