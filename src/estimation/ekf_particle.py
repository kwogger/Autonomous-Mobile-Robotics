'''
Extended Kalman filter and particle filter example.

Created on Dec 15, 2010

@author: Michael Kwan
'''
import math
import numpy as np
from matplotlib import pyplot
from scipy import stats
from amr import controller, draw

if __name__ == '__main__':
  # Discrete time step
  dt = 0.1

  # Initial State
  x0 = np.mat([15, -2, 3]).T

  # Prior
  mu = np.mat([16, -1.8, 4]).T  # mean (mu)
  S = np.mat(1 * np.eye(3))  # covariance (Sigma)
  mu_u = mu
  S_u = S

  # Discrete motion model
  Ad = np.mat([[1, dt, 0],
               [0, 1, 0],
               [0, 0, 1]])
  R = np.mat([[.001, 0, 0],
              [0, .001, 0],
              [0, 0, .001]])
  Re, RE = np.linalg.eig(R)
  Re = Re * np.eye(len(Re))

  mu_p = lambda mu, u, dt: np.dot(Ad, mu)
  u = None
  G = lambda mu, u, dt: Ad

  # Measurement model defined below
  Q = np.mat([.01])

  H = lambda mu_p, mu: np.mat([
      (mu_p[0, 0]) / (math.sqrt(mu_p[0, 0] ** 2 + mu_p[2, 0] ** 2)),
      0,
      (mu_p[2, 0]) / (math.sqrt(mu_p[0, 0] ** 2 + mu_p[2, 0] ** 2))])
  h = lambda mu_p, mu: math.sqrt(mu_p[0] ** 2 + mu_p[2] ** 2)

  # Simulation Initializations
  Tf = 8
  T = np.arange(0, Tf, dt)
  n = len(Ad)
  x = np.mat(np.zeros((n, len(T))))
  x[:, 0] = x0
  m = len(Q)
  y = np.mat(np.zeros((m, len(T))))
  mup_S = np.mat(np.zeros((n, len(T))))
  mu_S = np.mat(np.zeros((n, len(T))))
  K_S = np.mat(np.zeros((n, len(T))))
  muP_S = np.mat(np.zeros((n, len(T))))
  SP_S = np.zeros((n, n, len(T)))

  # Particle Filter Parameters
  # Number of particles
  I = 5000
  # Prior 
  X = np.random.randn(3, I)
  X[0, :] = X[0, :] + mu[0]
  X[1, :] = X[1, :] + mu[1]
  X[2, :] = X[2, :] + mu[2]
  X0 = X

  pyplot.figure(1)
  pyplot.clf()
  pyplot.hold(True)
  pyplot.plot(x0[0], x0[2], 'ro--', markersize=8)
  pyplot.plot(X0[0, 0:10:], X0[2, 0:10:], 'm.')
  # Ground
  pyplot.plot(0, 0, 'bx', markersize=6, linewidth=2)
  pyplot.plot([20, -1], [0, 0], 'b--')
  pyplot.title('True state, EKF and Particle Filter')
  pyplot.axis([-5, 20, -1, 10])
  pyplot.legend(('True state', 'EKF', 'Particles'))
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
    y[:, t] = np.sqrt(x[0, t] ** 2 + x[2, t] ** 2) + d


    ## Extended Kalman Filter Estimation
    ekf_data = controller.ekf(mu, y[:, t], S, Q, u, R, G, mu_p, H, h, t, t - 1)
    K = ekf_data['K']
    mu = ekf_data['mu']
    mup = ekf_data['mu_p']
    S = ekf_data['S']

    # Store results
    mup_S[:, t] = mup
    mu_S[:, t] = mu
    K_S[:, t] = K

    ## Particle Filter
    # Particle filter estimation
#    w = np.zeros(I)
#    Xp = np.zeros((n, I))
#    Xp[:, :] = mu_p(X, u, dt) + np.dot(np.dot(RE, np.sqrt(Re)), np.random.randn(n, I))
#    for i in xrange(I):
#      w[i] = stats.norm.pdf(y[:, t], np.sqrt(Xp[0, i] ** 2 + Xp[2, i] ** 2), np.sqrt(Q))
#    W = np.cumsum(w)
#    seed = np.dot(W[-1], np.random.rand(I))
#    for j in xrange(I):
#      X[:, j] = Xp[:, (W > seed[j]).nonzero()[0][0]]
    particle_filter_data = controller.particle_filter(
        X, y[:, t], Q, u, I, Re, RE, mu_p, h, dt)
    muParticle = particle_filter_data['mu']
    SParticle = particle_filter_data['S']
    X = particle_filter_data['X']

    muP_S[:, t] = muParticle
    SP_S[:, :, t] = SParticle

    ## Plot results
    pyplot.figure(1)
    pyplot.clf()
    pyplot.hold(True)
    # True state
    pyplot.plot(x[0, 1:t], x[2, 1:t], 'ro--')
    # EKF
    pyplot.plot(mu_S[0, 1:t], mu_S[2, 1:t], 'bx--')
    # Particle%
    pyplot.plot(muP_S[0, 1:t], muP_S[2, 1:t], 'mx--')
    # EKF Ellipses
    mu_pos = [mu[0], mu[2]];
    S_pos = [[S[0, 0], S[0, 2]], [S[2, 0], S[2, 2]]]
#     error_ellipse(S_pos,mu_pos,0.75);
    draw.error_ellipse(S_pos, mu_pos, 0.95);
    # Particle set
    pyplot.plot(X[0, 0::10], X[2, 0::10], 'm.')
    # Ground
    pyplot.plot(0, 0, 'bx', markersize=6, linewidth=2)
    pyplot.plot([20, -1], [0, 0], 'b--')
    pyplot.title('True state, EKF and Particle Filter')
    pyplot.axis([-5, 20, -1, 10])
    pyplot.legend(('True state', 'EKF', 'Particles'))
    pyplot.waitforbuttonpress(1e-9)
  pyplot.figure(2)
  pyplot.clf()
  pyplot.hold(True)
  e = np.sqrt(np.power(x[0, 1:] - mu_S[0, 1:], 2) + np.power(x[2, 1:] - mu_S[2, 1:], 2))
  pyplot.plot(T[1:], np.array(e)[0], 'b', linewidth=1.5)
  ep = np.sqrt(np.power(x[0, 1:] - muP_S[0, 1:], 2) + np.power(x[2, 1:] - muP_S[2, 1:], 2))
  pyplot.plot(T[1:], np.array(ep)[0], 'm', linewidth=1.5)
  pyplot.title('Position Estimation Errors for EKF and Particle Filter')
  pyplot.xlabel('Time (s)')
  pyplot.ylabel('X-Z Position Error (m)')
  pyplot.legend(('EKF', 'Particle'))
  pyplot.show()
