'''
1D Kalman filter example

Created on Dec 15, 2010

@author: Michael Kwan
'''
import numpy as np
from matplotlib import pyplot
from scipy import stats

if __name__ == '__main__':
  # Discrete time step
  dt = 0.1

  # Prior
  mu = 10.  # mean (mu)
  S = 1.  # covariance (Sigma)

  # Motion model
  A = 0.8
  B = 3.
  R = 2.

  # Measurement model
  C = 1.
  Q = 4.

  # Simulation Initializations
  Tf = 3.
  T = np.arange(0., Tf, dt)
  x = np.zeros(len(T) + 1)
  x[0] = mu + np.sqrt(S) * np.random.randn(1);
  y = np.zeros(len(T))
  u = np.zeros(len(T))
  mup_S = np.zeros(len(T))
  mu_S = np.zeros(len(T))

  ## Main loop
  for t in xrange(len(T)):
    ## Simulation
    # Select control action
    if t > 1:
      u[t] = u[t - 1]
    if mu > 10:
      u[t] = 0.
    elif mu < 2:
      u[t] = 1.

    # Select a motion disturbance
    e = np.sqrt(R) * np.random.randn(1)
    # Update state
    x[t + 1] = A * x[t] + B * u[t] + e

    # Take measurement
    # Select a motion disturbance
    d = np.sqrt(Q) * np.random.randn(1)
    # Determine measurement
    y[t] = C * x[t + 1] + d


    ## Kalman Filter Estimation
    # Store prior
    mu_old = mu
    S_old = S

    # Prediction update
    mup = A * mu + B * u[t];
    Sp = A * S * A + R
    # Measurement update
    K = Sp * C / (C * Sp * C + Q)
    mu = mup + K * (y[t] - C * mup)
    S = (1 - K * C) * Sp

    # Store estimates
    mup_S[t] = mup
    mu_S[t] = mu

    ## Plot first time step results
    if t == 0:
      L = 5.
      # Prior belief
      pyplot.figure(1)
      pyplot.clf()
      pyplot.hold(True)
      z = np.arange(mu_old - L * np.sqrt(S_old), mu_old + L * np.sqrt(S_old) + 0.01, 0.01)
      pyplot.plot(z, stats.norm.pdf(z, mu_old, S_old), 'b')
      pyplot.title('Prior')
      # Prediction step
      pyplot.figure(2)
      pyplot.clf()
      pyplot.hold(True)
      pyplot.plot(z, stats.norm.pdf(z, mu_old, S_old), 'b');
      z = np.arange(mup - L * np.sqrt(Sp), mup + L * np.sqrt(Sp) + 0.01, 0.01)
      pyplot.plot(z, stats.norm.pdf(z, mup, Sp), 'r');
      pyplot.title('Prior & Prediction')
      pyplot.legend(('Prior', 'Prediction'))
      # Measurement step
      pyplot.figure(3)
      pyplot.clf()
      pyplot.hold(True)
      pyplot.plot(z, stats.norm.pdf(z, mup, Sp), 'r')
      z = np.arange(y[t] - L * np.sqrt(Q), y[t] + L * np.sqrt(Q) + 0.01, 0.01)
      pyplot.plot(z, stats.norm.pdf(z, y[t], Q), 'g')
      z = np.arange(mu - L * np.sqrt(S), mu + L * np.sqrt(S) + 0.01, 0.01)
      pyplot.plot(z, stats.norm.pdf(z, mu, S), 'm')
      pyplot.axis([-10, 20, 0, .35])
      pyplot.title('Prediction, Measurement & Belief')
      pyplot.legend(('Prediction', 'Measurement', 'Belief'))
  # Plot full trajectory results
  pyplot.figure(4)
  pyplot.clf()
  pyplot.hold(True)
  pyplot.plot(T, x[1:], 'b')
  pyplot.plot(T, y, 'rx')
  pyplot.plot(T, mup_S, 'c--')
  pyplot.plot(T, mu_S, 'r--')
  pyplot.plot(T, u, 'g')
  pyplot.plot(T, 2 * np.ones(T.shape), 'm--')
  pyplot.plot(T, 10 * np.ones(T.shape), 'm--')
  pyplot.title('State and estimates')
  pyplot.legend(('State', 'Measurement', 'Prediction', 'Estimate', 'Input'))
  pyplot.show()
  print 'done'
