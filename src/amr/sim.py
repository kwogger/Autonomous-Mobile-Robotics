'''
Created on Dec 12, 2010

@author: Michael Kwan
'''
import numpy as np


def run_lqr(Ad, Bd, Q, R, t0, tf, dt, x0, ss):
  '''Runs an lqr simulation from t0 to tf with the system defined by Ad, Bd, Q and R.'''
  # Costate setup
  T = np.arange(t0, tf, dt)
  P = np.zeros(3, 3, len(T))
  P_S[:, :, len(T)] = Q
  Pn = P_S[:, :, len(T)]

  # Solve for costate
  for t in xrange(len(T) - 1, 0, -1):
    P = Q + Ad.T * Pn * Ad - Ad.T * Pn * Bd * np.linalg.inv(Bd.T * Pn * Bd + R) * Bd.T * Pn * Ad
    P_S[:, :, t] = P
    Pn = P

  # Setup storage structures
  x = np.zeros(3, len(T))
  x[:, 0] = x0.T
  u = np.zeros(1, len(T) - 1)
  Jx = 0
  Ju = 0
  # Steady state comparison
  if ss:
    Kss = dlqr(Ad, Bd, Q, R)

  # Solve for control gain and simulate
  for t in xrange(0, len(T) - 1):
    K = np.linalg.inv(Bd.T * P_S[:, :, t + 1] * Bd + R) * Bd.T * P_S[:, :, t + 1] * Ad
    u[:, t] = -K * x[:, t]
    if ss:
      u[:, t] = -Kss * x[:, t]
    x[:, t + 1] = Ad * x[:, t] + Bd * u[:, t]

    Jx = Jx + 1 / 2 * x[:, t].T * x[:, t]
    Ju = Ju + 1 / 2 * u[:, t].T * u[:, t]
