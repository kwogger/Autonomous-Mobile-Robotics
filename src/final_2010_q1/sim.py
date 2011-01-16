'''
Created on Dec 18, 2010

@author: Michael Kwan
'''
from amr import draw
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

def ekf(mu, y, S, Q, u, R, G, mu_p, H, h, t, prev_t):
  '''General Extended Kalman Filter algorithm.

  Args:
    mu: current state
    y: sensor reading
    S: covariance of the current state
    Q: covariance of the sensor reading
    u: latest control input
    R: covariance of the command
    G: Motion model function (linearized)
    mu_p: Motion model function
    H: Measurement model function (linearized)
    h: Measurement model function
    t: current time
    prev_t: time the last time this function was called

  Returns:
    mu, the best guess of current state (position x,y and heading)
    S, Covariance of this guess
  '''
#  print 'calculating ekf'
#  print 'mu: %s' % str(mu)
#  print 'y: %s' % str(y)
#  print 'S: %s' % str(S)
#  print 'Q: %s' % str(Q)
#  print 'u: %s' % str(u)
#  print 'R: %s' % str(R)
  dt = t - prev_t
#  print 'dt: %f' % dt
  G = G(mu.flatten(), u.flatten(), dt)
#  print 'G: %s' % str(G)
  mu_p = mu_p(mu.flatten(), u.flatten(), dt)
#  print 'mu_p: %s' % str(mu_p)
  H = H(mu_p.flatten(), mu)
#  print 'H: %s' % str(H)
  Sp = np.dot(np.dot(G, S), G.T) + R
#  print 'Sp: %s' % str(Sp)

  if np.size(H) == 0:
    return {
        'K': 0,
        'mu': mu_p,
        'mu_p': mu_p,
        'S': Sp,
        'prev_t': t,
        }
  else:
    # Calculate Kalman gain
    K = np.dot(Sp, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(Sp, H.T)) + Q)))
    print 'K: %s' % str(K)
    print 'h: %s' % str(h(mu_p.flatten(), mu))
    return {
        'K': K,
        'mu': mu_p + np.dot(K, (y - h(mu_p.flatten(), mu))),
        'mu_p': mu_p,
        'S': np.dot(np.eye(len(Sp)) - np.dot(K, H), Sp),
        'prev_t': t,
        }

def pt_to_pt_dist(a, b):
  return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[2]) ** 2)

def find_closest_cone(x, cones, range, minAngle, maxAngle):
  visible = (np.inf, 0, 0, 0)
  for cone in cones:
    dist = pt_to_pt_dist(cone, x)
    bearing = np.arctan2(cone[1] - x[1], cone[0] - x[0]) - x[2]
    if (dist < range
        and bearing < maxAngle
        and bearing > minAngle
        and dist < visible[0]):
      visible = (dist, bearing, cone[0], cone[1])
  if np.isinf(visible[0]):
    return None
  else:
    return visible

def generate_measurement_model(gps, cones):
  h = lambda x, _: np.array([[]])
  H = lambda x, _: np.array([[]])
  Q = np.array([[]])
  if gps:
    if len(cones) == 2:
      h = lambda x, _: np.array([[x[0]],
                                 [x[1]],
                                 [np.sqrt((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2)],
                                 [np.arctan2(cones[0][3] - x[1], (cones[0][2] - x[0])) - x[2]],
                                 [np.sqrt((cones[1][2] - x[0]) ** 2 + (cones[1][3] - x[1]) ** 2)],
                                 [np.arctan2(cones[1][3] - x[1], (cones[1][2] - x[0])) - x[2]], ])
      H = lambda x, _: np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [(x[0] - cones[0][2]) / np.sqrt((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), (x[1] - cones[0][3]) / np.sqrt((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), 0],
                                 [(cones[0][3] - x[1]) / ((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), (cones[0][2] - x[0]) / ((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), -1],
                                 [(x[1] - cones[1][2]) / np.sqrt((cones[1][2] - x[0]) ** 2 + (cones[1][3] - x[1]) ** 2), (x[1] - cones[1][2]) / np.sqrt((cones[1][2] - x[0]) ** 2 + (cones[1][3] - x[1]) ** 2), 0],
                                 [(cones[1][3] - x[1]) / ((cones[1][2] - x[0]) ** 2 + (cones[1][3] - x[1]) ** 2), (cones[1][2] - x[0]) / ((cones[1][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), -1], ])
      Q = np.array([[10, 0, 0, 0, 0, 0],
                    [0, 10, 0, 0, 0, 0],
                    [0, 0, 0.001, 0, 0, 0],
                    [0, 0, 0, (20 * np.pi / 180) ** 2, 0, 0],
                    [0, 0, 0, 0, 0.001, 0],
                    [0, 0, 0, 0, 0, (20 * np.pi / 180) ** 2], ])
    elif len(cones) == 1:
      h = lambda x, _: np.array([[x[0]],
                                 [x[1]],
                                 [np.sqrt((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2)],
                                 [np.arctan2(cones[0][3] - x[1], (cones[0][2] - x[0])) - x[2]], ])
      H = lambda x, _: np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [(x[0] - cones[0][2]) / np.sqrt((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), (x[1] - cones[0][3]) / np.sqrt((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), 0],
                                 [(cones[0][3] - x[1]) / ((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), (cones[0][2] - x[0]) / ((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), -1], ])
      Q = np.array([[10, 0, 0, 0],
                    [0, 10, 0, 0],
                    [0.001, 0, 0, 0],
                    [0, (20 * np.pi / 180) ** 2, 0, 0], ])
    elif len(cones) == 0:
      h = lambda x, _: np.array([[x[0]],
                                 [x[1]], ])
      H = lambda x, _: np.array([[1, 0, 0],
                                 [0, 1, 0], ])
      Q = np.array([[10, 0],
                    [0, 10], ])
  else:
    if len(cones) == 2:
      h = lambda x, _: np.array([[np.sqrt((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2)],
                                 [np.arctan2(cones[0][3] - x[1], (cones[0][2] - x[0])) - x[2]],
                                 [np.sqrt((cones[1][2] - x[0]) ** 2 + (cones[1][3] - x[1]) ** 2)],
                                 [np.arctan2(cones[1][3] - x[1], (cones[1][2] - x[0])) - x[2]], ])
      H = lambda x, _: np.array([[(x[0] - cones[0][2]) / np.sqrt((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), (x[1] - cones[0][3]) / np.sqrt((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), 0],
                                 [(cones[0][3] - x[1]) / ((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), (cones[0][2] - x[0]) / ((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), -1],
                                 [(x[1] - cones[1][2]) / np.sqrt((cones[1][2] - x[0]) ** 2 + (cones[1][3] - x[1]) ** 2), (x[1] - cones[1][2]) / np.sqrt((cones[1][2] - x[0]) ** 2 + (cones[1][3] - x[1]) ** 2), 0],
                                 [(cones[1][3] - x[1]) / ((cones[1][2] - x[0]) ** 2 + (cones[1][3] - x[1]) ** 2), (cones[1][2] - x[0]) / ((cones[1][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), -1], ])
      Q = np.array([[0.001, 0, 0, 0],
                    [0, (20 * np.pi / 180) ** 2, 0, 0],
                    [0, 0, 0.001, 0],
                    [0, 0, 0, (20 * np.pi / 180) ** 2], ])
    elif len(cones) == 1:
      h = lambda x, _: np.array([[np.sqrt((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2)],
                                 [np.arctan2(cones[0][3] - x[1], (cones[0][2] - x[0])) - x[2]], ])
      H = lambda x, _: np.array([[(x[0] - cones[0][2]) / np.sqrt((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), (x[1] - cones[0][3]) / np.sqrt((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), 0],
                                 [(cones[0][3] - x[1]) / ((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), (cones[0][2] - x[0]) / ((cones[0][2] - x[0]) ** 2 + (cones[0][3] - x[1]) ** 2), -1], ])
      Q = np.array([[0.001, 0],
                    [0, (20 * np.pi / 180) ** 2], ])
  return {
      'h': h,
      'H': H,
      'Q': Q,
      }

if __name__ == '__main__':
  # Define cones
  cones = np.array([
      np.concatenate([np.arange(1, 19 + 2, 2), np.arange(1, 19 + 2, 2)]),
      np.concatenate([np.arange(0.5, 9.5 + 1, 1) + 2 * np.ones(10) + 1.5 * np.sin(np.arange(0.5, 9.5 + 1, 1)),
                      np.arange(0.5, 9.5 + 1, 1) - 2 * np.ones(10) + 1.5 * np.sin(np.arange(0.5, 9.5 + 1, 1))])]).T
  d_cone = 0.5

  # Simulation Setup
  dt = 0.5
  Tmax = 5  # End point
  dt = 0.5  # Time step
  T = np.arange(0, Tmax + dt, dt)  # Time vector
  x = np.zeros((3, len(T)))
  mu = np.zeros((3, len(T)))
  S = np.zeros((3, 3, len(T)))
  u = np.array([20 - 3 * np.sin(2 * T),
                20 - 3 * np.cos(2 * T)])
  np.random.seed(1337)

  # Motion Model
  r = 0.25
  l = 0.75
  g = lambda x, u, dt: np.array([[x[0] + (r * u[0] + r * u[1]) / 2 * np.cos(x[2]) * dt],
                                 [x[1] + (r * u[0] + r * u[1]) / 2 * np.sin(x[2]) * dt],
                                 [x[2] + (r * u[0] - r * u[1]) / (2 * l) * dt]])
  G = lambda x, u, dt: np.array([[1, 0, -(r * u[0] + r * u[1]) / 2 * np.sin(x[2]) * dt],
                                 [1, 0, (r * u[0] + r * u[1]) / 2 * np.cos(x[2]) * dt],
                                 [0, 0, 1]])
  R = np.diag([0.1, 0.1, 0.01])
  x[:, 0] = [0, 0, np.pi / 4]
  mu[:, 0] = [0, 0, np.pi / 4]
  S[: , :, 0] = np.diag([1, 1, 0.1])

  for i in xrange(1, len(T)):
    x[:, i] = (g(x[:, i - 1], u[:, i], dt) + np.dot(R, np.random.randn(3, 1))).flat
    left = find_closest_cone(x[:, i], cones, 10, np.pi / 6 - np.pi / 18, np.pi / 6 + np.pi / 18)
    right = find_closest_cone(x[:, i], cones, 10, -np.pi / 6 - np.pi / 18, -np.pi / 6 + np.pi / 18)
    visible_cones = []
    if left:
      visible_cones.append(left)
    if right:
      visible_cones.append(right)
    meas_model = generate_measurement_model(i < 4, visible_cones)
    y = []
    if i < 4:
      gps_reading = x[0:2, i] + np.dot(np.diag([10, 10]), np.random.randn(2, 1)).flat
      y.append([gps_reading[0]])
      y.append([gps_reading[1]])
    for cone in visible_cones:
      y.append([cone[0]])
      if cone[1] > 0:
        y.append([np.pi / 6])
      else:
        y.append([-np.pi / 6])
    ekf_data = ekf(mu[:, i - 1].reshape(3, 1),
                   y,
                   S[:, :, i - 1],
                   meas_model['Q'],
                   np.array([u[:, i]]).T,
                   R,
                   G,
                   g,
                   meas_model['H'],
                   meas_model['h'],
                   T[i],
                   T[i - 1])
    mu[:, i] = ekf_data['mu'].flatten()
    S[:, :, i] = ekf_data['S']

  plt.figure(1)
  # Render cones
  patches = []
  for i in xrange(len(cones)):
    patches.append(Circle((cones[i, 0], cones[i, 1]), d_cone / 2))
  p = PatchCollection(patches, alpha=0.4)
  plt.gca().add_collection(p)
  # Render start position
  plt.plot(x[0, 0], x[1, 0], 'bx', markersize=10, linewidth=2)
  # Render car
  for i, pnt in enumerate(mu.T):
    plt.plot(pnt[0], pnt[1], 'rx', markersize=10, linewidth=2)
    draw.drawcar(pnt[0], pnt[1], pnt[2], 0.25, fig=1, style='k')
    draw.error_ellipse(S[0:2, 0:2, i], (pnt[0], pnt[1]), 0.95, style='g')
  for pnt in x.T:
    draw.drawcar(pnt[0], pnt[1], pnt[2], 0.25, fig=1)
  plt.axis('equal')
  plt.axis([0, 20, -2, 12])
  plt.show()
