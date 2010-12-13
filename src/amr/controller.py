'''
Controller for ME 597

Created on 2010-11-14

@author: Michael Kwan
'''
import math
import numpy as np


def pid(ref, cur, prev, integral, prev_e, dt, k_p, k_i, k_d):
  '''General PID controller.

  Args:
    ref: The reference signal
    cur: The current measurement
    prev: The previous measurement
    int: The integral of the error
    prev_e: The previous error
    dt: The difference in time since last PID
    k_p: The proportional gain
    k_i: The integral gain
    k_d: The diffential gain

  Returns:
    A dictionary the contains 'out' for the output signal, 'e' for the error
    term, and 'int' as the new integral of the error.
  '''
  e = ref - cur
  new_int = integral + (prev_e + e) / 2 * dt
  return {
      'out': e * k_p + new_int * k_i + (cur - prev) / dt * k_d,
      'e': e,
      'int': new_int,
      }


def pt_to_line(pt, pt1, pt2):
  '''Calculates various properties of a point to a line segment.
  
  Args:
    pt: The point in which to compare to the line segment
    pt1: The first point defining line segment
    pt2: The second point defining the line segment

  Returns:
    A dictionary the contains 'pt' as the closest point on the line segment to
    the point, 'theta' as the angle of the given line segment, 'line_seg_dist'
    as the closest distance from the point to the line segment, amd 'line_dist'
    as the closest distance of the point to the line defined by the line
    segment.
  '''
  r_numerator = ((pt[0] - pt1[0]) * (pt2[0] - pt1[0]) +
                 (pt[1] - pt1[1]) * (pt2[1] - pt1[1]))
  r_denomenator = ((pt2[0] - pt1[0]) * (pt2[0] - pt1[0]) +
                   (pt2[1] - pt1[1]) * (pt2[1] - pt1[1]))
  r = r_numerator / r_denomenator

  px = pt1[0] + r * (pt2[0] - pt1[0])
  py = pt1[1] + r * (pt2[1] - pt1[1])

  s = ((pt1[1] - pt[1]) * (pt2[0] - pt1[0]) -
       (pt1[0] - pt[0]) * (pt2[1] - pt1[1])) / r_denomenator

  line_dist = s * math.sqrt(r_denomenator)
  distanceLine = math.fabs(line_dist)

  # (xx, yy) is the point on the lineSegment closest to (pt[0] ,pt[1])
  xx = px;
  yy = py;

  if r >= 0 and r <= 1:
    distanceSegment = distanceLine
  else:
    dist1 = ((pt[0] - pt1[0]) * (pt[0] - pt1[0]) +
             (pt[1] - pt1[1]) * (pt[1] - pt1[1]))
    dist2 = ((pt[0] - pt2[0]) * (pt[0] - pt2[0]) +
             (pt[1] - pt2[1]) * (pt[1] - pt2[1]))
    if dist1 < dist2:
      xx = pt1[0];
      yy = pt1[1];
      distanceSegment = math.sqrt(dist1);
    else:
      xx = pt2[0];
      yy = pt2[1];
      distanceSegment = math.sqrt(dist2);

  theta = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])

  return {
      'pt': (xx, yy),
      'theta': theta,
      'line_seg_dist': distanceSegment,
      'line_dist': line_dist,
      }


def angle_limiter(angle):
  a = angle % (math.pi * 2)
  if a > math.pi:
    a -= math.pi * 2
  elif a < -math.pi:
    a += math.pi * 2
  return a


def stanley_steering(waypts, pt, theta, v_x, k=1):
  '''A Stanley steering controller.

  Args:
    waypts: The set of waypoints defining its path.
    pt: The current position in a tuple (x, y)
    theta: The current heading in radians
    v_x: The current velocity
    k: The gain parameter

  Returns:
    A dictionary with 'angle' as the resultant angle in which to steer towards
    and 'waypt' as the waypoint that the controller is travelling from.
  '''
  shortest = None
  for i in xrange(len(waypts) - 1):
    # iterate through and find the closest point
    dist = pt_to_line(pt, waypts[i], waypts[i + 1])
    if not shortest or dist['line_seg_dist'] <= shortest[0]['line_seg_dist']:
      shortest = (dist, i)
  delta = -(angle_limiter(-theta + shortest[0]['theta']) +
            math.atan2(k * shortest[0]['line_dist'], v_x))
  return {
      'angle': delta,
      'waypt': shortest[1],
      }


def ekf(x, y, S, Q, u, R, G, mu_p, H, h, t, prev_t):
  '''General Extended Kalman Filter algorithm.

  Args:
    x: current state
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
#  print 'x: %s' % str(x)
#  print 'y: %s' % str(y)
#  print 'S: %s' % str(S)
#  print 'Q: %s' % str(Q)
#  print 'u: %s' % str(u)
#  print 'R: %s' % str(R)
  dt = t - prev_t
#  print 'dt: %f' % dt
  G = G(x, u, dt)
#  print 'G: %s' % str(G)
  mu_p = mu_p(x, u, dt)
#  print 'mu_p: %s' % str(mu_p)
  H = H(mu_p, x)
#  print 'H: %s' % str(H)
  Sp = np.dot(np.dot(G, S), G.T) + R
#  print 'Sp: %s' % str(Sp)
  #Calculate Kalman gain
  K = np.dot(Sp, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(Sp, H.T)) + Q)))
#  print 'K: %s' % str(K)
#  print 'h: %s' % str(h(mu_p, x))
  return {
      'K': K,
      'mu': (mu_p + np.dot(K, (y - h(mu_p, x))).T),
      'S': np.dot(np.eye(len(Sp)) - np.dot(K, H), Sp),
      'prev_t': t,
      }


def inversescanner(M, N, x, y, theta, meas_phi, meas_r, rmax, alpha, beta):
  # Calculates the inverse measurement model for a laser scanner
  # Identifies three regions, the first where no new information is
  # available, the second where objects are likely to exist and the third
  # where objects are unlikely to exist

  # Range finder inverse measurement model
  m = np.zeros((M, N))
  for i in xrange(M):
    for j in xrange(N):
      # Find range and bearing to the current cell
      r = math.sqrt((i - x) ** 2 + (j - y) ** 2)
      phi = math.fmod(math.atan2(j - y, i - x) - theta + math.pi, 2 * math.pi) - math.pi

      # Find the applicable range measurement 
      k = np.argmin(abs(phi - meas_phi))

      # If out of range, or behind range measurement, or outside of field
      # of view, no new information is available
      if (r > min(rmax, meas_r[k] + alpha / 2.0)) or (abs(phi - meas_phi[k]) > beta / 2.0):
        m[i, j] = 0.5

      # If the range measurement was in this cell, likely to be an object
      elif (meas_r[k] < rmax) and (abs(r - meas_r[k]) < alpha / 2.0):
        m[i, j] = 0.6

      # If the cell is in front of the range measurement, likely to be
      # empty
      elif r < meas_r[k]:
        m[i, j] = 0.4
  return m
