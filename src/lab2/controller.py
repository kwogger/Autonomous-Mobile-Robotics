'''Controllers for ME 597 Lab 1

Created on 2010-10-15

@author: Michael Kwan
'''
import math
import numpy as np

TICK_PER_ENC = 100 * 4
ENC_PER_DIFF = 27
DIFF_PER_WHEEL = 2.5
WHEEL_DIAMETER = 0.099

METER_PER_TICK = WHEEL_DIAMETER * math.pi / (TICK_PER_ENC * ENC_PER_DIFF
                                             * DIFF_PER_WHEEL)


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


def encoder_to_velocity(cur_ticks, prev_ticks, dt):
  '''Convert the encoder ticks to velocity.

  Args:
    cur_ticks: The current number of ticks
    prev_ticks: The previous number of ticks
    dt: The difference in time

  Returns:
    The velocity of the robot
  '''
  return (cur_ticks - prev_ticks) / dt * METER_PER_TICK


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
      'angle': delta, #angle_limiter(delta),
      'waypt': shortest[1],
      }


# EKF Constants
EKF_CONSTS = { 
#    'R': np.array([
#        [0.14162522524774, 0.0135871726258, -0.01885776500978],
#        [0.0135871726258, 0.00130502292838, -0.0018274115723],
#        [-0.01885776500978, -0.0018274115723, 0.0058360123874],
#        ]),
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
    }   

EKF_CONSTS_GPS = {
    'Q': np.array([
        [4.22603233, 8.1302549, -0.05544],
        [8.13025, 16.192, -0.10088],
        [-0.05544, -0.10088, 0.003102],
        ]),
    'H': lambda mu_p,x: np.array([
        [1/80913.694760278, 0, 0],
        [0, 1/111101.911587005, 0],
        [0,0,1],
        ]),
    
    'h': lambda lat_orig, long_orig: lambda mu_p,x: np.array([
         mu_p[1]/80913.694760278 + long_orig,
         mu_p[0]/111101.911587005 + lat_orig,
         mu_p[2]+math.pi/2,
         ])
    }

EKF_CONSTS_GPS.update(EKF_CONSTS)
EKF_CONSTS_ENC = {
    'Q': 0.0000380112014723476,
    'H': lambda mu_p,x: np.array([[
        x[0] + (mu_p[0]-x[0]) / np.sqrt((mu_p[0]-x[0])**2 + (mu_p[1]-x[1])**2),
        x[1] + (mu_p[1]-x[1]) / np.sqrt((mu_p[0]-x[0])**2 + (mu_p[1]-x[1])**2),
        0,
        ]]),
    'h': lambda mu_p,x: np.sqrt((mu_p[0]-x[0])**2 + (mu_p[1]-x[1])**2)
    }
EKF_CONSTS_ENC.update(EKF_CONSTS)



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
  VELOCITY_STICTION_OFFSET = 18
  DRIVING_DISTANCE = 10.0
  ROBOT_LENGTH = 0.238

  print 'calculating ekf'
  print 'x: %s' % str(x)
  print 'y: %s' % str(y)
  print 'S: %s' % str(S)
  print 'Q: %s' % str(Q)
  print 'u: %s' % str(u)
  print 'R: %s' % str(R)
  dt = t-prev_t
  print 'dt: %f' % dt
  
  G = G(x, u, dt)
  print 'G: %s' % str(G)
  mu_p = mu_p(x, u, dt)
  print 'mu_p: %s' % str(mu_p)
  H = H(mu_p,x)
  print 'H: %s' % str(H)
  Sp = np.dot(np.dot(G, S), G.T) + R
  print 'Sp: %s' % str(Sp)
  #Calculate Kalman gain
  if np.isscalar(Q):
    K = np.dot(Sp, np.dot(H.T, 1 / (np.dot(H, np.dot(Sp, H.T)) + Q)))
  else:
    K = np.dot(Sp, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(Sp, H.T)) + Q)))
  print 'K: %s' % str(K)
  print 'h: %s' % str(h(mu_p,x))
  return {
      'K': K,
      'mu': (mu_p + np.dot(K, (y - h(mu_p,x))).T),
      'S': np.dot(np.eye(len(Sp)) - np.dot(K, H), Sp),
      'prev_t': t,
      }

def frange(start, end=None, inc=None):
  "A range function, that does accept float increments..."

  if end == None:
    end = start + 0.0
    start = 0.0

  if inc == None:
    inc = 1.0

  L = []
  while 1:
    next = start + len(L) * inc
    if inc > 0 and next >= end:
      break
    elif inc < 0 and next <= end:
      break
    L.append(next)
        
  return L


def inversescanner(grid, x, y, theta, phi_min, phi_inc, meas_r, rmax, alpha, beta):
  '''Calculates inverse measurement model for laser scanner.
     Identifies 3 regions of 0.4 (object unlikely), 0.5 (no new info)
   and 0.6 (object likely).
   Args (all in grid coordinates):
   grid: occupancy grid of size M by N.
   x,y,theta: current robot states (in grid coordinates).
   meas_phi,meas_r: list of last scan
   rmax: max range after which we determine object to not exist
   alpha: distance tolerance around measurement to exclude
   beta: angle tolerance around measurement to exlude (in rads)
   '''
  # taking constants out so we don't have to calculate them over and over
  logit_5 = math.log(1) #log(1) is zero, does it really need to be calculated?
  logit_6 = math.log(0.6 / 0.4)
  logit_4 = math.log(0.4 / 0.6)
  
  #Voodoo to make us not be schmucks, so we ignore grid cells we can't see.
  j_start = max(0, x - rmax)
  j_stop = min(50, x + rmax)
  #j_stop = min(xrange(grid), x+rmax)
  i_start = max(0, y - rmax)
  i_stop = min(50, y + rmax)
  #i_stop = min(xrange(grid[0]),y+rmax)
  meas_phi = frange(phi_min, phi_min + phi_inc * len(meas_r), phi_inc)

  for j in xrange(j_start,j_stop):
    for i in xrange(i_start,i_stop):
  #for j in xrange(0, 50):
    #for i in xrange(0, 50):
      # for each array cell
      # find range and bearing to the current cell
      r = math.sqrt((i - y) * (i - y) + (j - x) * (j - x))
      # phi: angle difference between point and heading
      # modded to [-pi, pi] range
      phi = (math.atan2(j - x, i - y) - theta + math.pi) % (2 * math.pi) - math.pi

      # find index of closest measurement to cell
      ri = rmax
      min_delta_phi = 361
      for k in xrange(len(meas_phi)):
        if abs(phi - meas_phi[k]) < min_delta_phi:
          min_delta_phi = abs(phi - meas_phi[k])
          ri = meas_r[k]

      print 'ri = %s' % str(ri)

      # calculate probability
      # check range
      if ri < (r - alpha) or min_delta_phi > beta:
        grid[j][i] += logit_5
        print i,',',j,',',min_delta_phi,',',phi,',',ri,',',r
      # if range measurement in cell, mark likely
      elif ri < rmax and abs(r - ri) < alpha:
        grid[j][i] += logit_6
      # otherwise empty
      else:
        grid[j][i] += logit_4
  return grid

