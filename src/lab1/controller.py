'''Controllers for ME 597 Lab 1

Created on 2010-10-15

@author: Michael Kwan
'''
import math

TICK_PER_ENC = 100
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
  a = angle #% (math.pi * 2)
  if a > math.pi:
    a -= math.pi * 2
  elif a < -math.pi:
    a += math.pi * 2
  return a


def stanley_steering_lab(waypts, pt, theta, v_x, k=1):
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
  # hack hack, const v to see if disturbances
  # v_x = 0.5
  # speed compensator. Deals with low speeds, and fluctuations in speeds.
  v_x += 0.2
  # if the direction is wrong to start with, don't bother with the dist correction,
  # since it will screw around with the heading
  dir = angle_limiter(-theta) - shortest[0]['theta']
  if dir < -math.pi/2 or dir > math.pi/2:
    k = 0
  # omg sign ugliness!
  delta = -(dir -
            math.atan2(k * shortest[0]['line_dist'], v_x))
  return {
      'angle': angle_limiter(delta),
      'waypt': shortest[1],
      }


def stanley_steering(waypts, pos, theta, v_x, k=1):
  '''A Stanley steering controller.

  Args:
    waypts: The set of waypoints defining its path.
    pos: The current position in a tuple (x, y)
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
    dist = pt_to_line(pos, waypts[i], waypts[i + 1])
    if not shortest or dist['line_seg_dist'] <= shortest[0]['line_seg_dist']:
      shortest = (dist, i)
  
  delta = -(shortest[0]['theta'] - theta +
            math.atan2(k * shortest[0]['line_dist'], v_x))
  return {
      'angle': angle_limiter(delta),
      'waypt': shortest[1],
      }
