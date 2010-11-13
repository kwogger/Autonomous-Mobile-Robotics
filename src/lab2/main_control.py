#!/usr/bin/env python
'''
Main control for Lab 2 of ME 597.

Created on 2010-10-13

@author: Michael Kwan
'''
import controller
import csv
import math
import numpy as np
import roslib; roslib.load_manifest('grp6lab2')
import rospy
from clearpath_horizon.msg import RawEncoders
from collections import deque
from geometry_msgs.msg import Twist
from gps_common.msg import GPSFix
from gps_common.msg import GPSStatus
from indoor_pos.msg import ips_msg
#from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


# Callback Constants
ENCODER, IPS, GPS_FIX, GPS_STATUS, LIDAR = range(5)
msg_buffer = deque(maxlen=10)

# Robot Driving Constants
VELOCITY_STICTION_OFFSET = 18
DRIVING_DISTANCE = 2.0
ROBOT_LENGTH = 0.238

# Debugging Constants
CSV_FILES = ['gps', 'ekf']
CSV_FOLDER = '/home/administrator/'

# PID Constants
REF_SPEED = 0.4
K_P = 45
K_I = 50
K_D = 3

# EKF Constants
ekf_state = {
    'K': None,
    'mu': None,
    'S': None,
    }
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
        x[2] + ((u[0] * math.tan(u[1]) * dt) / ROBOT_LENGTH)
        ]),
    }
EKF_CONSTS_GPS = {
    'Q': np.array([
        [4.22603233, 8.1302549, -0.05544],
        [8.13025, 16.192, -0.10088],
        [-0.05544, -0.10088, 0.003102]
        ]),
    'H': lambda mu_p: np.array([[
        [1 / (6365 * (mu_p[0] ^ 2 / 6365 ^ 2)), 0, 0],
        [0, 1 / (6365 * (mu_p[1] ^ 2 / 6365 ^ 2)), 0],
        [0, 0, 1],
        ]]),
    }
h_GPS = lambda lat_orig, long_orig: lambda mu_p: np.array([
    np.arcsin(mu_p[0] / 6365) + lat_orig,
    np.arcsin(mu_p[1] / 6365) + long_orig,
    mu_p[3],
    ])
EKF_CONSTS_GPS.update(EKF_CONSTS)
EKF_CONSTS_ENC = {
    'Q': 0.0000380112014723476,
    'H': lambda mu_p: np.array([[
        1/np.cos(mu_p[2]),
        1/np.sin(mu_p[2]),
        mu_p[0]*np.tan(mu_p[2])/np.cos(mu_p[2]),
        ]]),
    'h': lambda mu_p: mu_p[0]/np.cos(mu_p[2])
    }
EKF_CONSTS_ENC.update(EKF_CONSTS)


def encoder_pid_processing(cur_time, cur_ticks, pid_data):
  update = {}
  if pid_data['prev_time'] is None:
    pass  # Just copy the ticks and time if there's no data at all
  elif pid_data['prev_velocity'] is None:
    # Just calculate the velocity
    dt = (cur_time - pid_data['prev_time']).to_sec()
    update['prev_velocity'] = controller.encoder_to_velocity(
        cur_ticks,
        pid_data['prev_ticks'],
        dt,
        )
  else:
    # Figure out the current velocity
    dt = (cur_time - pid_data['prev_time']).to_sec()
    cur = controller.encoder_to_velocity(cur_ticks, pid_data['prev_ticks'], dt)
    # Put everything into the PID controller
    pid_output = controller.pid(REF_SPEED, cur, pid_data['prev_velocity'],
                                pid_data['error_integral'],
                                pid_data['prev_error'], dt, K_P, K_I, K_D)
    # Record everything for the next iteration
    update['prev_error'] = pid_output['e']
    update['error_integral'] = pid_output['int']
    update['linear_velocity_cmd'] = pid_output['out']
    update['prev_velocity'] = cur
  update['prev_ticks'] = cur_ticks
  update['prev_time'] = cur_time
  return update


def msg_buffer_callback(msg, callback_type):
  global msg_buffer
  msg_buffer.append((callback_type, msg))


if __name__ == '__main__':
  rospy.init_node('main_control')

#  lidar_sub = rospy.Subscriber(
#      'scan', LaserScan, queue_size=1000,
#      callback=msg_buffer_callback, callback_args=LIDAR)
  gps_sub = rospy.Subscriber(
      'fix', GPSFix, queue_size=1000,
      callback=msg_buffer_callback, callback_args=GPS_FIX)
  gpsm_sub = rospy.Subscriber(
      'gps_message', GPSStatus, queue_size=1000,
      callback=msg_buffer_callback, callback_args=GPS_STATUS)
  enc_sub = rospy.Subscriber(
      '/clearpath/robots/default/data/raw_encoders', RawEncoders, queue_size=1000,
      callback=msg_buffer_callback, callback_args=ENCODER)
#  ips_sub = rospy.Subscriber(
#      'indoor_pos', ips_msg, queue_size=1000,
#      callback=msg_buffer_callback, callback_args=IPS)

  # Variables for when the robot is running
  vel_cmd = Twist()
  vel_pub = rospy.Publisher('/clearpath/robots/default/cmd_vel', Twist)

  # Steering setup
  steering_angle = {
      'angle': 0,
      'waypt': 0,
      }

  # EKF Data
  ekf_data = {
      'K': 0,
      'mu': np.array([0, 0, math.pi]), # Initial state
      'S': np.eye(3),
      'prev_t': None,
      }

  # PID Data
  pid_data = {
      'error_integral': 0,
      'prev_time': None,
      'prev_velocity': None,
      'prev_ticks': None,
      'prev_error': 0,
      'linear_velocity_cmd': 0,
      }
  initial_ticks = None

  # LIDAR Data
  grid = np.zeros(50)
  grid_1 = np.zeros(50)

  # Setup csv files for logging
  #TODO(mkwan): close files at the end of the script
  csv_writers = {}
  csv_files = {}
  for csv_file in CSV_FILES:
    csv_files[csv_file] = open(CSV_FOLDER + csv_file + '.csv', 'w')
    csv_writers[csv_file] = csv.writer(csv_files[csv_file])


  # Retrieve initial GPS reading
  lat_orig = None
  long_orig = None
  print 'Finding initial GPS fix...'
  while not rospy.is_shutdown() and lat_orig is None and long_orig is None:
    if not msg_buffer:
      rospy.sleep(1e-3)
      continue
    callback_type, msg = msg_buffer.popleft()
    if (callback_type == GPS_FIX
        and not math.isnan(msg.latitude)
        and not math.isnan(msg.longitude)):
      lat_orig = msg.latitude
      long_orig = msg.longitude
      ekf_data['prev_t'] = msg.header.stamp
  print 'Found GPS fix'

  # MAIN LOOP
  while not rospy.is_shutdown():
    # If the message buffer is empty sleep and then re-loop
    if not msg_buffer:
      rospy.sleep(1e-4)
      continue

    # Dequeue and parse a message from the message buffer
    callback_type, msg = msg_buffer.popleft()

    if callback_type == ENCODER:
      # Encoder Message Processing
      # Get the current time, and the number of ticks recorded
      cur_time = msg.header.stamp
      cur_ticks = msg.ticks[0]
      rospy.loginfo('ENCODER:I got: [%d] as encoder ticks at [%s]',
                    cur_ticks, cur_time)

      # EKF Update
      if not pid_data['prev_ticks'] is None:
        ekf_data = controller.ekf(
            x=ekf_data['mu'],
            S=ekf_data['S'],
            prev_t=ekf_data['prev_t'],
            y=(cur_ticks-pid_data['prev_ticks']) * controller.METER_PER_TICK,
            t=cur_time,
            u=np.array([
                # steering_angle['linear_velocity_cmd'],
                REF_SPEED,
                steering_angle['angle']]),
            **EKF_CONSTS_ENC) # Q, H, h, R, G, mu_p
        print ekf_data
        csv_writers['ekf'].writerow([cur_time, ekf_data['mu'], ekf_data['S'], 'ENC'])

      # Update the velocity with a PID controller
      pid_data.update(encoder_pid_processing(cur_time, cur_ticks, pid_data))

      # Limit the movement to a specific distance
      if initial_ticks is None:
        initial_ticks = cur_ticks
      if cur_ticks - initial_ticks > (DRIVING_DISTANCE
                                      / controller.METER_PER_TICK):
        pid_data['linear_velocity_cmd'] = 0
        rospy.signal_shutdown('Distance limit reached')

    elif callback_type == IPS:
      # IPS Message Processing
      cur_time = msg.header.stamp
      rospy.loginfo('IPS:I got:  X:%f, Y:%f, Yaw:%f,',
                    msg.X, msg.Y, msg.Yaw)

    elif callback_type == GPS_FIX:
      # GPS Fix Message Processing
      rospy.loginfo("GPS:I got: [%f]N,[%f]W as location",
                    msg.latitude, msg.longitude)
      # use msg.header.stamp instead of msg.time
      csv_writers['gps'].writerow([msg.time, msg.latitude, msg.longitude,
                                   msg.altitude, msg.track, msg.err_track,
                                   msg.speed])

    elif callback_type == GPS_STATUS:
      # GPS Status Message Processing
      rospy.loginfo("GPS:I got: [%c] as number of visible satellites",
                    msg.satellites_visible)

    elif callback_type == LIDAR:
      # LIDAR Message Processing
      rospy.loginfo("LIDAR:I got: [%f] as range_min",
                    msg.range_min)

    # Steering controller
#    if waypoints is None:
#      waypoints = [((msg.latitude-lat_orig)*111101.911, (msg.longitude-long_orig)*80913.6947), (0, 0), (-10, 10)]
#    if not prev_velocity is None:
#      steering_angle = controller.stanley_steering(waypoints, (((msg.latitude-lat_orig)*111101.911, (msg.longitude-long_orig)*80913.6947), msg.Yaw, prev_velocity, 0.01))

    # Set the motor commands with stiction and steering angle correction
    vel_cmd.linear.x = max(
        - 100,
        min(100,
            (pid_data['linear_velocity_cmd'] + VELOCITY_STICTION_OFFSET) if
            pid_data['linear_velocity_cmd'] > 0 else
            (pid_data['linear_velocity_cmd'] - VELOCITY_STICTION_OFFSET) if
            (pid_data['linear_velocity_cmd'] < 0) else 0))
    if steering_angle:
      vel_cmd.angular.z = max(-100, min(100, steering_angle['angle'] * 400 - 3))
    else:
      vel_cmd.angular.z = -3

    # Publish velocity command
    vel_pub.publish(vel_cmd)

  for csv_file in csv_files.itervalues():
    csv_file.close()

  rospy.signal_shutdown('End of Main Control')
