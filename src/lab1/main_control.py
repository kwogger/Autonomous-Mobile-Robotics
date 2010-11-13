#!/usr/bin/env python
'''
Main control for Lab 1 of ME 597.

Created on 2010-10-13

@author: Michael Kwan
'''
import controller
import roslib; roslib.load_manifest('grp6_lab1')
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from clearpath_horizon.msg import RawEncoders
from geometry_msgs.msg import Twist
from indoor_pos.msg import ips_msg
import csv


REF_SPEED = 1.5
K_P = 45
K_I = 0
K_D = 0

error_integral = 0
prev_time = None
prev_velocity = None
prev_ticks = None
prev_error = 0

linear_velocity_cmd = 0

waypoints = None
steering_angle = None


def encoder_callback(msg):
  '''Callback to handle encoder odometry information.

  Uses a PID controller to control the velocity of the robot.
  '''
  # Setup globals
  # TODO(mkwan): Make it not rely on globals
  global REF_SPEED
  global K_P
  global K_I
  global K_D
  global error_integral
  global prev_time
  global prev_velocity
  global prev_ticks
  global prev_error
  global linear_velocity_cmd

  # Get the current time, and the number of ticks recorded
  cur_time = rospy.get_time()
  cur_ticks = msg.ticks[0]

  # Logging the info
  rospy.loginfo('ENCODER:I got: [%d] as encoder ticks at [%f]',
                cur_ticks, cur_time)

  if prev_time is None:
    pass  # Just copy the ticks and time if there's no data at all
  elif prev_velocity is None:
    # Just calculate the velocity
    dt = cur_time - prev_time
    prev_velocity = controller.encoder_to_velocity(cur_ticks, prev_ticks, dt)
  else:
    # Figure out the current velocity
    dt = cur_time - prev_time
    cur = controller.encoder_to_velocity(cur_ticks, prev_ticks, dt)
    # Put everything into the PID controller
    pid_output = controller.pid(REF_SPEED, cur, prev_velocity, error_integral,
                                prev_error, dt, K_P, K_I, K_D)
    # Record everything for the next iteration
    prev_error = pid_output['e']
    error_integral = pid_output['int']
    linear_velocity_cmd = pid_output['out']
    prev_velocity = cur
  prev_ticks = cur_ticks
  prev_time = cur_time


  # Write the encoder tick data to file
  if not prev_velocity is None:
    writer = csv.writer(open("/home/administrator/ROS_packages/grp6_lab1/nodes/data.csv", "a"))
    writer.writerow((
        cur_time,
        cur_ticks,
        linear_velocity_cmd,
        prev_error,
        error_integral,
        prev_velocity))


#def ips_callback(msg):
#  '''Callback to handle IPS location information.'''
#  global waypoints
#  global prev_velocity
#  global steering_angle
#  if waypoints is None:
#    waypoints = [(msg.X, msg.Y), (0, 170), (0, -170)]
#  #rospy.loginfo('IPS:I got:  X:%f, Y:%f, Yaw:%f', msg.X, msg.Y, msg.Yaw)
#  if not prev_velocity is None:
#    steering_angle = controller.stanley_steering(waypoints, msg, prev_velocity, 0.01)
#    waypt = steering_angle['waypt']
#    rospy.loginfo('IPS:I got:  X:%f, Y:%f, Yaw:%f, Delta:%f, TX:%f, TY:%f',
#                  msg.X, msg.Y, msg.Yaw, steering_angle['angle'], waypoints[waypt][0], waypoints[waypt][1])

#  writer = csv.writer(open("/home/administrator/data.csv", "a"))
#  writer.writerow([vel_cmd.linear.x, vel_cmd.angular.z, msg.X, msg.Y, msg.Yaw])


#def gps_callback(msg):

#def gps_callback(msg):
#  rospy.loginfo("GPS:I got: [%c] as status", msg.status)




def lidar_callback(msg):
  '''Callback to handle LIDAR feedback information.'''
  rospy.loginfo("LIDAR:I got: [%f] as range_min", msg.range_min)


if __name__ == '__main__':
  rospy.init_node('main_control')
  rospy.loginfo('initialised')

  vel_cmd = Twist()

#  lidar_sub = rospy.Subscriber(
#      'scan',
#      LaserScan,
#      lidar_callback,
#      queue_size=1000)
#  gps_sub = rospy.Subscriber(
#      'fix',
#      LaserScan,
#      gps_callback,
#      queue_size=1000)
#  gps_sub = rospy.Subscriber(
#      'gps_message',
#      LaserScan,
#      gps_callback,
#      queue_size=1000)
  enc_sub = rospy.Subscriber(
      '/clearpath/robots/default/data/raw_encoders',
      RawEncoders,
      encoder_callback,
      queue_size=1000)
#  ips_sub = rospy.Subscriber(
#      'indoor_pos',
#      ips_msg,
#      ips_callback,
#      queue_size=1000)

  vel_pub = rospy.Publisher('/clearpath/robots/default/cmd_vel', Twist)
  i = 0
  loop_rate = rospy.Rate(20)

  while not rospy.is_shutdown() and i < 20*10:
#    rospy.loginfo("LOOP")
    vel_cmd.linear.x = max(-100, min(100, linear_velocity_cmd))
    vel_cmd.angular.z = 0
    vel_pub.publish(vel_cmd)

    loop_rate.sleep()



  rospy.signal_shutdown(0)

