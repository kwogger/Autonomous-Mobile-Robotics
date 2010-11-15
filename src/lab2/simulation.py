'''
Created on 2010-10-15

@author: Michael Kwan
'''
import math
import numpy as np
import controller
from matplotlib import pyplot
from amr import draw


def matlab_steering_simulation():
  '''Non-linear steering simulation converted from MATLAB code.''' 
  # Trajectory tracking
  # Fixed vehicle parameters
  v = 5  # Speed
  delta_max = 25 * math.pi / 180  # max steering angle
  k = 2.5  # Gain
  l = 0.238  # Car length

  # Desired line state through 0,0 [theta]
  xp = 0  # Line heading

  # Initial conditions in [e psi]
  x0 = np.mat([5, 0])  # translational offset
  #x0 = [0 2]; # heading offset
  #x0 = mat([5, 2])  # both

  # Simulation time
  Tmax = 3  # End point
  dt = 0.001  # Time step
  T = np.arange(0, Tmax, dt)  # Time vector

  # Simulation setup
  xd = np.mat(np.zeros((len(T) - 1, 2)))  # Derivative of state ([edot psidot])
  x = np.mat(np.zeros((len(T), 2)))  # State ([e psi])
  x[0, :] = x0  # Initial condition
  delta = np.mat(np.zeros((len(T), 1)))  # Steering angles
  p = np.mat(np.zeros((len(T), 2)))  # Position in x,y plane
  p[0, :] = x0[0, 0] * np.mat([np.sin(xp), np.cos(xp)])  # Initial position

  for i in xrange(len(T) - 1):
    # Calculate steering angle
    delta[i] = max(-delta_max, min(delta_max,
                                   x[i, 1] + np.arctan2(k * x[i, 0], v)))
    # State derivatives
    xd[i, 0] = v * np.sin(x[i, 1] - delta[i])
    xd[i, 1] = -(v * np.sin(delta[i])) / l
    # State update
    x[i + 1, 0] = x[i, 0] + dt * xd[i, 0]
    x[i + 1, 1] = x[i, 1] + dt * xd[i, 1]
    # Position update
    p[i + 1, 0] = p[i, 0] + dt * v * np.cos(x[i, 1] - delta[i] - xp)
    p[i + 1, 1] = p[i, 1] + dt * v * np.sin(x[i, 1] - delta[i] - xp)

  ## Plotting

  # Trajectory
  pyplot.figure(1)
  pyplot.clf()
  pyplot.hold(True)

  pyplot.plot([0, Tmax * v * np.cos(xp)], [0, Tmax * v * np.sin(xp)], 'b--')
  pyplot.plot(x0[0, 0] * np.sin(x0[0, 1]), x0[0, 0] * np.cos(x0[0, 1]))
  pyplot.plot(p[:, 0], p[:, 1], 'r')

  for t in xrange(0, len(T), 300):
    draw.drawbox(p[t, 0], p[t, 1], x[t, 1], .3, 1);

  pyplot.xlabel('x (m)')
  pyplot.ylabel('y (m)')
  pyplot.axis('equal')

  # Phase portrait
  (e, psi) = np.meshgrid(np.arange(-10, 10.5, .5), np.arange(-3, 3.2, .2))  # Create a grid over values of e and psi
  delta = np.maximum(-delta_max, np.minimum(delta_max, psi + np.arctan2(k * e, v)))  # Calculate steering angle at each point
  ed = v * np.sin(psi - delta)  # Find crosstrack derivative
  psid = -(v * np.sin(delta)) / l  # Find heading derivative

  psibplus = -np.arctan2(k * e[0, :], v) + delta_max  # Find border of max region
  psibminus = -np.arctan2(k * e[0, :], v) - delta_max  # Find border of min region

  pyplot.figure(2)
  pyplot.clf
  pyplot.hold(True)
  pyplot.quiver(e, psi, ed, psid)
  pyplot.plot(e[0, :], psibplus, 'r', linewidth=2)
  pyplot.plot(e[0, :], psibminus, 'r', linewidth=2)
  pyplot.axis([-10, 10, -3, 3])
  pyplot.xlabel('e (m)')
  pyplot.ylabel('\psi (rad)')

  pyplot.show()


def stanley_steering_simulation(waypts):
  '''Stanley steering simulation.

  Args:
    waypts: The waypoints in which the robot should move through.
  '''
  # Vehicle parameters
  v = 0.4  # Speed
  delta_max = 15 * math.pi / 180  # max steering angle
  k = 0.25  # Gain
  l = 0.238  # Car length

  # Initial conditions
  x0 = np.array([0, 45 * math.pi / 180, 0])

  # Setup the simulation time
  Tmax = 60*1.5
  dt = 0.001
  T = np.arange(0, Tmax, dt)

  # Setup simulation
  xd = np.zeros((len(T) - 1, 3))  # Derivative of state
  x = np.zeros((len(T), 3))  # State
  x[0, :] = x0  # Initial condition
  delta = np.zeros((len(T), 1))
  p = np.zeros((len(T), 2))
  p[0, :] = [0, 0]  # Initial position
  waypt = 0

  for i in xrange(len(T) - 1):
    # Steering calculation
    steering = controller.stanley_steering(
        waypts[waypt:waypt + 3],
        p[i, :],
        x[i, 1],
        v,
        k,
        )
    waypt = waypt + steering['waypt']
    delta[i] = max(-delta_max, min(delta_max, steering['angle']))
    # PLEASE MAKE SURE THIS IS CORRECT AND ALL
    # State derivatives
    xd[i, 0] = v * np.sin(x[i, 1] - delta[i])
    xd[i, 1] = -(v * np.sin(delta[i])) / l
    # State update
    x[i + 1, 0] = x[i, 0] + dt * xd[i, 0]
    x[i + 1, 1] = x[i, 1] + dt * xd[i, 1]
    # Position update
    p[i + 1, 0] = p[i, 0] + dt * v * np.cos(x[i, 1] - delta[i])
    p[i + 1, 1] = p[i, 1] + dt * v * np.sin(x[i, 1] - delta[i])


  ## Plotting

  # Trajectory
  pyplot.figure(1)
  pyplot.clf()
  pyplot.hold(True)

  pyplot.plot(x0[0] * np.sin(x0[1]), x0[0] * np.cos(x0[1]))
  pyplot.plot(p[:, 0], p[:, 1], 'r')
  pyplot.plot([pt[0] for pt in waypts], [pt[1] for pt in waypts], 'go')

  for t in xrange(0, len(T), 300):
    draw.drawbox(p[t, 0], p[t, 1], x[t, 1], .3, 1);

  pyplot.xlabel('x (m)')
  pyplot.ylabel('y (m)')
  pyplot.axis('equal')
  pyplot.show()


if __name__ == '__main__':
  waypts = [
      (2, 2),
      (2, 10),
      (10, 10),
      (10, 2),
      (2, 2),
      ]
  stanley_steering_simulation(waypts)
