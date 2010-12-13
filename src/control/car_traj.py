'''
Created on Dec 12, 2010

@author: Michael Kwan
'''
import math
from amr import draw
from amr import controller
from matplotlib import pyplot
import numpy as np

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

  # Initial conditions [e, psi]
  x0 = np.array([0, 45 * math.pi / 180])

  # Setup the simulation time
  Tmax = 45  # in seconds
  dt = 0.001  # time step
  T = np.arange(0, Tmax, dt)

  # Setup simulation
  xd = np.zeros((len(T) - 1, 2))  # Derivative of state
  x = np.zeros((len(T), 2))  # State
  x[0, :] = x0  # Initial condition
  delta = np.zeros((len(T), 1)) # Angle change
  p = np.zeros((len(T), 2))  # Position
  p[0, :] = [0, 0]  # Initial position
  waypt = 0 # Initial waypoint to travel from

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

  # Setup trajectory plot
  pyplot.figure(1)
  pyplot.clf()
  pyplot.hold(True)
  pyplot.xlabel('x (m)')
  pyplot.ylabel('y (m)')
  pyplot.axis('equal')

  # Plot trajectory
  pyplot.plot(x0[0] * np.sin(x0[1]), x0[0] * np.cos(x0[1]))
  pyplot.plot(p[:, 0], p[:, 1], 'r')
  pyplot.plot([pt[0] for pt in waypts], [pt[1] for pt in waypts], 'go')

  for t in xrange(0, len(T), 300):
    draw.drawbox(p[t, 0], p[t, 1], x[t, 1], .3, 1);


  # Phase portrait
  # Create a grid over values of e and psi
  e, psi = np.meshgrid(np.arange(-10, 10, .5), np.arange(-3, 3, .2))
  # Calculate steering angle at each point
  delta = np.maximum(-delta_max,
                     np.minimum(delta_max, psi + np.arctan2(k * e, v)))
  ed = v * np.sin(psi - delta)  # Find crosstrack derivative
  psid = -(v * np.sin(delta)) / l  # Find heading derivative

  # Find border of max region
  psibplus = -np.arctan2(k * e[0, :], v) + delta_max
  # Find border of min region
  psibminus = -np.arctan2(k * e[0, :], v) - delta_max

  # Setup phase portrait plot
  pyplot.figure(2)
  pyplot.clf()
  pyplot.hold(True)
  pyplot.axis([-10, 10, -3, 3])
  pyplot.xlabel('e (m)')
  pyplot.ylabel('\psi (rad)')

  # Plot phase portrait
  pyplot.quiver(e, psi, ed, psid)
  pyplot.plot(e[0, :], psibplus, 'r', linewidth=2)
  pyplot.plot(e[0, :], psibminus, 'r', linewidth=2)

  pyplot.show()


if __name__ == '__main__':
  # Define a set of waypoints to traverse
  waypts = [
      (2, 2),
      (2, 5),
      (5, 5),
      (5, 2),
      (2, 2),
      ]
  stanley_steering_simulation(waypts)
