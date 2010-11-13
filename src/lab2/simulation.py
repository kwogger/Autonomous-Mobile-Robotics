'''
Created on 2010-10-15

@author: Michael Kwan
'''
import math
import numpy as np
import controller
from matplotlib import pyplot


def drawbox(x, y, h, scale, fig):
  '''This function plots a box on the fig at the specified pos, heading & scale.

  Args:
    x: The x coordinate of the box
    y: The y coordinate of the box
    h: The heading of the box
    scale: The size of the box
    fig: The figure number
  '''
  # Car outline
  box = np.mat([[-1, -0.5],
             [1, -0.5],
             [1, 0.5],
             [-1, 0.5],
             [-1, -0.5]])

  # Size scaling
  box = scale * box

  # Rotation matrix
  R = np.mat([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
  box = (R * box.H).H

  # Centre
  box[:, 0] = box[:, 0] + x
  box[:, 1] = box[:, 1] + y

  # Plot
  pyplot.figure(fig)
  pyplot.plot(box[:, 0], box[:, 1], 'b', linewidth=2)
  pyplot.axis('equal')


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
    drawbox(p[t, 0], p[t, 1], x[t, 1], .3, 1);

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
  Tmax = 60 * 1.5
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
    drawbox(p[t, 0], p[t, 1], x[t, 1], .3, 1);

  pyplot.xlabel('x (m)')
  pyplot.ylabel('y (m)')
  pyplot.axis('equal')
  pyplot.show()


def ekf_simulation():
  # Discrete time step
  dt = 0.1;
  
  # Initial State
  x0 = np.array([20, -2, 3])
  
  # Prior
  mu = np.array([22, -1.8, 3.5])
  S = 1 * np.eye(3)  # covariance (Sigma)
  
  # Discrete motion model
  Ad = np.array([[1, dt, 0], [0, 1, 0], [0, 0, 1]])

  R = 1.0e-4 * np.eye(3)
  Re, RE = np.linalg.eig(R)
  # Measurement model defined below
  Q = .0001
  
  # Simulation Initializations
  Tf = 10
  T = np.arange(0, Tf, dt)
  n = len(Ad[0])
  x = np.zeros((n, len(T)))
  x[:, 0] = x0
  m = 1 #len(Q[:, 0])
  y = np.zeros((m, len(T)))
  mup_S = np.zeros((n, len(T)))
  mu_S = np.zeros((n, len(T)))
  K_S = np.zeros((n, len(T)))

  Re = np.array(1.0e-4*np.eye(3))
  RE = np.array(np.eye(3))

  # Main loop
  for t in xrange(1, len(T)):
    # Simulation
    # Select a motion disturbance
    e = np.dot(np.dot(RE, np.sqrt(Re)), np.random.randn(n, 1))
    # Update state
    x[:, t] = np.dot(Ad, x[:, t - 1]) + e.T

    # Take measurement
    # Select a motion disturbance
    d = np.dot(np.sqrt(Q), np.random.randn(m, 1))
    # Determine measurement
    y[:, t] = np.sqrt(x[0, t] ** 2 + x[2, t] ** 2) + d

    # Extended Kalman Filter Estimation
    # Prediction update
    mup = np.dot(Ad, mu)
    Sp = np.dot(np.dot(Ad, S), Ad.T) + R

    # Linearization
    Ht = np.array([[mup[0] / np.sqrt(mup[0] ** 2 + mup[2] ** 2), 0, mup[2] / np.sqrt(mup[0] ** 2 + mup[2] ** 2)]])
    # Measurement update
    #K = np.dot(np.dot(Sp, Ht.T), np.linalg.inv(np.dot(np.dot(Ht, Sp), Ht.T) + Q))
    K = np.dot(np.dot(Sp, Ht.T), 1/(np.dot(np.dot(Ht, Sp), Ht.T) + Q))
    mu = mup + np.dot(K, (y[:, t] - np.sqrt(mup[0] ** 2 + mup[2] ** 2)))
    S = np.dot((np.eye(n) - np.dot(K, Ht)), Sp)

    # Store results
    mup_S[:, t] = mup
    mu_S[:, t] = mu
    K_S[:, t] = K.T


    # Plot results
    pyplot.figure(1)
    pyplot.clf()
    pyplot.hold(True)

    pyplot.plot(0, 0, 'bx', markersize=6, linewidth=2)
    pyplot.plot([20, -1], [0, 0], 'b--')
    pyplot.plot(x[0, 1:t], x[2, 1:t], 'ro--')
    pyplot.plot(mu_S[0, 1:t], mu_S[2, 1:t], 'bx--')
    #mu_pos = [mu(1) mu(3)];
    #S_pos = [S(1, 1) S(1, 3); S(3, 1) S(3, 3)];
    #error_ellipse(S_pos, mu_pos, 0.75);
    #error_ellipse(S_pos, mu_pos, 0.95);
    pyplot.title('True state and belief')
    pyplot.axis([-1, 20, -1, 10])
    #F[t - 1] = getframe
  pyplot.show()






if __name__ == '__main__':
  ekf_simulation()
#  waypts = [
#      (2, 2),
#      (2, 10),
#      (10, 10),
#      (10, 2),
#      (2, 2),
#      ]
#  stanley_steering_simulation(waypts)
