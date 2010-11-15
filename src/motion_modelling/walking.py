'''
Rolling polygon approximation to walking.

Created on 2010-11-13

@author: Michael Kwan
'''
import cv
import math
import numpy as np
import os
from matplotlib import pyplot

WIN_NAME = 'Walking Simulation'


if __name__ == '__main__':
  # Create a window and video to display and write the figure animation
  video_writer = cv.CreateVideoWriter(
      'walking.avi',
      cv.CV_FOURCC('I', '4', '2', '0'),
      15,
      (800, 600))

  # Time
  t = np.arange(0, 3, 0.01)

  # Step rate
  m = 2  # Hz

  # Leg length
  l = 1.0  # m

  # Step length 
  d = 1.0  # m

  # Polygon sides
  n = math.pi / math.asin(d / (2 * l))

  # Forward speed
  v = d * m  # m/s

  # Leg angle
  a = np.arcsin((-d / 2 + (v * t % d)) / (l))

  # CG height
  h = l * np.cos(a);

  # Contact point
  c = np.zeros(len(t))
  c[0] = d / 2
  for i in xrange(0, len(t)):
    # New step counter
    if i > 0:
      if a[i] - a[i - 1] < 0:
        c[i] = c[i - 1] + d
      else:
        c[i] = c[i - 1]

  # Walking Figure
  k = 1;
  for i in xrange(0, len(t), 2):
    pyplot.figure(2)
    pyplot.clf()
    pyplot.hold(True)
    # Center of motion
    cm = np.mat([v * t[i], h[i]])
    # Grounded leg
    leg = np.mat([[c[i], 0],
                  [v * t[i], h[i]]])
    foot = np.concatenate((leg[0, :],
                           [[c[i] + 0.2 * l, 0]]))
    # Other (swinging) leg
    oleg = np.mat([[c[i] - d + (2 * v * t[i] % (2 * d)), 0.03],
                   [v * t[i], h[i]]])
    ofoot = np.concatenate((oleg[0, :],
                            oleg[0, :] + np.mat([0.2 * l, 0])))
    # Body
    body = np.concatenate((cm,
                           cm + np.mat([0, 0.6 * l])))
    # Head
    head = cm + np.mat([0, 0.6 * l])
    # Polygon
    s = np.arange(0, 2 * math.pi, 2 * math.pi / n)
    # Plotting
    pyplot.plot(cm[0, 0], cm[0, 1], 'ro', markersize=8 , linewidth=2)
    pyplot.plot(leg[:, 0], leg[:, 1], 'b', linewidth=2)
    pyplot.plot(foot[:, 0], foot[:, 1], 'b', linewidth=2)
    pyplot.plot(oleg[:, 0], oleg[:, 1], 'b', linewidth=2)
    pyplot.plot(ofoot[:, 0], ofoot[:, 1], 'b', linewidth=2)
    pyplot.plot(body[:, 0], body[:, 1], 'b', linewidth=2)
    pyplot.plot(head[0, 0], head[0, 1], 'bo', markersize=18 , linewidth=2)
    pyplot.plot(cm[0, 0] + l * np.sin(s + a[i]), cm[0, 1] + l * np.cos(s + a[i]), 'g')
    pyplot.axis('equal')
    pyplot.axis([-1, 6.5, -1, 3])
    if pyplot.waitforbuttonpress(1e-9):
      break

    # Write figure to file and display it on screen
    pyplot.savefig('walking.png')
    frame = cv.LoadImage('walking.png', cv.CV_LOAD_IMAGE_COLOR)
    cv.WriteFrame(video_writer, frame)

  # Remove the PNG buffer file used for video frame writing
  os.remove('walking.png')


# Known bug - polygon does not match motion if l not equal to d!
