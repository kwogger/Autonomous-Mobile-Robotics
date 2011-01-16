'''
Functions to draw various shapes in matlab.

Created on Nov 14, 2010

@author: Michael Kwan
'''
import math
import numpy as np
from matplotlib import pyplot
from scipy import stats


def drawcar(x, y, h=0, scale=1, fig=0, style='b'):
  '''Plots a car at position x,y heading h and size scale on figure number fig.

  The default x,y = (0,0), h = 0 points to the right, and scale=1 plots a car
  with a body of radius 2 units.

  Args:
    x: The x-coordinate to draw the car
    y: The y-coordinate to draw the car
    h: The heading of the car
    scale: The scale to draw the car
    fig: The number of the figure to draw the car in.
  '''
  # Make a circle for the body
  t = np.mat(np.arange(0, 2 * math.pi, 0.01))
  length = len(t.T)
  bx = np.sin(t)
  by = np.cos(t)

  # Wheel locations on body
  wh1 = round(length / 4) - 1
  wh2 = round(3 * length / 4) - 1

  # Draw the wheels
  wwidth = 0.2
  wheight = 0.4
  w = np.mat([[0, -wheight],
              [wwidth, -wheight],
              [ wwidth, wheight],
              [ 0, wheight],
              [ 0, 0]])

  # Body top
  top = round(length / 2)
  # Top pointer
  pwidth = 0.1
  pheight = 0.2
  tp = np.mat([[pwidth / 2, 0],
               [ 0, -pheight],
               [ -pwidth / 2, 0],
               [ pwidth / 2, 0]])

  # Car outline
  car = np.bmat([[bx[0, :wh1].T, by[0, :wh1].T],
                [bx[0, wh1] + w[:, 0], by[0, wh1] + w[:, 1]],
                [bx[0, wh1:wh2].T, by[0, wh1:wh2].T],
                [bx[0, wh2] - w[:, 0], by[0, wh2] - w[:, 1]],
                [bx[0, wh2:].T, by[0, wh2:].T]])

  point = np.bmat([bx[0, top] + tp[:, 0], by[0, top] + tp[:, 1]])

  # Size scaling
  car = scale * car
  point = scale * point

  # Rotation matrix
  R = np.mat([[np.cos(h + math.pi / 2), -np.sin(h + math.pi / 2)],
              [np.sin(h + math.pi / 2), np.cos(h + math.pi / 2)]])
  car = (R * car.T).T
  point = (R * point.T).T

  # Centre
  car[:, 0] = car[:, 0] + x
  car[:, 1] = car[:, 1] + y
  point[:, 0] = point[:, 0] + x
  point[:, 1] = point[:, 1] + y

  # Plot
  pyplot.figure(fig)
  pyplot.plot(car[:, 0], car[:, 1], style)
  pyplot.plot(car[:, 0], car[:, 1], style)
  pyplot.plot(point[:, 0], point[:, 1], 'r')
  pyplot.axis('equal')


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


def error_ellipse(C, mu=np.array([0, 0]), conf=0.5, scale=1, style='',
                  clip=float('inf')):
  '''Plot an error ellipse, or ellipsoid, defining confidence region.

  ERROR_ELLIPSE(C22) - Given a 2x2 covariance matrix, plot the associated error
  ellipse, at the origin. It returns a graphics handle of the ellipse that was
  drawn.

  ERROR_ELLIPSE(C33) - Given a 3x3 covariance matrix, plot the associated error
  ellipsoid, at the origin, as well as its projections onto the three axes.
  Returns a vector of 4 graphics handles, for the three ellipses (in the X-Y,
  Y-Z, and Z-X planes, respectively) and for the ellipsoid.

  ERROR_ELLIPSE(C,MU) - Plot the ellipse, or ellipsoid, centered at MU, a vector
  whose length should match that of C (which is 2x2 or 3x3).

  ERROR_ELLIPSE(...,'Property1',Value1,'Name2',Value2,...) sets the values of
  specified properties, including:
    'C' - Alternate method of specifying the covariance matrix
    'mu' - Alternate method of specifying the ellipse (-oid) center
    'conf' - A value betwen 0 and 1 specifying the confidence interval.
      the default is 0.5 which is the 50% error ellipse.
    'scale' - Allow the plot the be scaled to difference units.
    'style' - A plotting style used to format ellipses.
    'clip' - specifies a clipping radius. Portions of the ellipse, -oid,
      outside the radius will not be shown.

  NOTES: C must be positive definite for this function to work properly.
  '''
  [r, c] = np.shape(C)

  x0 = mu[0]
  y0 = mu[1]

  # Compute quantile for the desired percentile
  k = stats.chi.ppf(conf, r)  # r is the number of dimensions (degrees of freedom)


#  if r == 3 and c == 3:
#    z0 = mu[2]
#
#    # C is 3x3; extract the 2x2 matricies, and plot the associated error
#    # ellipses. They are drawn in space, around the ellipsoid; it may be
#    # preferable to draw them on the axes.
#    Cxy = C[0:1, 0:1]
#    Cyz = C[1:2, 1:2]
#    Czx = C[[2, 0], [2, 0]]
#
#    [x, y, z] = getpoints(Cxy, clip)
#    h1 = pyplot.plot(x0 + k * x, y0 + k * y, z0 + k * z, style)
#    pyplot.hold(True)
#    [y, z, x] = getpoints(Cyz, clip)
#    h2 = pyplot.plot(x0 + k * x, y0 + k * y, z0 + k * z, style)
#    pyplot.hold(True)
#    [z, x, y] = getpoints(Czx, clip)
#    h3 = pyplot.plot(x0 + k * x, y0 + k * y, z0 + k * z, style)
#    pyplot.hold(True)
#
#
#    eigval, eigvec = np.linalg.eig(C);
#
#    [X, Y, Z] = ellipsoid(0, 0, 0, 1, 1, 1)
#    XYZ = [X[:], Y[:], Z[:]] * sqrt(eigval) * eigvec.T
#
#    X[:] = scale * (k * XYZ[:, 0] + x0)
#    Y[:] = scale * (k * XYZ[:, 1] + y0)
#    Z[:] = scale * (k * XYZ[:, 2] + z0)
#    h4 = surf(X, Y, Z)
#    pyplot.gray()
#    alpha(0.3)
#    camlight
#    return [h1 h2 h3 h4]
  if r == 2 and c == 2:
    x, y, _ = getpoints(C, clip)
    return pyplot.plot(scale * (x0 + k * x), scale * (y0 + k * y), style)


def getpoints(C, clipping_radius=None):
  n = 100  # Number of points around ellipse
  p = np.arange(0, 2 * math.pi + math.pi / n, math.pi / n)  # angles around a circle

  eigval, eigvec = np.linalg.eigh(C)  # Compute eigen-stuff
  xy = np.dot(np.dot(np.array([np.cos(p), np.sin(p)]).T, np.sqrt(np.eye(2) * eigval)), eigvec.T)  # Transformation
  x = xy[:, 0]
  y = xy[:, 1]
  z = np.zeros(np.shape(x))

  # Clip data to a bounding radius
  if clipping_radius is None:
    r = np.sqrt(np.sum(np.power(xy, 2), 2)) # Euclidian distance (distance from center)
    x[r > clipping_radius] = np.nan
    y[r > clipping_radius] = np.nan
    z[r > clipping_radius] = np.nan

  return (x, y, z)
