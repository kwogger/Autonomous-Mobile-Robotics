'''
Created on Dec 15, 2010

@author: Michael Kwan
'''

import numpy as np
from matplotlib import pyplot


def closest_feature(map, x):
  '''Finds the closest feature in a map based on distance.'''
  ind = 0
  mind = np.inf

  for i in xrange(len(map)):
    d = np.sqrt(np.power(map[i, 0] - x[0], 2) + np.power(map[i, 1] - x[1], 2))
    if d < mind:
      mind = d
      ind = i
  return (map[ind, :].T, ind)


def distToEdge(v, edge):
  '''Computes the shortest distance to a line segment, returns distance and closest point.'''
  S = edge[2:3] - edge[0:1]
  S1 = v - edge[0:1]
  m1 = np.asscalar(np.dot(S1, S.T))
  if m1 <= 0:
    d = np.linalg.norm(v - edge[0:1])
    pt = edge[0:1]
  else:
    m2 = np.asscalar(np.dot(S, S.T))
    if m2 <= m1:
      d = np.linalg.norm(v - edge[2:3])
      pt = edge[2:3]
    else:
      b = m1 / m2
      pt = edge[0:1] + np.dot(b, S)
      d = np.linalg.norm(v - pt)
  return d, pt


def minDistToEdges(v, edges, fig=None):
  '''Computes the shortest distance to a sequence of edges, which may form a
  path, for example.  Returns the min distance, the closest point, and the 
  list of distances and closest points for each edge.'''
  n = len(edges)
  d = np.zeros(n)
  pt = np.zeros((n, 2))
  for i in xrange(len(edges)):
    d[i], pt[i, :] = distToEdge(v, edges[i - 1, :])

  ind = np.argmin(d)
  minD = d[ind]
  minPt = pt[int(ind) - 1, :]
  if not fig is None:
    pyplot.figure(fig)
    pyplot.hold(True)
    l = np.array([v, minPt])
    pyplot.plot(l[:, 0], l[:, 1], 'g')
  return (minD, minPt, d, pt, ind)


def polygonal_world(posMinBound, posMaxBound, minLen, maxLen, numObsts,
                    startPos, endPos, obst_buffer, max_count):
  '''Creates an environment of non-overlapping rectangles  within a bound.

  In addition, none of the rectangles can be on top of the start or end
  position.
  '''
  a = np.zeros((numObsts, 4, 2))
  b = np.zeros((numObsts, 4))

  pos = np.zeros((numObsts, 4))
  length = np.zeros((numObsts, 4))
  fake_len = np.zeros((numObsts, 4))
  theta = np.zeros(numObsts)

  for j in xrange(numObsts):
    a[j, 0, :] = np.array([0, -1])
    a[j, 1, :] = np.array([1, 0])
    a[j, 2, :] = np.array([0, 1])
    a[j, 3, :] = np.array([-1, 0])

  #create the number of obsts
  count = 0
  for i in xrange(numObsts):
    #loop while there are collisions with obstacles
    while True:
      #generate random positions and lengths
      pos[i, :] = posMinBound + [np.random.rand(1) * (posMaxBound[0] - posMinBound[0]), np.random.rand(1) * (posMaxBound[1] - posMinBound[1])];
      length[i, :] = [np.random.rand(1) * (maxLen.a - minLen.a) + minLen.a, np.random.rand(1) * (maxLen.b - minLen.b) + minLen.b];
      fake_len[i, :] = length[i, :] + obst_buffer
      theta[i] = np.random.rand(1) * np.pi

      rotationMatrix = [[np.cos(theta[i]), np.sin(theta[i])],
                        [-np.sin(theta[i]), np.cos(theta[i])]];
      #find the points
      pts = [[-length[i, 0] / 2, -length[i, 1] / 2],
             [length[i, 0] / 2, -length[i, 1] / 2],
             [length[i, 0] / 2, length[i, 1] / 2],
             [-length[i, 0] / 2, length[i, 1] / 2]]
      fake_pts = [[-fake_len[i, 0] / 2, -fake_len[i, 1] / 2],
                  [ fake_len[i, 0] / 2, -fake_len[i, 1] / 2],
                  [fake_len[i, 0] / 2, fake_len[i, 1] / 2],
                  [-fake_len[i, 0] / 2, fake_len[i, 1] / 2]]
      for j in xrange(4):
        pts[j, :] = (np.dot(rotationMatrix, pts[j, :].T)).T + np.array([pos[i, 0], pos[i, 1]])
        fake_pts[j, :] = np.dot(rotationMatrix, fake_pts[j, :].T).T + np.array([pos[i, 0], pos[i, 1]])

      ##need to redo these checks
      #check to see if it is outside the region
      if (np.min(fake_pts[:, 0]) <= posMinBound[0]
          or np.max(fake_pts[:, 0]) >= posMaxBound[0]
          or np.min(fake_pts[:, 1]) <= posMinBound[1]
          or np.max(fake_pts[:, 1]) >= posMaxBound[1]):
        continue
      if (np.min(pts[:, 0]) < startPos[0]
          and np.max(pts[:, 0]) > startPos[0]
          and np.min(pts[:, 1]) < startPos[1]
          and np.max(pts[:, 1]) > startPos[1]):
        continue
      #check to see if it is on top of the end pos
      if (np.min(pts[:, 0]) < endPos[0]
          and np.max(pts[:, 0]) > endPos[0]
          and np.min(pts[:, 1]) < endPos[1]
          and np.max(pts[:, 1]) > endPos[1]):
        continue

      #check to see if it collided with any of the other obstacles
      collided = 0;
      for j in xrange(i):
        #check for collision
        if polygonsOverlap(fake_pts, ptsStore[:, j * 2:j * 2 + 1]):
          collided = 1;
          break;

      if not collided:
        break
      count = count + 1
      if count >= max_count:
        a = np.array([])
        b = np.array([])
        ptsStore = np.array([])
        return a, b, ptsStore

    ptsStore[:, i * 2:i * 2 + 1] = pts

    for j in xrange(4):
      next = j + 1;
      if j == 3:
        next = 0
      temp = np.array([[-(pts[j, 1] - pts[next, 1])], [(pts[j, 0] - pts[next, 0])]])
      temp = (temp / np.linalg.norm(temp)).flatten()
      a[i, j, 0] = temp[0]
      a[i, j, 1] = temp[1]

    #calculate the b matrix
    for k in xrange(4):
      for j in xrange(2):
        b[i, k] = b[i, k] + pts[k, j] * a[i, k, j]

  return a, b, ptsStore


def function val = polygonsOverlap(poly1, poly2)
  '''Checks if two polygons intersect.

  Assumes Polygon vertices are ordered counter clockwise (can be enforced with
  our scripts)

  Args:
      poly1: N1x2 matrix of x,y coordinates of polygon vertices
      poly2: N2x2 matrix of x,y coordinates of polygon vertices
  '''

  val = False

  # Simple test to check if 1 is fully or partially enclosed in polygon 2
  if sum(inpolygon(poly1(:, 1), poly1(:, 2), poly2(:, 1), poly2(:, 2)))
      val = true;
      return
  end

  # Simple test to check if 2 is fully or partially enclosed in polygon 1
  if sum(inpolygon(poly2(:, 1), poly2(:, 2), poly1(:, 1), poly1(:, 2)))
      val = true;
      return
  end

  # Close the polygons
  poly1 = [poly1;poly1(1, :)];
  obstEdges = [poly2, [poly2(2:end, :);poly2(1, :)]];
  # Loop over all possible intersections
  for vertex = 1:(length(poly1) - 1)
      if (CheckCollision(poly1(vertex, :), poly1(vertex + 1, :), obstEdges))
          val = true;
          return
      end
  end
  return val
