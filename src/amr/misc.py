'''
Created on Dec 15, 2010

@author: Michael Kwan
'''

import numpy as np

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
