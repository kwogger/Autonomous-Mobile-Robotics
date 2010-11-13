'''
Created on 2010-10-13

@author: Michael Kwan
'''
import csv
from matplotlib import pyplot


def decode_pid_csv(filename):
  # Extract the data
  reader = csv.reader(open(filename, 'r'))
  data = []
  for row in reader:
    data.append(row)
  cur_time = [r[0] for r in data if len(r) == 6]
  
  LABELS = ('cur_ticks', 'linear_velocity_cmd', 'prev_error', 'error_integral', 'prev_velocity')
  
  # Plot the data
  for i in xrange(1, 6):
    pyplot.figure(i)
    pyplot.clf()
    pyplot.hold(True)
    pyplot.plot(cur_time, [r[i] for r in data if len(r) == 6], 'r', label=LABELS[i-1])
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('y')
    pyplot.title(LABELS[i-1])
  
  pyplot.show()


if __name__ == '__main__':
  decode_pid_csv('data.csv')
