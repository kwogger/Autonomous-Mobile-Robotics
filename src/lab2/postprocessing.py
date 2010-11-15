'''
Created on 2010-11-15

@author: Michael Kwan
'''
import collections
import csv


if __name__ == '__main__':
  # Load the data
  enc_reader = csv.reader(open('enc.csv', 'r'))
  gps_reader = csv.reader(open('gps.csv', 'r'))
  enc_data = collections.deque()
  gps_data = collections.deque()
  for t, tick in enc_reader:
    enc_data.append((float(t)/1e9, int(tick)))
  for t, lat, long, alt, track, err_track, speed in gps_reader:
    if not t == '0.0':
      gps_data.append((float(t), float(lat), float(long)))
  sorted_data = collections.deque()
  while len(enc_data) > 0 and len(gps_data) > 0:
    if enc_data[0][0] > gps_data[0][0]:
      sorted_data.append(gps_data.popleft())
    else:
      sorted_data.append(enc_data.popleft())
  if len(enc_data) > 0:
    for data in enc_data:
      sorted_data.append(data)
  else:
    for data in gps_data:
      sorted_data.append(data)

  # Data has been loaded. Run the EKF now.
