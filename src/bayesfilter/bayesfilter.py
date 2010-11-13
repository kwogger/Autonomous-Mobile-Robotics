'''
Occupancy Grid and Bayesian Filtering

A vehicle manoeuvres about a random grid, taking noisy measurements of the
location, attempting to localise itself as it moves. To keep things simple, the
vehicle does not avoid obstacles, follows a fixed route and receives
full/empty measurements of the 8 cells around it.

Created on 2010-10-21

@author: Michael Kwan
'''

import math
import numpy as np

if __name__ == '__main__':
  # Simulation parameters
  # Define a simulation length
  T = 20
  # Define a grid size
  n = 10
  # Define a state size
  N = np.pow(n, 2)
  # Define a state vector and initial position for the vehicle
  x = np.zeros(N, T + 1)
  pos = np.array([5, 5])
  x[pos[0]+n*(pos[1]-1)] = 1
  # Define an input vector
  u = np.zeros(1,T)
  # Define a measurement vector
  y = np.zeros(N,T)
  
  
  ## Create the motion model, 
  # which moves up, right, down or left from each possible state, except when
  # that would move out of the room.
  # p(x_t | x_t-1,u_t)
  mot_mod = np.zeros(N, N, 4)
  for i in xrange(n):
    for j in xrange(n):
      cur = i+j*n;
      # Move up
      if j > 0:
        mot_mod[cur - n, cur, 0] = 0.8
        mot_mod[cur, cur, 0] = 0.2
      else:
        mot_mod[cur, cur, 0] = 1
      # Move right
      if i < n:
        mot_mod[cur+1,cur,1] = 0.8
        mot_mod[cur,cur,1] = 0.2
      else:
        mot_mod[cur,cur,1] = 1
      # Move down
      if j < n:
        mot_mod[cur+n,cur,2] = 0.8
        mot_mod[cur,cur,2] = 0.2
      else:
        mot_mod[cur,cur,2] = 1
      # Move left
      if i > 1:
        mot_mod[cur-1,cur,3] = 0.8
        mot_mod[cur,cur,3] = 0.2
      else:
        mot_mod[cur,cur,3] = 1


  ## Create the measurement model
  # Define locally first, in two axes about point i,j at center
  #p(y_t(i-1:i+1,j-1:j+1) | x_t(i,j))
  meas_mod_rel = np.array([[0.11, 0.11, 0.11],
                           [0.11, 0.12, 0.11],
                           [0.11, 0.11, 0.11]])
  # Convert to full measurement model
  # p(y_t | x_t)
  meas_mod = np.zeros(N, N)
  # Fill in non-boundary measurements
  for i=2:n-1
      for j=2:n-1
          cur = i+(j-1)*n;
          meas_mod(cur-n+[-1:1:1],cur) = meas_mod_rel(1,:); 
          meas_mod(cur+[-1:1:1],cur) = meas_mod_rel(2,:); 
          meas_mod(cur+n+[-1:1:1],cur) = meas_mod_rel(3,:); 
      end
  end
  
  # Fill in boundaries by dropping impossible measurements
  scale = 1 - sum(meas_mod_rel(1,:));
  for i=2:n-1
      #Top
      cur = i;
      meas_mod(cur+[-1:1:1],cur)=meas_mod_rel(2,:)/scale;
      meas_mod(cur+n+[-1:1:1],cur)=meas_mod_rel(3,:)/scale;
      #Right
      cur = i*n;
      meas_mod(cur-1+n*[-1:1:1],cur)=meas_mod_rel(:,1)/scale;
      meas_mod(cur+n*[-1:1:1],cur)=meas_mod_rel(:,2)/scale;
      #Bottom
      cur = (n-1)*n+i;
      meas_mod(cur-n+[-1:1:1],cur)=meas_mod_rel(1,:)/scale;
      meas_mod(cur+[-1:1:1],cur)=meas_mod_rel(2,:)/scale;
      #Left
      cur = (i-1)*n+1;
      meas_mod(cur+n*[-1:1:1],cur)=meas_mod_rel(:,2)/scale;
      meas_mod(cur+1+n*[-1:1:1],cur)=meas_mod_rel(:,3)/scale;
  end
  
  # Fill in corners, assume fixed 
  meas_mod([1 2 n+1 n+2], 1) = .25;
  meas_mod([n-1 n 2*n-1 2*n], n) = .25;
  meas_mod([n*(n-2)+1 n*(n-2)+2 n*(n-1)+1 n*(n-1)+2],n*(n-1)+1) = .25;
  meas_mod([n*(n-1)-1 n*(n-1) n*n-1 n*n],n*n) = .25;
  
  ## Define an initial belief for the vehicle states
  # 0 = no obstacle, 1 = obstacle
  bel = 1/N^2*ones(N,1);
  belp = bel;
  
  figure(1);clf; hold on;
  beliefs = reshape(bel,n,n);
  imagesc(beliefs);
  plot(pos(2),pos(1),'ro','MarkerSize',6,'LineWidth',2)
  colormap(summer);
  title('True state and beliefs')
  F = getframe;
  
  ## Main loop
  for t=1:T
      ## Simulation
      # Select motion input
      u(t) = ceil(4*rand(1));
      # Select a motion
      thresh = rand(1);
      new_x = find(cumsum(squeeze(mot_mod(:,:,u(t)))*x(:,t))>thresh,1);
      # Move vehicle
      x(new_x,t+1) = 1;
      # Take measurement
      thresh = rand(1);
      new_y = find(cumsum(meas_mod(:,:)*x(:,t+1))>thresh,1);
      y(new_y,t) = 1;
      # Store for plotting
      curx = reshape(x(:,t+1)',n,n);
      [xi, xj] = find(curx >0);
      xp(:,t) = [xi;xj]; 
      cury = reshape(y(:,t)',n,n);
      [yi, yj] = find(cury >0);
      yp(:,t) = [yi;yj]; 
      xt(t) = new_x;
      yt(t) = new_y;
      
      ## Bayesian Estimation
      # Prediction update
      belp = squeeze(mot_mod(:,:,u(t)))*bel;
      # Measurement update
      bel = meas_mod(new_y,:)'.*belp;
      bel = bel/norm(bel);
  
      [pmax y_bel(t)] = max(bel); 
      
      ## Plot beliefs
      beliefs = reshape(bel,n,n);
      imagesc(beliefs);
      plot(xj,xi,'ro','MarkerSize',6,'LineWidth',2)
      plot(yj,yi,'bx','MarkerSize',4,'LineWidth',1)
      colormap(summer);
      title('True state and beliefs')
      pause(0.1);
      F(t+1) = getframe;
  end
  
  figure(2);clf; hold on;
  plot(xt);
  plot(yt,'r--');
  plot(y_bel, 'g:')
  title('State, measurement and max belief')
  xlabel('Time')
  ylabel('Location');
  
  movie2avi(F,'bayesgrid.avi','fps',1