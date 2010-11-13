function [m,ind] = closestfeature(map,x)
% Finds the closest feature in a map based on distance

ind = 0;
mind = inf;

for i=1:length(map(1,:))
    d = sqrt( (map(1,i)-x(1))^2 + (map(2,i)-x(2))^2);
    if (d<mind)
        mind = d;
        ind = i;
    end
end
m = map(:,ind);