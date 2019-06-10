function Y = force(X,options)

% WHAT IS THE MOST EFFICIENT WAY OF DOING THIS JOB?
% force.m - Build a 2D projection using Force Scheme.
%
% y = force(x, options);
% 
%   x is a matrix N x D containing the data organized by rows. N is the
%   number of instances and D is the dimension of the data
%   OR
%   x is the distance matrix of the data
%   options:
%     options.data_type must be 'data' or 'dmat', indicating that x is the
%       data matrix or the distance matrix
%     options.iter is the number of iterations (optional, default is 50)
%     options.frac is the fraction of delta (optional, default is 8)
%
%   y is a matrix N x 2 containing the projected data in R^2.

tol = 1e-6;

if ~isfield(options,'data_type')  
    options.data_type = 'data';
end

if ~isfield(options,'iter')  
    iter = 50;
else
    iter = options.iter;
end

if ~isfield(options,'frac')
    fraction = 8;
else
    fraction = options.frac;
end

N = size(X,1);

% inicializacao
Y(:,1) = rand(N,1);
Y(:,2) = rand(N,1);

%--------------%
% Force Scheme %
%--------------%

% distancia em R^n
if strcmp(options.data_type, 'data')  % data matrix
    distRn = dist(X');
elseif strcmp(options.data_type,'dmat')
    distRn = X;
else
	disp('Force ERROR: Undefined data type');
end

idx = randperm(N);

for k = 1:iter % iteracoes

  % para cada x'
  for i = 1:N
    inst1 = idx(i);

    % para cada q' ~= x'
    for j = 1:N
      inst2 = idx(j);

      if (inst1 ~= inst2)
        % calcula a direcao v
        v = Y(inst2,:)-Y(inst1,:);
        distR2 = hypot(v(1),v(2));
        if (distR2 < tol)
          distR2 = tol;
        end
        delta = distRn(inst1,inst2) - distR2;
        delta = delta/fraction;
        v = v./distR2;
        % move q' = Y(j,:) na direcao de v por uma fracao delta
        Y(inst2,:) = Y(inst2,:) + delta*v;
      end
    end
  end

end % fim iteracao

