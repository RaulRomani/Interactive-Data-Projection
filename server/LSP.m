function [proj_points, proj_cp, sc] = LSP(D, control_points, options)
% LSP Projects a distance matrix D using Least-Squares Projection.
%
% Description:
%   D is a distance matrix
%   options.k defines number of neighbors on Laplacian matrix construction
%   options.nc defines number of control points as a factor of dataset size
%   options.proj_type project CP with classical MDS(='mds') or 
%                                     ForceScheme projection(=default)
%   options.debug show debug information   
%
% Examples:
%   proj_points = LSP(D) 
%       proj_points are projected points onto visual space with defaul values
%       for k(=10) and nc (=the square root of dataset size).
%
%   options.k = 50; options.nc = 0.1;
%   proj_points = LSP(D,options) 
%       proj_points are projected points onto visual space with k=10 and 
%       nc = 10% of dataset size.
%
%   [proj_points, proj_cp] = LSP(D)
%       proj_points are projected points onto visual space with defaul values
%       for k(=10) and nc (=the square root of dataset size).
%       proj_cp are control points projections onto visual space    
%
% Authors:
%   Danilo Mota, Douglas Cedrim
%
% Last modification:
%   March,2014

%% Definitions
pkg load statistics;


n = size(D,1);
if ~isfield(options,'k')
    k = 10;
else
    k = options.k;
end

if ~isfield(options,'nc')
    nc = ceil(sqrt(n));
else
    #nc = ceil(options.nc*n);
    nc = options.nc;
end

if ~isfield(options,'proj_type')
    proj_type = 'force';    
else
	proj_type = options.proj_type;
end

fprintf('Encontrando pontos de controle.\n');
disp(['k: ' num2str(k) ' nc: ' num2str(nc)]);

%% Laplacian construction
L = zeros(n);

#pick the k nearest neighbors
for i=1:n
    L(i, i) = 1;
    [~, idx_full] = sort(D(i,:),'ascend');
    
    idx = idx_full(2:k+1);                              # pick the k smallest distances except the minimum    
    L(i,idx) = -1/k; 
end

[~, idx_full] = sort(D(1,:),'ascend');
idx = idx_full(2:k+1);
[~, idx_full] = sort(D(2,:),'ascend');
idx = idx_full(2:k+1);


if isfield(options,'debug')
    % check rows sum
    if (norm(sum(L,2)) < 1e-12)
        disp('Row sum: OK')
    else
        disp('Row sum: PROBLEM')
    end
    
    % Check matrix img
    spy(L);
    pause(2);
end


%% Control points projection
% Distance matrix of control points



sc = sort(control_points(:,3)' +1,'ascend')
# sc = sort(randsample(n,nc),'ascend');

d = D(sc,sc); #sampling of control points
switch (proj_type)
    case 'force'
        % Force Scheme
        options.data_type = 'dmat';
        proj_cp = force(d,options);
        fprintf('Projetando pontos de controle com a Force.\n');
    case 'mds'
        % MDS
        options = statset('MaxIter',1000);
        proj_cp = mdscale(d,2,'Start','random','Criterion','metricstress','Options',options);
        % proj_cp = mdscale(d,2,'Options',options);
        fprintf('Projetando pontos de controle com MDS.\n');
    otherwise
        disp('Unknown control points projection method.');
end



%% Least-squares system solving
C = zeros(nc,n);
for i=1:nc
    C(i,sc(i)) = 1;
end
A = [L;C];

# fprintf("proj_cp size:")
# size(proj_cp)


% Dimension of visual space
m = 2;
proj_points = zeros(n,m);
for i=1:m 
    b = [zeros(n,1); proj_cp(:,i)];
    # b = [zeros(n,1); control_points(:,i)];
    
    proj_points(:,i) = A\b; #mapped points  
end

fprintf('Feito.\n');