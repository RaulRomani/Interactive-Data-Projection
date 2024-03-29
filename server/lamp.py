# Author: Luis Gustavo Nonato  -- <gnonato@icmc.usp.br>
# License: BSD 3 - Clause License 2018

# This is an implementation of the technique described in:
# http://www.lcad.icmc.usp.br/~nonato/pubs/lamp.pdf

import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.utils.extmath import randomized_svd


epsilon = 1e-7

def plotMatrix(X, tittle):

    # np.random.seed(1) # random seed for consistency, debugging same results every time
    # data = np.random.rand(500,500)

    fig, ax = plt.subplots()
    im = ax.imshow(X)#, cmap=cm.RdBu)


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Create colorbar
    cbar_kw={}

    cbar = ax.figure.colorbar(im, ax=ax,  **cbar_kw)
    cbar.ax.set_ylabel("cbarlabel", rotation=-90, va="bottom")

    ax.set_title(tittle)
    fig.tight_layout()
    plt.show()


class Lamp():
    #center the data (per features)
    def __init__(self, Xdata = None, control_points = None, weights = None, label=False, scale=True, dim = 2):
        '''
          Xdata: N_by_K matrix where N is the number of instances and K is the dimension

          control_points: M_by_(dim + 1) matrix with the coordinates of the control points
                        in the image space. The last column contains the ids of the control
                        points in the Xdata matrix

          weights: N_by_M matrix where weights[i,j] is the weight between Xdata[i] and
                   control_point[j]. If weights=None the inverse of the Euclidean distance is used as weight

          label: when True assumes the last column of Xdata as labels

          scale: apply a transformation in the control points before accomplish the mapping.
                 Produce better projections when the original and control points coordiantes has very different scales.

          dim: dimension of the image space
        '''
        self.data = None
        self.mapped = None
        self.control_points = None
        self.control_points_data = None
        self.weights = weights
        self.dim = dim
        self.labels = None
        self.label = label
        self.scale = scale
        self.data_center = None
        self.control_points_center = None
        self.U = None
        self.S = None
        self.V = None

        #center the original data by features
        if Xdata is not None:
            try:
                if type(Xdata) is not np.ndarray:
                    raise TypeError()

                ##########################
                if label is False:
                    self.center(Xdata, o_c_s='o')
                else:
                    self.center(Xdata[:,:-1], o_c_s='o') 
                    self.labels = Xdata[:,-1].astype(int)

            except (AttributeError, TypeError):
                print('----- LAMP Error constructor-----')
                print('Xdata must be a Numpy Array')
                sys.exit()

        #center the projected control points by features
        if control_points is not None:
            try:
                if type(control_points) is not np.ndarray:
                    raise TypeError()
            except (AttributeError, TypeError):
                print('----- LAMP Error -----')
                print('control_points must be a Numpy Array')
                sys.exit()

            ##########################
            self.center(control_points, o_c_s='c')

           


#########
    def fit(self, Xdata=None):
        ''' decide if the mapping should be trigger with or without control points'''
        if Xdata is not None:
            try:
                ############  
                if type(Xdata) is not np.ndarray:
                    raise TypeError()

                if self.data is not None:
                    if self.label is True:
                        self.center(Xdata[:,:-1],o_c_s='s')
                        self.labels = Xdata[:,-1].astype(int)
                    else:
                        self.center(Xdata,o_c_s='s')
                else:

                    if self.label is True:
                        self.center(Xdata[:,:-1],o_c_s='o')
                        self.labels = Xdata[:,-1].astype(int)
                    else:
                        self.center(Xdata,o_c_s='o')
            except (AttributeError, TypeError):
                print('----- LAMP Error fit-----')
                print('Type Error: Xdata must be a Numpy Array')
                sys.exit()
        else:
            try:

                if self.data is None:
                    raise ValueError()

            except ValueError:
                print('----- LAMP Error -----')
                print('No data to map')
                sys.exit()

        if self.control_points is not None:

            ############  

            if (self.scale is True) and (self.S is None):
                print("scaling control points")
                self.scale_control_points()


            self.map()


            if self.scale is True:
                print("unscaling control points")
                self.unscale()

            # the projected points in visual space ("2D")
            print("self.control_points_center: ", self.control_points_center)
            mapped = self.mapped + self.control_points_center
            
            if self.label is True:
                mapped = np.hstack((mapped, self.labels.reshape(self.labels.shape[0],1)))
        else:
            self.control_points_free_mapping()  # To be implemented soon
            mapped = self.mapped


        return(mapped)

######### 
    def map(self):
        ''' mapping using control points '''
        n,k = self.data.shape
        # m   = self.control_points.shape[0]

        # the solution
        self.mapped = np.zeros((n, self.dim))

        ctp_data    = self.control_points_data    # high dim 
        ctp_mapped  = self.control_points[:,:-1]  # low dim 2D


        # print(self.data.shape)
        # print(ctp_data.shape)


        if self.weights is None:
            # print("weights")
            self.weights = cdist(self.data, ctp_data, p=2.) #distance matrix (|self.data|, |ctp_data|) from an instance to every mapped point

            # when a cp meets P, the distance(cp, P) is zero and the inverse of zero is inf, to prevent that we add epsilon
            # inv(distance(cp,P) + epsilon)  = inv(epsilon)
            self.weights = 1.0/(self.weights+epsilon)
            self.weights = 0.0*self.weights + 0.1
            # self.weights = 1.0/(self.weights**2+1)
            # self.weights = np.exp(-(self.weights**2))
            # print(self.weights.shape)

        # normalize weights per data point in high dim
        # for i in range(self.weights.shape[0]):
        #     suma = 0.0
        #     for j in range(self.weights.shape[1]):
        #       suma += self.weights[i][j]
        #     self.weights[i] = self.weights[i]/suma



        #distances of the instance 0 with all control points  (1x|ctp_data|)
        # for i in range(15):
        #     print(self.weights[i])

        # print(self.weights[0].T)
        # alpha0 = np.sum(self.weights[0])
        # print("weights 0:")
        # print(alpha0)

        # plotMatrix(X = self.weights, tittle = "weights" )

        # print(ctp_data.T.shape)
        # print(self.weights[0].T.shape)

        for i in range(n):   #FOR EACH INSTANCE

            #By taking partial derivatives with respect to t=0
            alpha   = np.sum(self.weights[i])
            x_tilde = np.dot(ctp_data.T  , self.weights[i].T)/alpha
            y_tilde = np.dot(ctp_mapped.T, self.weights[i].T)/alpha

            # The minimization problem can be rewritten as:
            x_hat = ctp_data - x_tilde
            y_hat = ctp_mapped - y_tilde

            # building the A and B matrix
            D = np.diag(np.sqrt(self.weights[i]))
            A = np.dot(D, x_hat)
            B = np.dot(D, y_hat)

            U,s,V = randomized_svd(np.dot(A.T,B), n_components=2, random_state=None)

            M = np.dot(U,V)
            self.mapped[i] = np.dot((self.data[i] - x_tilde),M) + y_tilde

#########
    def center(self, X, o_c_s='o'):
        '''o_c_s distriminate between original data (o), control points (c), and streaming data (s)'''
        if o_c_s == 'o':
            print(X.shape)
            Xmean = np.average(X, axis=0) #axis=0 , average the all the row vectors (feature vectors)
            print("original")
            print(Xmean.shape)
            print(Xmean)
            self.data_center = Xmean
            self.data = np.subtract(X, Xmean)  # centered in the origin

        if o_c_s == 'c':
            # in this case X is the control point matrix
            ctp_ids = X[:,-1].astype(int)  
            self.control_points_data = self.data[ctp_ids]

            Xmean = np.average(X[:,0:2], axis=0)  #2D points
            print("control_points")
            print(Xmean.shape)
            print(Xmean)
            self.control_points_center = Xmean


            centered = np.subtract(X[:,0:2],Xmean)
            self.control_points = np.hstack((centered,X[:,-1].reshape(-1,1))) #control points centered in the origin

        if o_c_s == 's':
            self.data = np.subtract(X,self.data_center)

#########
    # scale the projected control points by the singular values of the original control points (high dim)
    def scale_control_points(self):
        ctp_ids    = self.control_points[:,-1].astype(int)
        ctp_data   = self.data[ctp_ids]           # high dim : X
        ctp_mapped = self.control_points[:,:-1]   # low dim  : Y

        Uo, So, Vo = randomized_svd(ctp_data.T   , n_components=2, random_state=None)
        Uc, Sc, Vc = randomized_svd(ctp_mapped.T , n_components=2, random_state=None)
        DSo        = np.diag(So) #singular values of ctp_data

        # left singular vector x singular values x  right singular vector

        # print(self.control_points[:,:-1])
        self.control_points[:,:-1] = np.dot(np.dot(Uc, DSo), Vc).T
        # print(self.control_points[:,:-1])
        self.U  = Uc
        self.S  = np.diag(Sc)
        self.V  = Vc
        self.So = np.diag(So)

#########
    def unscale(self):
        proj = np.dot(self.U.T, self.mapped.T)

        print("self.U.T.shape: ", self.U.T.shape)
        print("self.mapped.T.shape: ", self.mapped.T.shape)
        print("proj.shape: ", proj.shape)

        Sinv = np.zeros((2,2))
        Sinv[0,0] = 1.0/self.So[0,0]
        Sinv[1,1] = 1.0/self.So[1,1]

        proj_unscaled = np.dot(Sinv, proj)  ## equals to Vc unscaled

        self.mapped = np.dot(self.U, np.dot(self.S, proj_unscaled)).T

#########
    def control_points_free_mapping(self):
        ''' Mapping without control points '''
        print("control_points_free_mapping")
        def stress(p,pt,d):
            cost = 0
            for i in range(3):
                cost += ((p[0]-pt[i,0])**2+(p[1]-pt[i,1])**2-d[i])**2
            return(cost)

        max_dist = 1e8
        n = self.data.shape[0]
        knn = 5
        self.mapped = np.zeros((n,self.dim))

        D = cdist(self.data,self.data,p=2.)

        # finding the 3 first points that give rise to the inicial triangle

        ### starting with the farthest 2 points
        idx_flat = np.argmax(D)
        idx_i = idx_flat//n
        idx_j = idx_flat - n*idx_i

        ### starting with 2 random
        #idx_i = int(np.random.randint(low=0,high=n,size=1))
        #idx_j = int(np.random.randint(low=0,high=n,size=1))
        #while (idx_j == idx_i):
        #    idx_j = int(np.random.randint(low=0,high=n,size=1))

        d01 = D[idx_i,idx_j]
        d01 += epsilon
        D[idx_i,idx_j] = 0
        D[idx_j,idx_i] = 0
        processed_points = [idx_i,idx_j]

        D_processed = D[processed_points]
        idx_flat = np.argmax(D_processed)
        idx_i = idx_flat//n
        idx_j = idx_flat - n*idx_i
        d02 = D[processed_points[0],idx_j]**2 + epsilon
        d12 = D[processed_points[1],idx_j]**2 + epsilon
        D[processed_points,idx_j] = 0
        D[idx_j,processed_points] = 0
        processed_points.append(idx_j)

        # computing the initial triangle
        x = (-d12+d02+d01**2)/(2.0*d01)
        y = np.sqrt(d02-x**2)
        ltemp = [(0,0),(d01,0),(x,y)]
        proj_tri = np.asarray(ltemp)
        self.mapped[processed_points] = proj_tri[:]


        # computing k more initial points
        for i in range(3,knn):
            D_processed = D[processed_points]
            idx_flat = np.argmax(D_processed)
            idx_i = idx_flat//n
            idx_j = idx_flat - n*idx_i
            d = np.asarray([D[processed_points[0],idx_j]**2+epsilon, D[processed_points[1],idx_j]**2+epsilon, D[processed_points[2],idx_j]**2+epsilon])
            D[processed_points,idx_j] = 0
            D[idx_j,processed_points] = 0
            processed_points.append(idx_j)

            res = minimize(stress, np.asarray([0,0]),args=(proj_tri,d))
            self.mapped[idx_j] = res.x

        D = D + max_dist*np.identity(n)
        for idx_j in processed_points:
            D[processed_points,idx_j] = max_dist
            D[idx_j,processed_points] = max_dist

        # mapping the data set
        weights = np.zeros((knn,))
        for i in range(knn,n):
            D_processed = D[processed_points]
            idx_flat = np.argmin(D_processed)
            idx_i = idx_flat//n
            idx_j = idx_flat - n*idx_i

            neighbors_ids = np.argpartition(D[idx_j,processed_points],knn-1)[:knn]
            ctp_ids = [processed_points[j] for j in neighbors_ids]
            weights[:] = D[idx_j,ctp_ids]
            weights = 1.0/(weights+epsilon)
            ctp_data = self.data[ctp_ids]
            ctp_mapped = self.mapped[ctp_ids]

            alpha = np.sum(weights)
            x_tilde = np.dot(ctp_data.T,weights.T)/alpha
            y_tilde = np.dot(ctp_mapped.T,weights.T)/alpha
            x_hat = ctp_data - x_tilde
            y_hat = ctp_mapped - y_tilde
            S = np.diag(np.sqrt(weights))
            A = np.dot(S,x_hat)
            B = np.dot(S,y_hat)
            U,s,V = randomized_svd(np.dot(A.T,B), n_components=2, random_state=None)
            M = np.dot(U,V)
            self.mapped[idx_j] = np.dot((self.data[idx_j] - x_tilde),M)+y_tilde

            D[processed_points,idx_j] = max_dist
            D[idx_j,processed_points] = max_dist
            processed_points.append(idx_j)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # generating data by drawning points from 2 gaussian in R^20
    D = np.random.multivariate_normal(np.zeros((20,)), 0.1*np.identity(20,dtype=float), 500)
    D = np.vstack((D,np.random.multivariate_normal(np.ones((20,)), 0.1*np.identity(20,dtype=float), 500)))

    # defining control points by projecting orthogonally a random subset of D in R^2
    sample_ids = np.random.randint(low=0, high=1000, size=(100,))  # getting 100 random points in D as control points
    control_points = D[sample_ids,0:2]                           # coordinates of control points
    control_points = np.hstack((control_points, sample_ids.reshape(-1,1)))  # including ids of original points as the last column of control_points
    lamp = Lamp(Xdata=D,control_points=control_points)
    proj = lamp.fit()
    plt.scatter(proj[:,0],proj[:,1])
    plt.scatter(control_points[:,0],control_points[:,1],c='r',s=2)
    plt.show()


