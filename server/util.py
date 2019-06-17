
import numpy as np
from scipy.spatial.distance import cdist


try:
  # Python 2
  xrange
except NameError:
  # Python 3, xrange is now named range
  xrange = range


def fitLamp(self, X):
  sample_size =  int(np.sqrt(X.shape[0]))
  ctp_ids = np.random.randint(0, high=X.shape[0], size=(sample_size,))  #list of indexes

  ##### projecting control points with MDS #####
  ctp_mds = MDS(n_components=2)
  ctp_samples = ctp_mds.fit_transform(X[ctp_ids])# - np.average(x[ctp_ids])) #center in the origin

  # including ids of control points as the last column of the projected control points
  ctp_samples = np.hstack((ctp_samples, ctp_ids.reshape(sample_size, 1)))

  # # including labels as the last column
  # data = np.hstack((X, labels.reshape(labels.shape[0], 1)))

  # # ##### using Lamp
  # lamp_proj = Lamp(Xdata = data, control_points = ctp_samples, label=True)


  # including labels as the last column
  data = X

  # ##### using Lamp
  lamp_proj = Lamp(Xdata = data, control_points = ctp_samples, label=False)
  
  return lamp_proj.fit()



# global metric
# A value of zero represent a perfect projection
def stress(X,Y):

    #Normalize each feature mean = 0 and std = 1
    #Normalize each instance to have length 1 by dividing by the norm2


    D = cdist(X, X, p=2.) 
    d = cdist(Y, Y, p=2.)

    D = MinMaxScaler().fit_transform(D)
    d = MinMaxScaler().fit_transform(d)
    
    # D_sum = 0.0
    # d_sum = 0.0
    # for i in xrange(1, Y.shape[0]): #rows
    #     for j in xrange(i):         #columns
    #         D_sum += (D[i][j]+d[i][j])/2.0  # original space - projected space
    #         # d_sum += d[i][j]

    upper_sum = 0.0
    lower_sum = 0.0
    for i in xrange(1, Y.shape[0]): #rows
        for j in xrange(i):         #columns
            upper_sum += pow( D[i][j] - d[i][j]  , 2) # original space - projected space
            lower_sum += pow( D[i][j], 2)

    # print(upper_sum)
    # print(lower_sum)
        
    return np.sqrt(upper_sum/lower_sum)

# local metric
# A value of 1 represent a perfect projection
# measures the trustworthiness of the projection neighborhoods
def trustworthiness(X, X_embedded, n_neighbors=5, precomputed=False):
    """Expresses to what extent the local structure is retained.
    The trustworthiness is within [0, 1]. It is defined as

    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.
    """
    if precomputed:
        dist_X = X
    else:
        dist_X = cdist(X, X, p=2.)
    dist_X_embedded = cdist(X_embedded, X_embedded, p=2.)
    
    
    
    ind_X          = np.argsort(dist_X         , axis=1)
    ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1]

    n_samples = X.shape[0]
    
    t = 0.0
    ranks = np.zeros(n_neighbors)
    for i in range(n_samples):
        for j in range(n_neighbors):
            ranks[j] = np.where(ind_X[i] == ind_X_embedded[i, j])[0][0]
        ranks -= n_neighbors
        t += np.sum(ranks[ranks > 0])

        
    t = 1.0 - t * (2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
    return t # t: measure of trustworthiness

# local metric
# A value of 1 represent a perfect projection
# 1 when there is not missing and false neighbors in the kNNs of the projected points
def NeighborhoodPreservation(X, X_embedded, n_neighbors=5, precomputed=False):

    if precomputed:
        dist_X = X
    else:
        dist_X = cdist(X, X, p=2.)
    dist_X_embedded = cdist(X_embedded, X_embedded, p=2.)
    
    #select the kNN for each instance
    ind_X          = np.argsort(dist_X         , axis=1)[:, 1:n_neighbors + 1]
    ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1] 

    NP = 0.0
    print(len(np.setdiff1d(ind_X[0], ind_X_embedded[0])) )
    print(len(np.setdiff1d(ind_X[1], ind_X_embedded[1])) )
    print(ind_X.shape)
    print(ind_X_embedded.shape)
    # could be done in parallel
    for i in xrange(X.shape[0]): # for all the examples
        NP += n_neighbors - len(np.setdiff1d(ind_X[i], ind_X_embedded[i])) 
    NP = NP / float(n_neighbors*X.shape[0])

    return NP

def NeighborhoodHit(X_embedded, C, n_neighbors=5):

    dist_X_embedded = cdist(X_embedded, X_embedded, p=2.)
    
    #select the kNN for each instance
    ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1] 
    m = X_embedded.shape[0]

    def ratio(x, kNN): # indices
        # if the class of the KNN belongs to the class of the point at evaluation
        same_class = len(np.where(C[kNN] == C[x])[0])
        return same_class

    NH = 0.0
    for x in xrange(m): # for all the examples
        NH += ratio(x, ind_X_embedded[x])
    NH = NH / (float(m) *float(n_neighbors) )
    return NH


def trustworthiness_(ind_X, ind_X_embedded, n_neighbors=5):

    n_samples = ind_X.shape[0]
    
    t = 0.0
    ranks = np.zeros(n_neighbors)
    for i in range(n_samples):
        for j in range(n_neighbors):
            ranks[j] = np.where(ind_X[i] == ind_X_embedded[i, j])[0][0]
        ranks -= n_neighbors
        t += np.sum(ranks[ranks > 0])

        
    t = 1.0 - t * (2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
    return t # t: measure of trustworthiness


def NeighborhoodPreservation_(ind_X, ind_X_embedded, n_neighbors=5):

    NP = 0.0
    for i in xrange(ind_X.shape[0]): # for all the examples
        NP += n_neighbors - len(np.setdiff1d(ind_X[i], ind_X_embedded[i])) 
    NP = NP / float(n_neighbors*ind_X.shape[0])
    print(NP)
    return NP

def NeighborhoodHit_(ind_X_embedded, C, n_neighbors=5):

    # dist_X_embedded = cdist(X_embedded, X_embedded, p=2.)
    
    # #select the kNN for each instance
    # ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1] 

    m = ind_X_embedded.shape[0]


    def ratio(x, kNN): # indices
      # if the class of the KNN belongs to the class of the point at evaluation
      same_class = len(np.where(C[kNN] == C[x])[0])
      return same_class

    NH = 0.0
    for x in xrange(m): # for all the examples
        NH += ratio(x, ind_X_embedded[x])
    NH = NH / float(m*n_neighbors)
    # print(NH)
    return NH

# evaluate metrics for many projections
def getMetricsForAllProjections(X, projections, labels, n_executions):

  metrics = {'NP': [], 'T': [], 'NH': []}

  if 'LAMP' in projections:
    LAMP_projected = np.hstack( (np.array(projections['LAMP']['x']).reshape((-1,1)), np.array(projections['LAMP']['y']).reshape((-1,1))) ) 
    trustworthiness, NeighborhoodPreservation, NeighborhoodHit = evaluateMetrics(X, LAMP_projected, labels, n_executions)

    # metrics['T'].append({'methodName': 'LAMP', 'values': trustworthiness})
    metrics['NP'].append({'methodName': 'LAMP', 'values': NeighborhoodPreservation})
    metrics['NH'].append({'methodName': 'LAMP', 'values': NeighborhoodHit})

  if 'LSP' in projections:
    LSP_projected = np.hstack( (np.array(projections['LSP']['x']).reshape((-1,1)), np.array(projections['LSP']['y']).reshape((-1,1))) ) 
    trustworthiness, NeighborhoodPreservation, NeighborhoodHit = evaluateMetrics(X, LSP_projected, labels, n_executions)
    # metrics['T'].append({'methodName': 'LSP', 'values': trustworthiness})
    metrics['NP'].append({'methodName': 'LSP', 'values': NeighborhoodPreservation})
    metrics['NH'].append({'methodName': 'LSP', 'values': NeighborhoodHit})

  if 'PLMP' in projections:
    PLMP_projected = np.hstack( (np.array(projections['PLMP']['x']).reshape((-1,1)), np.array(projections['PLMP']['y']).reshape((-1,1))) ) 
    trustworthiness, NeighborhoodPreservation, NeighborhoodHit = evaluateMetrics(X, PLMP_projected, labels, n_executions)
    # metrics['T'].append({'methodName': 'PLMP', 'values': trustworthiness})
    metrics['NP'].append({'methodName': 'PLMP', 'values': NeighborhoodPreservation})
    metrics['NH'].append({'methodName': 'PLMP', 'values': NeighborhoodHit})

  if 'Ensemble' in projections:
    Ensemble_projected = np.hstack( (np.array(projections['Ensemble']['x']).reshape((-1,1)), np.array(projections['Ensemble']['y']).reshape((-1,1))) ) 
    trustworthiness, NeighborhoodPreservation, NeighborhoodHit = evaluateMetrics(X, Ensemble_projected, labels, n_executions)
    # metrics['T'].append({'methodName': 'Ensemble', 'values': trustworthiness})
    metrics['NP'].append({'methodName': 'Ensemble', 'values': NeighborhoodPreservation})
    metrics['NH'].append({'methodName': 'Ensemble', 'values': NeighborhoodHit})

  return metrics

def getMetrics(X, X_projection, labels, n_executions):

  metrics = {'NP': {}, 'T': {}, 'NH': {}}
  trustworthiness, NeighborhoodPreservation, NeighborhoodHit = evaluateMetrics(X, X_projection, labels, n_executions)
  # metrics['T'].append({'methodName': 'Ensemble', 'values': trustworthiness})
  metrics['NP'] = {'values': NeighborhoodPreservation}
  metrics['NH'] = {'values': NeighborhoodHit}

  return metrics

# evaluate metrics per projection/method
def evaluateMetrics(X, X_embedded, labels, n_executions):
    m = X.shape[0]
    dist_X          = cdist(X, X, p=2.)
    ind_X           = np.argsort(dist_X         , axis=1)
    dist_X_embedded = cdist(X_embedded, X_embedded, p=2.)

    list_ind_X_embedded = []
    increment = 3
    for i in xrange(1,n_executions+1):
        n_neighbors    = int(m*i*increment/100.0)
        ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1]
        list_ind_X_embedded.append(ind_X_embedded)

    trustworthiness          = []
    NeighborhoodPreservation = []
    NeighborhoodHit          = []
    for i in xrange(1,n_executions+1):
        n_neighbors = int(m*i*increment/100.0)
        # trustworthiness.append(trustworthiness_(ind_X, list_ind_X_embedded[i-1], n_neighbors = n_neighbors))
        NeighborhoodPreservation.append(NeighborhoodPreservation_(ind_X[:, 1:n_neighbors + 1], list_ind_X_embedded[i-1], n_neighbors = n_neighbors))
        NeighborhoodHit.append(NeighborhoodHit_(list_ind_X_embedded[i-1], labels, n_neighbors = n_neighbors ))

    return trustworthiness, NeighborhoodPreservation, NeighborhoodHit




def testMetricsElapsedTime():
  #TODO: Make stress test of NH with fake data or real?
  import pandas as pd
  # dataset_name = "Caltech"
  # dataset_name = "Iris"
  dataset_name = "Synthetic4Classes"
  X           = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_prep_encoding2.csv', header=None).values
  labels      = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_labels.csv', header=None).values.reshape((-1))
  X_projected = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_projected_octave.csv', header=None).values

  print(X.shape)
  print(labels.shape)
  print(X_projected.shape)

  projections = {'LSP' : {'x' : X_projected[:,0].tolist(),
                          'y' : X_projected[:,1].tolist()
                         } 
                }

  getMetricsForAllProjections(X= X, projections = projections, labels= labels, n_executions= 4)

  import time
  # start = time.time()
  # print(NeighborhoodHit(X_projected, labels, int(X.shape[0]*0.50)))
  # end   = time.time()
  # print("elapsed time:", end - start)

  # start = time.time()
  # print(trustworthiness(X, X_projected, int(X.shape[0]*0.24)))
  # end   = time.time()
  # print("elapsed time:", end - start)

  print("")
  # start = time.time()

  # print(NeighborhoodHit(X_projected, labels, int(X.shape[0]*0.03)))
  # print(NeighborhoodHit(X_projected, labels, int(X.shape[0]*0.06)))
  # print(NeighborhoodHit(X_projected, labels, int(X.shape[0]*0.09)))
  # print(NeighborhoodHit(X_projected, labels, int(X.shape[0]*0.12)))

  print(NeighborhoodPreservation(X, X_projected, int(X.shape[0]*0.03)))
  print(NeighborhoodPreservation(X, X_projected, int(X.shape[0]*0.06)))
  print(NeighborhoodPreservation(X, X_projected, int(X.shape[0]*0.09)))
  print(NeighborhoodPreservation(X, X_projected, int(X.shape[0]*0.12)))
  # end = time.time()
  # print("elapsed time:", end - start)

if __name__ == '__main__':
  testMetricsElapsedTime()
