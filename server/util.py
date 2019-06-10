
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

    def matchedPoints(kNN1, kNN2):
        n_matchedPoints = 0
        for i in xrange(n_neighbors):
            for j in xrange(n_neighbors):
                if kNN1[i] == kNN2[j]:
                    n_matchedPoints += 1 
                    break
        return n_matchedPoints

    NP = 0.0
    for i in xrange(X.shape[0]): # for all the examples
        NP += matchedPoints(ind_X[i], ind_X_embedded[i])
    NP = NP / float(n_neighbors*X.shape[0])

    return NP

# local metric
# A value of 1 represent a perfect projection
# 1 when the classes are clearly separated
def NeighborhoodHit(X_embedded, C, n_neighbors=5):

    dist_X_embedded = cdist(X_embedded, X_embedded, p=2.)
    
    #select the kNN for each instance
    ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1] 
    m = X_embedded.shape[0]

    def ratio(x, kNN): # indices
        same_class = 0.0
        for i in xrange(n_neighbors):
            if int(C[x]) == int(C[kNN[i]]):
                same_class += 1.0
        return same_class/float(n_neighbors)

    NH = 0.0
    for x in xrange(m): # for all the examples
        NH += ratio(x, ind_X_embedded[x])
    NH = NH / float(m)
    return NH


# local metric
# A value of 1 represent a perfect projection
# measures the trustworthiness of the projection neighborhoods
def trustworthiness_(ind_X, ind_X_embedded, n_neighbors=5, precomputed=False):
    """Expresses to what extent the local structure is retained.
    The trustworthiness is within [0, 1]. It is defined as

    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.
    """
    # if precomputed:
    #     dist_X = X
    # else:
    #     dist_X = cdist(X, X, p=2.)
    # dist_X_embedded = cdist(X_embedded, X_embedded, p=2.)
    
    
    
    # ind_X          = np.argsort(dist_X         , axis=1)
    # ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1]

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

# local metric
# A value of 1 represent a perfect projection
# 1 when there is not missing and false neighbors in the kNNs of the projected points
def NeighborhoodPreservation_(ind_X, ind_X_embedded, n_neighbors=5, precomputed=False):

    # if precomputed:
    #     dist_X = X
    # else:
    #     dist_X = cdist(X, X, p=2.)
    # dist_X_embedded = cdist(X_embedded, X_embedded, p=2.)
    
    # #select the kNN for each instance
    # ind_X          = np.argsort(dist_X         , axis=1)[:, 1:n_neighbors + 1]
    # ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1] 

    def matchedPoints(kNN1, kNN2):
        n_matchedPoints = 0
        for i in xrange(n_neighbors):
            for j in xrange(n_neighbors):
                if kNN1[i] == kNN2[j]:
                    n_matchedPoints += 1 
                    break
        return n_matchedPoints

    NP = 0.0
    for i in xrange(ind_X.shape[0]): # for all the examples
        NP += matchedPoints(ind_X[i], ind_X_embedded[i])
    NP = NP / float(n_neighbors*ind_X.shape[0])

    return NP

# local metric
# A value of 1 represent a perfect projection
# 1 when the classes are clearly separated
def NeighborhoodHit_(ind_X_embedded, C, n_neighbors=5):

    # #select the kNN for each instance
    # ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1] 

    m = ind_X_embedded.shape[0]

    def ratio(x, kNN): # indices
        same_class = 0.0
        for i in xrange(n_neighbors):
            if int(C[x]) == int(C[kNN[i]]):
                same_class += 1.0
        return same_class/float(n_neighbors)

    NH = 0.0
    for x in xrange(m): # for all the examples
        NH += ratio(x, ind_X_embedded[x])
    NH = NH / float(m)
    return NH


# evaluate metrics for many projections
def getMetricsForAllProjections(X, projections, labels, n_executions):

  metrics = {'NP': [], 'T': [], 'NH': []}

  if 'LAMP' in projections:
    LAMP_projected = np.hstack( (np.array(projections['LAMP']['x']).reshape((-1,1)), np.array(projections['LAMP']['y']).reshape((-1,1))) ) 
    trustworthiness, NeighborhoodPreservation, NeighborhoodHit = evaluateMetrics(X, LAMP_projected, labels, n_executions)

    metrics['T'].append({'methodName': 'LAMP', 'values': trustworthiness})
    metrics['NP'].append({'methodName': 'LAMP', 'values': NeighborhoodPreservation})
    metrics['NH'].append({'methodName': 'LAMP', 'values': NeighborhoodHit})

  if 'LSP' in projections:
    LSP_projected = np.hstack( (np.array(projections['LSP']['x']).reshape((-1,1)), np.array(projections['LSP']['y']).reshape((-1,1))) ) 
    trustworthiness, NeighborhoodPreservation, NeighborhoodHit = evaluateMetrics(X, LSP_projected, labels, n_executions)
    metrics['T'].append({'methodName': 'LSP', 'values': trustworthiness})
    metrics['NP'].append({'methodName': 'LSP', 'values': NeighborhoodPreservation})
    metrics['NH'].append({'methodName': 'LSP', 'values': NeighborhoodHit})

  if 'PLMP' in projections:
    PLMP_projected = np.hstack( (np.array(projections['PLMP']['x']).reshape((-1,1)), np.array(projections['PLMP']['y']).reshape((-1,1))) ) 
    trustworthiness, NeighborhoodPreservation, NeighborhoodHit = evaluateMetrics(X, PLMP_projected, labels, n_executions)
    metrics['T'].append({'methodName': 'PLMP', 'values': trustworthiness})
    metrics['NP'].append({'methodName': 'PLMP', 'values': NeighborhoodPreservation})
    metrics['NH'].append({'methodName': 'PLMP', 'values': NeighborhoodHit})

  if 'Ensemble' in projections:
    Ensemble_projected = np.hstack( (np.array(projections['Ensemble']['x']).reshape((-1,1)), np.array(projections['Ensemble']['y']).reshape((-1,1))) ) 
    trustworthiness, NeighborhoodPreservation, NeighborhoodHit = evaluateMetrics(X, Ensemble_projected, labels, n_executions)
    metrics['T'].append({'methodName': 'Ensemble', 'values': trustworthiness})
    metrics['NP'].append({'methodName': 'Ensemble', 'values': NeighborhoodPreservation})
    metrics['NH'].append({'methodName': 'Ensemble', 'values': NeighborhoodHit})

  return metrics





# evaluate metrics per projection/method
def evaluateMetrics(X, X_embedded, labels, n_executions):
    m = X.shape[0]
    dist_X          = cdist(X, X, p=2.)
    ind_X           = np.argsort(dist_X         , axis=1)
    dist_X_embedded = cdist(X_embedded, X_embedded, p=2.)

    list_ind_X_embedded = []
    for n in xrange(1,n_executions+1):
        n_neighbors    = int(m*n*5/100.0)
        ind_X_embedded = np.argsort(dist_X_embedded, axis=1)[:, 1:n_neighbors + 1]
        list_ind_X_embedded.append(ind_X_embedded)

    trustworthiness          = []
    NeighborhoodPreservation = []
    NeighborhoodHit          = []
    for n in xrange(1,n_executions+1):
        trustworthiness.append(trustworthiness_(ind_X, list_ind_X_embedded[n-1], n_neighbors= int(m*n*5/100.0), precomputed=False))
        NeighborhoodPreservation.append(NeighborhoodPreservation_(ind_X, list_ind_X_embedded[n-1], n_neighbors= int(m*n*5/100.0), precomputed=False))
        NeighborhoodHit.append(NeighborhoodHit_(list_ind_X_embedded[n-1], labels, n_neighbors= int(m*n*5/100.0)))

    return trustworthiness, NeighborhoodPreservation, NeighborhoodHit



def main():
  #TODO: Make stress test of NH with fake data or real?
  import pandas as pd
  dataset_name = "Synthetic4Classes"
  X           = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_prep_encoding2.csv', header=None).values
  labels      = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_labels.csv', header=None).values.reshape((-1))
  X_projected = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_projected_octave.csv', header=None).values

  print(X.shape)
  print(labels.shape)
  print(X_projected.shape)

  print(NeighborhoodHit(X_projected, labels, int(X.shape[0]*0.5)))
  # print(NeighborhoodPreservation(X, X_projected, int(X.shape[0]*0.5)))
  print(trustworthiness(X, X_projected, int(X.shape[0]*0.5)))

    
if __name__ == '__main__':
    main()
