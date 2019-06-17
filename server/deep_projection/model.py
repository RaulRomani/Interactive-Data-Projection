# Low-level implementation using autograd

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

import torch
import torch.nn.functional as F
import pandas as pd
# from  util import getProjectedData, projectData
from  .util import getProjectedData, projectData

##########################
### DATASET
##########################

# n   = 100
# rng = np.random.RandomState(1)
# x   = 10 * rng.rand(n)
# y   = 2 * x - 5 + rng.randn(n)

# np.random.seed(1)
# idx = np.arange(y.shape[0])

# # splitting
# np.random.shuffle(idx)
# X_test, y_test   = x[idx[:25]], y[idx[:25]]    #select 25 examples randomly
# X_train, y_train = x[idx[25:]], y[idx[25:]]  #select 75 examples randomly

# # normalizing
# mu, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
# X_train, X_test = (X_train - mu) / std, (X_test - mu) / std    #normalize the features
# X_train = X_train.reshape(-1,1)




# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class DREstimator(torch.nn.Module):
  def __init__(self, num_features):
    super(DREstimator, self).__init__()
    self.cost_fn = None
    self.layer1 = torch.nn.Linear(num_features, 256)
    self.layer2 = torch.nn.Linear(256, 512)
    self.layer3 = torch.nn.Linear(512, 256)
    self.layer4 = torch.nn.Linear(256, 2)
  def forward(self, x):
    z1    = F.relu(self.layer1(x))
    z2    = F.relu(self.layer2(z1))
    z3    = F.relu(self.layer3(z2))
    # y_hat = F.sigmoid(self.layer4(z3))
    y_hat = self.layer4(z3)

    return y_hat
  def train_model(self, X_train, y_train, cost_fn, nEpochs):
    # optimizer = torch.optim.SGD(self.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device) #.view(-1, 1)
    for epoch in range(nEpochs):
      #### Compute outputs ####
      y_hat = self(X_train_tensor) # forward pass
      # print("X_train_tensor.size(): ", X_train_tensor.size() )
      # print("y_hat.size(): ", y_hat.size() )
      # print("y_train_tensor.size(): ", y_train_tensor.size() )

      # #### Compute gradients ####
      cost = cost_fn(y_hat, y_train_tensor)
      # print(cost)

      optimizer.zero_grad()
      cost.backward()

      #### Update weights ####  
      optimizer.step()

      #### Logging ####      
      y_hat = self(X_train_tensor)
      print('Epoch: %03d' % (epoch + 1), end="")
      print(' | Cost: %.3f' % cost_fn(y_hat, y_train_tensor))
      # print('\nModel parameters:')
    # print('  Weights: %s' % self.linear.weight)
    # print('  Bias: %s' % self.linear.bias)
    y_hat = self.forward(X_train_tensor).cpu().detach().numpy()
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.scatter(y_train[:,0], y_train[:,1], c = labels)
    # ax2.scatter(y_hat[:,0], y_hat[:,1], c = labels)
    # plt.show()
    return y_hat


def myMSELoss(y_hat, y):
  # print("y_hat.size(): ", y_hat.size() )
  # print("y.size(): ", y.size() )

  # return (y_hat-y).pow(2).sum()
  return torch.norm(y - y_hat, 2)**2

#Using save data in mysql
def testProjection():
  # print(X_train.shape)

  # dataset_name = "Synthetic4Classes"
  # dataset_name = "Dermatology"
  # dataset_name = "AustralianCA"

  projections = getProjectedData(2, dataset_name, ['UMAP - 1HE'])

  # print(len(projections))

  Y_train = projections[0][:,:2]
  labels = projections[0][:,2]
  print(Y_train.shape)

  df_encoding2 = pd.read_csv("../../../datasets/" + dataset_name + '/'+ dataset_name + '_prep_encoding2.csv', header=None)  
  X_train = df_encoding2.values
  num_features = X_train.shape[1]

  print(type(X_train))
  print(type(Y_train))

  print(Y_train)


  cost_fn = torch.nn.MSELoss(size_average=False)
  # cost_fn = myMSELoss
  model = DREstimator(num_features=num_features).to(device)
  y_hat = model.train(X_train, Y_train, cost_fn, 500)


  f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
  ax1.scatter(Y_train[:,0], Y_train[:,1], c = labels)
  ax2.scatter(y_hat[:,0], y_hat[:,1], c = labels)
  plt.show()

#Using local datasets and projecting them on hot
def testInference():
  np.random.seed(15)
  # print(X_train.shape)

  # dataset_name = "Synthetic4Classes"
  # dataset_name = "Dermatology"
  # dataset_name = "AustralianCA"
  dataset_name = "Caltech"


  X_normalized, labels, X_sample_embedded, sample_ids , rest_ids = projectData(dataset_name, 'UMAP - 1HE', 300 )

  # print(len(projections))

  Y_train = X_sample_embedded
  X_train = X_normalized[sample_ids]

  print(Y_train.shape)
  print(X_train.shape)


  num_features = X_train.shape[1]


  ######### PROJECTION
  cost_fn = torch.nn.MSELoss(size_average=False)
  # cost_fn = myMSELoss
  model = DREstimator(num_features=num_features).to(device)
  y_hat = model.train_model(X_train, Y_train, cost_fn, 1000)


  ######## PLOT
  f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
  ax1.scatter(Y_train[:,0], Y_train[:,1], c = labels[sample_ids])
  ax1.set_title("umap projection")
  ax2.scatter(y_hat[:,0], y_hat[:,1], c = labels[sample_ids])
  ax2.set_title("DL estimate")
  plt.show()



  ######### SAVING MODEL
  torch.save(model.state_dict(), "./DREstimator_state_dict.pt")



  ######## INFERENCE

  DREstimator_model = DREstimator(num_features=num_features).to(device)
  DREstimator_model.load_state_dict(torch.load("./DREstimator_state_dict.pt"))
  DREstimator_model.eval()

  
  X_train_tensor = torch.tensor(X_normalized[rest_ids], dtype=torch.float32, device=device)
  y_hat_inference = DREstimator_model(X_train_tensor).cpu().detach().numpy()

  #umap projection
  import umap
  X_embedded = umap.UMAP().fit_transform(X_normalized)

  ######## PLOT
  f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
  # ax1.scatter(Y_train[:,0], Y_train[:,1], c = labels[sample_ids])
  ax1.scatter(X_embedded[:,0], X_embedded[:,1], c = labels)
  ax1.set_title("umap projection")
  ax2.scatter(Y_train[:,0], Y_train[:,1], c = labels[sample_ids])
  ax2.set_title("umap sample projection")

  ax3.scatter(y_hat[:,0], y_hat[:,1], c = labels[sample_ids])
  ax3.set_title("umap sample DL estimate")
  # ax3.scatter(y_hat[:,0], y_hat[:,1], c = labels[sample_ids])
  ax4.scatter(y_hat_inference[:,0], y_hat_inference[:,1], c = labels[rest_ids])
  ax4.set_title("umap inference with sample DL")
  plt.show()


if __name__ == '__main__':

  # testProjection()
  testInference()

  




