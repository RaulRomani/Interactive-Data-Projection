import random
from flask import Flask, render_template,  request, jsonify
import simplejson as json
import pandas as pd
from lamp import Lamp
from util import fitLamp
from util import getMetricsForAllProjections
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt


from plmp.projection import projection
from plmp.force import force
from plmp.plmp import PLMP
import torch

from deep_projection.model import DREstimator, myMSELoss

import sys
import os

try:
    # Python 2
    xrange
except NameError:
    # Python 3, xrange is now named range
    xrange = range

app = Flask(__name__, static_folder='../static/dist', template_folder='../static')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/controlPoints', methods=['post']) # take note of this decorator syntax, it's a common pattern
def getControlPoints():
  data = json.loads(request.data.decode('utf-8'))
  # print(data)
  dataset_name = data['datasetName']

  print(dataset_name)


  # ########### READ DATA ###################
  labels = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_labels.csv', header=None).values.reshape((-1))
  df_encoding2 = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_prep_encoding2.csv', header=None)  

  X = df_encoding2.values

  # ########### SELECT CONTROL POINTS ###################

  sample_size =  int(np.ceil(np.sqrt(X.shape[0])))

  ctp_ids = np.random.randint(0, high=X.shape[0], size=(sample_size,))  #list of indexes

  ##### projecting control points with MDS #####
  # ctp_mds = MDS(n_components=2)
  # ctp_samples = ctp_mds.fit_transform(X[ctp_ids])# - np.average(x[ctp_ids])) #center in the origin

  ##### projecting control points with Force #####
  f = force.Force(X[ctp_ids], [])
  f.project()
  ctp_samples = f.get_projection()

  # including ids of control points as the last column of the projected control points
  proj_ctp = np.hstack((ctp_samples, ctp_ids.reshape(sample_size, 1)))
  np.savetxt("../../datasets/" + dataset_name + '/'+ dataset_name + "_prep_encoding2_proj_ctp.csv", proj_ctp, delimiter=",")

  # projection = { 'x': x_project[:,0].tolist(), 'y': x_project[:,1].tolist() }
  control_points = { 'x': ctp_samples[:,0].tolist(), 'y': ctp_samples[:,1].tolist(), 'labels': labels[ctp_ids].tolist() }

  # print(control_points)

  # X_embedded   = self.fitLamp(df_encoding2.values)
  #select random points like lamp

  #return: projected control points
  return json.dumps(control_points)

@app.route('/projectUsingEnsemble', methods=['post']) # take note of this decorator syntax, it's a common pattern
def projectUsingEnsemble():
  data = json.loads(request.data.decode('utf-8'))
  # print(data)
  # print(data)# data['values'] controlPoints datasetName

  dataset_name = data['datasetName']
  form_values  = data['values']

  if 'methods' in form_values:
    print("methods are selected")
  else:
    print("methods are not selected")

  labels         = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_labels.csv', header=None).values.reshape((-1,1))
  df_encoding2   = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_prep_encoding2.csv', header=None)  
  control_points = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_prep_encoding2_proj_ctp.csv', header=None)  

  # print("\n\n\ncontrol_points.shape")
  # print(control_points.shape)

  # cp_x           = np.array(data['controlPoints']['x']).reshape( (-1,1) )
  # cp_y           = np.array(data['controlPoints']['y']).reshape( (-1,1) )
  # # cp_labels      = np.array(data['controlPoints']['labels']).reshape( (-1,1) )
  # control_points = np.hstack((cp_x, cp_y))

  # methods = {'LAMP':True}

  data = projectEnsemble(form_values, df_encoding2.values, labels, control_points.values, dataset_name )
  # print(projection.shape)

  # projection_json = { 'x': projection[:,0].tolist(), 'y': projection[:,1].tolist(), 'labels': labels.reshape(-1).tolist() }

  return json.dumps(data) 
    
def projectEnsemble(form_values, X, labels, control_points, dataset_name ):
  #Normalize weights
  weight1 = float(form_values['LAMP_weight'])/ (float(form_values['LAMP_weight']) + float(form_values['LSP_weight']) + float(form_values['PLMP_weight']))  
  weight2 = float(form_values['LSP_weight'])/ (float(form_values['LAMP_weight']) + float(form_values['LSP_weight']) + float(form_values['PLMP_weight']))  
  weight3 = float(form_values['PLMP_weight'])/ (float(form_values['LAMP_weight']) + float(form_values['LSP_weight']) + float(form_values['PLMP_weight']))  

  form_values['LAMP_weight'] = weight1
  form_values['LSP_weight']  = weight2
  form_values['PLMP_weight'] = weight3

  projection_json = {}
  ensemble = np.zeros( (X.shape[0], 2) )

  if float(form_values['LAMP_weight']) != 0.0:
    lamp_proj   = Lamp(Xdata = X, control_points = control_points[:,:2], label=False)
    x_projected = lamp_proj.fit()
    projection_json['LAMP'] = { 'x': x_projected[:,0].tolist(), 'y': x_projected[:,1].tolist(), 'labels': labels.reshape(-1).tolist() }
    ensemble += x_projected*float(form_values['LAMP_weight'])


  if float(form_values['LSP_weight']) != 0.0:
    os.system("./runLSP.m " + dataset_name)
    x_projected = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_projected_octave.csv', header=None).values
    projection_json['LSP'] = { 'x': x_projected[:,0].tolist(), 'y': x_projected[:,1].tolist(), 'labels': labels.reshape(-1).tolist() }
    ensemble += x_projected*float(form_values['LSP_weight'])


  if float(form_values['PLMP_weight']) != 0.0:
    plmp = PLMP(X, labels.reshape(-1), control_points[:,2].reshape(-1).astype(int), control_points[:,:2])
    plmp.project()
    x_projected = plmp.get_projection()
    projection_json['PLMP'] = { 'x': x_projected[:,0].tolist(), 'y': x_projected[:,1].tolist(), 'labels': labels.reshape(-1).tolist() }
    ensemble += x_projected*float(form_values['PLMP_weight'])

  projection_json['Ensemble'] = { 'x': ensemble[:,0].tolist(), 'y': ensemble[:,1].tolist(), 'labels': labels.reshape(-1).tolist() }

  print("evaluating metrics...")
  metrics = getMetricsForAllProjections(X, projection_json, labels, 8)
  print("evaluating metrics finished")

  data = {'projections': projection_json, 'metrics': metrics  } #{'NP': [], 'T': [], 'NH': []}

  return data

@app.route('/estimateEnsemble', methods=['post']) # take note of this decorator syntax, it's a common pattern
def estimateEnsemble():
  print("Estimating ensemble")
  data = json.loads(request.data.decode('utf-8'))

  # data['datasetName']
  # data['labels']
  # print(np.array(data['Ensemble']['x']).shape)
  # print(np.array(data['Ensemble']['y']).reshape( (-1,1) ).shape)
  Y_train = np.hstack( (np.array(data['Ensemble']['x']).reshape( (-1,1) ), np.array(data['Ensemble']['y']).reshape( (-1,1) )) )
  print("Y_train")
  print(Y_train.shape)

  dataset_name = data['datasetName']
  df_encoding2 = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_prep_encoding2.csv', header=None)  
  X_train      = df_encoding2.values
  num_features = X_train.shape[1]

  # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  device  = torch.device("cpu")
  model   = DREstimator(num_features=num_features).to(device)
  cost_fn = torch.nn.MSELoss(size_average=False)
  cost_fn = myMSELoss

  # TODO: return also the filename of the saved model
  Y_estimate = model.train(X_train, Y_train, cost_fn, 500)

  Y_estimate_json = { 'x': Y_estimate[:,0].tolist(), 'y': Y_estimate[:,1].tolist(), 'labels': data['Ensemble']['labels'] }

  return json.dumps(Y_estimate_json) 

if __name__ == '__main__':
  app.run(debug=True)
  # app.run(host="0.0.0.0", port="5000", debug=False)
