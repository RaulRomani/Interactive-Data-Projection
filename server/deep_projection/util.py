import random
from flask import Flask, render_template,  request, jsonify
import simplejson as json
from flask_sqlalchemy import SQLAlchemy 
from flask_marshmallow import Marshmallow 

from sqlalchemy.sql import text
from sqlalchemy import create_engine, MetaData, Table
import numpy as np

# pip install pymysql flask-sqlalchemy flask-marshmallow marshmallow-sqlalchemy


engine = create_engine('mysql+pymysql://root:romani@localhost/DR_test')
# metadata = MetaData(bind=engine)


def sql2json(results, cursor):
  items = []
  row_headers=[x[0] for x in cursor.description] #this will extract row headers
  for result in results:
    items.append(dict(zip(row_headers,result)))

  return items
def getProjectedData(idExperiment, datasetName, methods):
  conn = engine.raw_connection()

  cursor = conn.cursor()
  query = "SELECT idDataset from dataset where idExperiment = {} and name = '{}'".format(idExperiment, datasetName)
  cursor.execute(query) 
  result = sql2json(cursor.fetchall(), cursor)  # list of tuples "vector of features/values"

  assert (len(result) == 1), "problem: repeated datasetName"
  
  methods_prep = str(methods).replace('[', '').replace(']', '')
  query = "SELECT idMethod from method where idDataset = {} and name in ({})".format(result[0]['idDataset'], methods_prep)
  cursor.execute(query)
  result = sql2json(cursor.fetchall(), cursor)

  assert (len(result) == len(methods)), "problem: more results"

  
  projections = []
  for method in result:
    
    query = "SELECT x+ 0E0,y+ 0E0,class + 0E0 from projection where idMethod = {}".format(method['idMethod'])
    cursor.execute(query)
    result = cursor.fetchall()
    projections.append(np.array(result))

  # print(len(projections))
  # print(projections[0].shape)
  return projections

def projectData(datasetName, method, sample_size):

  import umap
  import pandas as pd
  import matplotlib.pyplot as plt
  from sklearn import preprocessing

  min_max_scaler = preprocessing.MinMaxScaler()

  relative = "../../../"
  labels   = pd.read_csv(relative + "datasets/" + datasetName + '/' + datasetName + "_labels.csv"        , sep=",", header=None).values.reshape(-1)
  X_df     = pd.read_csv(relative + "datasets/" + datasetName + '/' + datasetName + "_prep_encoding2.csv", sep=",", header=None)

  # print(X_df.describe())

  X_normalized = min_max_scaler.fit_transform(X_df.values) # min_max

  ####### SAMPLE DATA 
  # sample_size = 1000
  sample_ids  = np.random.randint(0, high = X_normalized.shape[0], size=(sample_size,))  #list of indexes
  rest_ids = np.setdiff1d(np.arange(0, X_normalized.shape[0]), sample_ids)
  # sample_ids = list(range(sample_size,2*sample_size))

  X_normalized_sample = X_normalized[sample_ids]
  # X_normalized_sample = X_normalized[rest_ids]
  # X_normalized = preprocessing.scale(X_df.values)  # mean std

  # X_df_normalized = pd.DataFrame(X_normalized)
  # print(X_df_normalized.describe())


  if method == 'UMAP - 1HE':
    print("ok")
    X_embedded = umap.UMAP().fit_transform(X_normalized_sample)



    # plt.scatter(X_embedded[:,0], X_embedded[:,1], c = labels[sample_ids])
    # plt.show()

    return X_normalized, labels, X_embedded, sample_ids , rest_ids


if __name__ == '__main__':
  # projectData("Caltech", 'UMAP - 1HE',1000)
  projectData("Iris", 'UMAP - 1HE', 100)