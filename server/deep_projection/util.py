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
    
# def getProjectedData(idExperiment, dataset, methods)

#   #tables: dataset, method, projection
#   #idDataset = select dataset.idDataset from dataset where idExperiment = 5 and dataset.name = "4 Classes"
#   #idMethods = select idMethod form method where method.idDataset = idDataset and method.name in (methods)
#   # projection1 =  select x,y,class from projection where idMethod = idMethods[0]
#   # projection2 =  select x,y,class from projection where idMethod = idMethods[1]

#   return (projection1, projection2, projection3)

# conn = engine.connect() # .fetchall() .keys()

# print( str(["TSNE-Gower"]))

# getProjectedData(2, "Synthetic4Classes", ["TSNE-Gower",'UMAP - 1HE'])

# shepherd = "Martha"
# age = 34
# stuff_in_string = "Shepherd %s is %d years old." % (shepherd, age)
# print(stuff_in_string)

# stuff_in_string = "Shepherd {} is {} years old.".format(shepherd, age)
# print(stuff_in_string)