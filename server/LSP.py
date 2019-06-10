import pandas as pd
import sys
import os


sys.path.insert(0, './LSP')

                    
dataset_name = "OrthopedicPatients"
os.system("./runLSP.m " + dataset_name)

# labels = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_labels.csv', header=None).values.reshape((-1))
# proj   = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_projected_octave.csv', header=None).values

# # os.remove("Runtime.csv")

# import matplotlib.pyplot as plt

# plt.scatter(proj[:,0], proj[:,1], c= labels)
# plt.show()
