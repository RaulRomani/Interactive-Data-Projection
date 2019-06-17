import pandas as pd
import numpy as np


dataset_name = "Caltech"

relative = "../../../"

df = pd.read_csv(relative + "datasets/" + dataset_name + '/'+ dataset_name + '.csv', sep=";", header=None)
df = df.drop(0, 1)


print(df.describe())
print(df.nunique())
print(df.head())
print(df.shape)


df[11]   = pd.Categorical(df[11])
df[11]   = df[11].cat.codes
num_cols = df.shape[1]-1

np.savetxt(relative + "datasets/" + dataset_name + '/' + dataset_name + "_prep_encoding2.csv", df.values[:,:num_cols], delimiter=",")
np.savetxt(relative + "datasets/" + dataset_name + '/' + dataset_name + "_labels.csv", df.values[:,num_cols], delimiter=",")



import umap


X_embedded = umap.UMAP().fit_transform(df.values[:,:num_cols])


import matplotlib.pyplot as plt


plt.scatter(X_embedded[:,0], X_embedded[:,1], c = df.values[:,num_cols])
plt.show()