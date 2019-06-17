# from sklearn import datasets
# import numpy as np

# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data  # we only take the first two features.
# y = iris.target

# print(X.shape)
# print(y.shape)


# dataset_name = "Iris"

# np.savetxt("../../datasets/" + dataset_name + '/' + dataset_name + "_prep_encoding2.csv", X, delimiter=",")
# np.savetxt("../../datasets/" + dataset_name + '/' + dataset_name + "_labels.csv", np.reshape(y, (-1,1)), delimiter=",")

import pandas as pd
import numpy as np

# # dataset_name = "GirondeEmployment"
# # dataset_name = "GirondeServices"
# dataset_name = "GirondeEnvironment"
# # df = pd.read_csv("PCAMix/"+ dataset_name + '_projected_PCAMix.csv')
# df = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '.csv')
# # labels = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_labels.csv', header=None).values.reshape((-1))

# # print(df.shape)
# # print(df.head(1))
# # print(df.values[0,:])
# np.savetxt("../../datasets/" + dataset_name + '/' + dataset_name + "_prep_encoding2.csv", df.values, delimiter=",")


dataset_name = "Iris"
df = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_prep_encoding2.csv', header=None)
print(df.describe())

# dataset_name = "OrthopedicPatients"
# df = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '.csv', header=None)
# labels = pd.read_csv("../../datasets/" + dataset_name + '/'+ dataset_name + '_labels.csv', header=None).values.reshape((-1))


# df[6] = pd.Categorical(df[6])
# df[6] = df[6].cat.codes

# # print(df.head)
# # categorical_values = np.array(list(set(df.values[:,6])))
# # print(categorical_values)

# # print(df.shape)
# # print(df.head(2))
# # print(df.values[0,:])
# np.savetxt("../../datasets/" + dataset_name + '/' + dataset_name + "_prep_encoding2.csv", df.values[:,:6], delimiter=",")
# np.savetxt("../../datasets/" + dataset_name + '/' + dataset_name + "_labels.csv", df.values[:,6], delimiter=",")


