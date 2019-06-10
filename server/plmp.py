from __future__ import print_function
import time
import sys
# from projection import projection
# sys.path.insert(0, './plpm/projection')

from plmp.projection import projection
from plmp.force import force
from plmp.plmp import PLMP
import numpy as np


print("Loading data set... ", end="")
sys.stdout.flush()
data_file = np.loadtxt("iris.data")
print("Done.")
ninst, dim  = data_file.shape
sample_size = int(np.ceil(np.sqrt(ninst)))
data        = data_file[:, range(dim - 1)]
data_class  = data_file[:, dim - 1]
sample      = np.random.permutation(ninst)
sample      = sample[range(sample_size)] # select sample_size random index

# force
start_time = time.time()
print("Projecting samples... ", end="")
sys.stdout.flush()
f = force.Force(data[sample, :], [])
f.project()
sample_projection = f.get_projection()
print("Done. (" + str(time.time() - start_time) + "s.)")


print(data.shape)
print(data_class.shape)
print(sample.shape)
print(sample_projection.shape)
# (150, 4)
# (150,)
# (13,)
# (13, 2)

# PLMP
start_time = time.time()
print("Projecting... ", end="")
sys.stdout.flush()
plmp = PLMP(data, data_class, sample, sample_projection)
plmp.project()
print("Done. (" + str(time.time() - start_time) + "s.)")
# plmp.plot()

x_projected = plmp.get_projection()
print(x_projected.shape)