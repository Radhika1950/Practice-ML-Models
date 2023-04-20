# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 08:50:59 2022

@author: rkbab
"""

##1. Practicing with NumPy
##A. Defining an array and printing. Use the following code
import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))
print(type(x))
print(x.shape)

#2. Practicing with SciPy
#A. Creating an array
from scipy import sparse
# Create a 2D NumPy array with a diagonal of ones, and zeroes everywhere else
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))

#B. Convert the NumPy array to a SciPy sparse matrix in CSR format
#Convert the NumPy array to a SciPy sparse matrix in CSR format
#Only the non zero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))

#C. Using coordinate format
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n{}".format(eye_coo))

#3. Practicing with MatPlotlib
#A. Plotting a line chart
# Generate a sequence of numbers from -10 to 10 with 100 steps in between
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 100)
# Create a second array using sine
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker = 'x')

#4. Practicing with Pandas
#A. Printing data frames
import pandas as pd
from IPython.display import display
# Create a simple data set of people
data = {'Name':["John", "Anna", "Peter", "Linda"], 
'Location': ["New York", "Paris", "Berlin", "Linda"],
'Age': [24, 13, 53, 33]}
data_pandas = pd.DataFrame(data)
# IPython.display allows “pretty printing” of data frames
# in the Spyder notebook
display(data_pandas)
# Select all rows that have an age column greater than 30 
display(data_pandas[data_pandas.Age > 30])

#5. TESTING THE VERSION AND LIBRARIES IN YOUR ANACONDA ENVIRONMENT
import sys
print("Python version: {}".format(sys.version))
import pandas as pd
print("pandas version: {}".format(pd.__version__))
import matplotlib
print("matplotlib version: {}".format(matplotlib.__version__))
import numpy as np
print("NumPy version: {}".format(np.__version__))
import scipy as sp
print("SciPy version: {}".format(sp.__version__))
import IPython
print("IPython version: {}".format(IPython.__version__))
import sklearn
print("Scikit-learn version: {}".format(sklearn.__version__))
