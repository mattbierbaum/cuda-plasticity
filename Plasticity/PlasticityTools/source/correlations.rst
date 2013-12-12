
Correlation Functions
===========================================

Package Overview
------------------------------------

This package calculates general spatial correlation functions of scalar, vector, and tensor fields contained in numpy.array like data structures.  These data structures can be in the form:

* (dim[0], dim[1], ..., dim[N]) - scalar field in N dimensions
* (M, dim[0], ..., dim[N]) - a length M vector field in N dimensions
* (K, L, dim[0], ..., dim[N]) - a KxL tensor field in N dimensions

To find the correlation function, call one of CorrelationFunctionsOf{Scalar,Vector,Scalar}Field or RadialCorrelationFunctions if you can assume radial symmetry. A radial correlation function can also be produced from a correlation function made from the first set of functions.

For some examples:

.. code-block:: python
	:linenos:

	import numpy as np
	import pylab as pl
	import CorrelationFuncions as cf

	# create a 3d vector field on a 2d grid
	vec = np.random.random((3,128,128))

	# calculate the radial correlation
	corr = cf.RadialCorrelationFunctions(vec, type='vector')
	pl.loglog(corr[0], corr[1], 'o-')

Module Listing
--------------------------------------

.. automodule:: CorrelationFunction
	:members:

