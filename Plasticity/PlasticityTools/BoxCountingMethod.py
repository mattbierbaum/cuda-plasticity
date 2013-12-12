import numpy

import sys
import getopt

def BoxCountingMethod(data,cutoff=0.,periodic_boundary_condition=True):
	"""
	This method works for fractal structures from 1D to 3D.
	
	If data is not a binary array, it will be converted according to the cutoff.
	That is, values above cutoff will be counted as 1, below it as 0.

	By default, the data array has the same length along all axes, and the length is 
	usually a power of 2. So the boxsizes are given as a list of powers of 2.
	Certainly, you can supply a special list of boxsizes for any certain system size L. 

	For example, if data is an array of [128,128], the boxsizes are given by
	boxsizes = [1,2,4,8,16,32,64]. 

	This function will return two numpy arrays: the former is the array of normalized 
	boxsizes, and the latter is the corresponding array of the total number of 'black' 
	boxes.  
	"""
	binarydata = (data>cutoff).astype(int) 

	shape = list(data.shape)
	padding = False
	for i in range(len(shape)):
		if numpy.log2(shape[i]) != round(numpy.log2(shape[i])):
			padding = True
			shape[i] = numpy.power(2,int(numpy.log2(shape[i]))+1)
	if padding:
		binarydata = PaddingData(binarydata,periodic_boundary_condition)
	
	boxsizes = numpy.power(2,numpy.arange(numpy.log2(min(binarydata.shape)))) 
	
	countedBoxes = []
	for b in boxsizes:
		countedBoxes.append((DivideIntoBoxes(binarydata,int(b))).sum())

	L = min(data.shape)
	return boxsizes/float(L), numpy.array(countedBoxes)


def DivideIntoBoxes(data,boxsize):
	"""
	Does the box counting by taking every boxsized-th element of the array
	by first adding all of the elements in a box then translating it to a binary
	value of has data or not.

	Binary data should be 1 for occupied sites and 0 for unoccupied.

	Along every axis, the length of data array should be divided by the boxsize. 
	"""
	dim = len(data.shape)
	newdata = 0. # Adding double to array gives array. 
	if dim == 1:
		for i in range(boxsize):
			newdata += data[i::boxsize]
	elif dim == 2:
		for i in range(boxsize):
			for j in range(boxsize):
				newdata += data[i::boxsize,j::boxsize]
	elif dim == 3:
		for i in range(boxsize):
			for j in range(boxsize):
				for k in range(boxsize):
					newdata += data[i::boxsize,j::boxsize,k::boxsize]
	return (newdata>0).astype(int)

def Demo():
	"""
	Shows a simple example of taking random data and using the 
	box-counting method on it with a cutoff of 1/2
	"""

	test = numpy.random.rand(256,256)
	cutoff = 0.5
	import pylab
	pylab.figure(0)
	pylab.imshow((test>cutoff).astype(int),interpolation='nearest',cmap=pylab.cm.gist_gray_r)
	pylab.axis('off')
	pylab.title("Binary data array",fontsize=25)
	pylab.figure(1)
	x,y = BoxCountingMethod(test,cutoff=0.7) 
	pylab.loglog(x,y,'.-')
	pylab.xlabel(r'$\Delta x$',fontsize=20)
	pylab.ylabel(r'$N(\Delta x)$',fontsize=20)
	pylab.show()

def Run(inputfile,shape,outputfile,cutoff,input=None):
	"""
	Input array can be provided either as a numpy array to the
	input argument or as a space separated file given as 
	inputfile.  Output is a plot of the number of boxes against
	the box size.
	"""	
	if input is None:
		import Data_IO
		data = Data_IO.ReadInScalarField(inputfile,shape) 
	else:
		data = input
	x,y = BoxCountingMethod(data,cutoff)
	if outputfile is None:
		outputfile = inputfile+'_boxcounting_output.dat'
	Data_IO.OutputXY(x,y,outputfile)
	import pylab
	pylab.figure()
	pylab.loglog(x,y,'.-')
	pylab.xlabel(r'$\Delta x$',fontsize=20)
	pylab.ylabel(r'$N(\Delta x)$',fontsize=20)
	pylab.show()

def main(argv):
	try:
		opts, args = getopt.getopt(argv,'hi:o:s:c:b:d',['help','input=','output=',\
										'shape=','cutoff=','boxsizes=','demo'])
	except getopt.GetoptError, err:
		print str(err)
		sys.exit(2)
	inputfile,outputfile,shape,cutoff,boxsizes = None,None,None,None,None
	for opt,arg in opts:
		if opt in ('-i','--input'):
			inputfile = arg
		elif opt in ('-o','--output'):
			outputfile = arg
		elif opt in ('-s','--shape'):
			shape = numpy.array(arg.rsplit(',')).astype(int) 
		elif opt in ('-c','--cutoff'):
			cutoff = float(arg)
		elif opt in ('-d','--demo'):
			Demo() 
		elif opt in ('-h','--help'):
			print """ 
					[OPTIONS]				   FUNCTIONS 
				  -h,--help				  This help page 
				  -d,--demo				  Show the demonstration 
				  -i,--input=<inputfile>	 Pass the name of input data file 
				  -o,--output=<outputfile>   Pass the name of output data file 
				  -s,--shape=<L,L>		   Specify the shape of data array 
				  -c,--cutoff=<cutoff>	   Set cutoff to get the binary data set 
				  """
	if (inputfile is not None) and (shape is not None):
		Run(inputfile,shape,outputfile,cutoff)
		
if __name__ == "__main__":
	main(sys.argv[1:])
