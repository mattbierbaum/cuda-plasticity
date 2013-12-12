import numpy

def ReadInScalarField(filename,shape):
    file = open(filename,'r')
    lines = file.readlines()
    data = []
    for i in range(len(lines)):
        items = lines[i].rsplit(' ')
        if '\n' in items:
            items.remove('\n')
        if '' in items:
            items.remove('')
        data.append(items)     
    file.close()
    return (numpy.array(data).reshape(shape)).astype(float) 

def InputXY(filename):
    file = open(filename,'r')
    lines = file.readlines()
    x,y = [],[]
    for i in range(len(lines)):
        items = lines[i].rsplit(' ')
        x.append(float(items[0]))     
        y.append(float(items[1]))     
    file.close()
    return numpy.array(x),numpy.array(y)

def InputXYE(filename):
    file = open(filename,'r')
    lines = file.readlines()
    x,y,e = [],[],[]
    for i in range(len(lines)):
        items = lines[i].rsplit(' ')
        x.append(float(items[0]))     
        y.append(float(items[1]))     
        e.append(float(items[2]))     
    file.close()
    return numpy.array(x),numpy.array(y),numpy.array(e)

def OutputXY(x,y,filename):
    file = open(filename,'w+') 
    for i in range(len(x)):
        file.write('%.14f'%(x[i])+' ')
        file.write('%.14f'%(y[i])+' ')
        file.write("\n")
    file.close() 

def OutputXYE(x,y,e,filename):
    file = open(filename,'w+') 
    for i in range(len(x)):
        file.write('%.14f'%(x[i])+' ')
        file.write('%.14f'%(y[i])+' ')
        file.write('%.14f'%(e[i])+' ')
        file.write("\n")
    file.close() 

def OutputXYE_MultipliedByRtoPower(x,y,e,power,filename):
    newy = y*x**power
    newe = e*x**power
    file = open(filename,'w+') 
    for i in range(len(x)):
        file.write('%.14f'%x[i]+' ')
        file.write('%.14f'%newy[i]+' ')
        file.write('%.14f'%newe[i]+'  ')
        file.write("\n")
    file.close() 

