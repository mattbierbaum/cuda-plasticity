import numpy
import vtk

def exitCheck(obj, event):
    if obj.GetEventPending() != 0:
        obj.SetAbortRender(1)
 
def plot(N, field, prefix="vtkrender", animate=True, write=True):
    field = numpy.tanh(field/field.mean()/8)
    maxfield = field.max()
    if maxfield > 5:
        maxfield = 5
    field = field*255/maxfield
    field = (field>255)*255+(field<=255)*field
    field = field.astype('uint8')

    timeSeries = True
    mip = False

    minopacity = 0.001
    maxopacity = 0.1

    dataImporter = vtk.vtkImageImport()
    dataImporter.SetDataScalarTypeToUnsignedChar()
    data_string = rho.tostring()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))

    dataImporter.SetDataExtent(0, N-1, 0, N-1, 0, N-1)
    dataImporter.SetWholeExtent(0, N-1, 0, N-1, 0, N-1)
 
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, maxopacity)
    alphaChannelFunc.AddPoint(255, minopacity)

    volumeProperty = vtk.vtkVolumeProperty()
    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(255, 1.0, 1.0, 1.0)

    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)

    volumeMapper = vtk.vtkVolumeRayCastMapper()
    if mip:
        mipFunction = vtk.vtkVolumeRayCastMIPFunction()
        mipFunction.SetMaximizeMethodToOpacity()
        volumeMapper.SetVolumeRayCastFunction(mipFunction)
    else:
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)

    renderer.AddVolume(volume)
    renderer.SetBackground(1.0,1.0,1.0)
    #renderer.SetBackground(0.6,0.6,0.6)
    renderWin.SetSize(400, 400)
                            
    renderWin.AddObserver("AbortCheckEvent", exitCheck)
                              
    if not animate:
        renderInteractor.Initialize()
    renderWin.Render()

    if not animate:
        renderInteractor.Start()

    if animate: 
        writer = vtk.vtkPNGWriter()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(renderWin)
        writer.SetInputConnection(w2i.GetOutputPort())

        renderWin.Render()
        ac = renderer.GetActiveCamera()
        ac.Elevation(20)
        step = 1
        current = 0
        for i in range(0,360,step):
            ac.Azimuth(step)
            #ac.Elevation(1*((-1)**(int(i/90))))
            renderer.ResetCameraClippingRange()
            renderWin.Render()
            w2i.Modified()
    
            if write:
                writer.SetFileName("%s%04d.gif" % (prefix,current))
                writer.Write()
            current += 1
        writer.End()

