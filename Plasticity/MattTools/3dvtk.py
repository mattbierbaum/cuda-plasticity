from Plasticity import PlasticitySystem
from Plasticity.FieldInitializers import FieldInitializer
import numpy
import vtk
from Plasticity.Observers import OrientationField
from Plasticity import TarFile

N=128
#t, st = FieldInitializer.LoadStateRaw("/a/woosong/cuda_3d/slip/3d_cuda_sliprelax_mix0.9_t1.0_128_dp_0_L1.plas", N, 3)
t,st = TarFile.LoadTarState("/media/scratch/plasticity/lvp3d128_s4_d.tar") 
#FieldInitializer.LoadState("3d_vac_cuda_relax_128_dp_0_L0.save")

print 't=',t
rho = st.CalculateRhoSymmetric()

print 'rho max:', rho.modulus().max()
print 'energy:', st.CalculateElasticEnergy().sum()

rho = rho.modulus()
maxrho = rho.max()
if maxrho > 5:
    maxrho = 5
rho = rho*255/maxrho
rho = (rho>255)*255+(rho<=255)*rho
rho = rho.numpy_array().astype('uint8')

timeSeries = True
mip = False
maxopacity = 1.0

dataImporter = vtk.vtkImageImport()
dataImporter.SetDataScalarTypeToUnsignedChar()
data_string = rho.tostring()
dataImporter.SetNumberOfScalarComponents(1)
dataImporter.CopyImportVoidPointer(data_string, len(data_string))

dataImporter.SetDataExtent(0, N-1, 0, N-1, 0, N-1)
dataImporter.SetWholeExtent(0, N-1, 0, N-1, 0, N-1)
 
alphaChannelFunc = vtk.vtkPiecewiseFunction()
alphaChannelFunc.AddPoint(0, 0.01)
alphaChannelFunc.AddPoint(255, maxopacity)

volumeProperty = vtk.vtkVolumeProperty()
colorFunc = vtk.vtkColorTransferFunction()
colorFunc.AddRGBPoint(0, 1.0, 1.0, 1.0)
colorFunc.AddRGBPoint(255, 0.0, 0.0, 0.0)
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
renderer.SetBackground(0.7,0.7,0.7)
renderWin.SetSize(400, 400)

def exitCheck(obj, event):
    if obj.GetEventPending() != 0:
        obj.SetAbortRender(1)
                             
renderWin.AddObserver("AbortCheckEvent", exitCheck)
                              
#renderInteractor.Initialize()
renderWin.Render()
#renderInteractor.Start()

#writer = vtk.vtkFFMPEGWriter()
#writer.SetQuality(2)
#writer.SetRate(24)
#w2i = vtk.vtkWindowToImageFilter()
#w2i.SetInput(renderWin)
#writer.SetInputConnection(w2i.GetOutputPort())
#writer.SetFileName('movie.avi')
#writer.Start()

writer = vtk.vtkPNGWriter()
w2i = vtk.vtkWindowToImageFilter()
w2i.SetInput(renderWin)
writer.SetInputConnection(w2i.GetOutputPort())
writer.SetFileName('movie.avi')

renderWin.Render()
ac = renderer.GetActiveCamera()
ac.Elevation(30)
num = 120
for i in range(num):
    ac.Azimuth(360/num)
    #ac.Elevation(1*((-1)**(int(i/90))))
    renderer.ResetCameraClippingRange()
    renderWin.Render()
    w2i.Modified()
    writer.SetFileName("movie_%03d.png" % i)
    writer.Write()
#writer.End()

