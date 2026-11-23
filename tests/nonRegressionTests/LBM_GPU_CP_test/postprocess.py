import numpy as np
import pyvista as pv
import sys as sy

dataConcurrent    = pv.get_reader("Concurrent_VTK_Out/field_writer_1000/simulation_step_1000.pvtu").read()
instVelConcurrent = dataConcurrent["Velocity"]    
meanVel           = dataConcurrent["meanVelocity"]
sop               = dataConcurrent["sumofsquares"]
force             = dataConcurrent["Force"]

dataPrecursor    = pv.get_reader("Precursor_VTK_Out/field_writer_2000/simulation_step_2000.pvtu").read()
instVelPrecursor  = dataPrecursor["Velocity"]    

refMeanInstVel     = 0.01844181
refMeanMeanVel     = 0.018711599
refMeanSOP         = 0.0039672027
refMeanForce       = -3.912854e-07
refMeanPrecInstVel = 0.018729053

errorMeanInstVel     = np.linalg.norm(refMeanInstVel - np.mean(instVelConcurrent[:,:]))
errorMeanMeanVel     = np.linalg.norm(refMeanMeanVel - np.mean(meanVel[:,:]))
errorMeanSOP         = np.linalg.norm(refMeanSOP - np.mean(sop[:,:]))
errorMeanForce       = np.linalg.norm(refMeanForce - np.mean(force[:,:]))
errorMeanPrecInstVel = np.linalg.norm(refMeanPrecInstVel - np.mean(instVelPrecursor[:,:]))

# Reference values
print(np.mean(instVelConcurrent[:,:]), np.mean(meanVel[:,:]), np.mean(sop[:,:]), np.mean(force[:,:]), np.mean(instVelPrecursor[:,:]))

succes = True
epsilon = 1e-4
if(errorMeanInstVel > epsilon):
    print('Computed error on the mean instantaneaous velocity ', 'errorMeanInstVel, is higher than threshold of ', epsilon)
    succes = False
    
if(errorMeanMeanVel > epsilon):
    print('Computed error on the mean mean velocity ', 'errorMeanMeanVel, is higher than threshold of ', epsilon)
    succes = False
    
if(errorMeanSOP > epsilon):
    print('Computed error on the mean SOP ', 'errorMeanSOP, is higher than threshold of ', epsilon)    
    succes = False   
    
if(errorMeanForce > epsilon):
    print('Computed error on the mean force', 'errorMeanForce, is higher than threshold of ', epsilon)    
    succes = False       
   
if(errorMeanPrecInstVel > epsilon):
    print('Computed error on the mean precursor instantaneous velocity', 'errorMeanPrecInstVel, is higher than threshold of ', epsilon)    
    succes = False   
    
if(succes):
    exit_code = 0
else:
    exit_code = 1

sy.exit(exit_code)
