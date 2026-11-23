import numpy as np
import pyvista as pv
import sys as sy

data    = pv.get_reader("vtk_out/field_writer_1000/simulation_step_1000.pvtu").read()
instVel = data["Velocity"]    
meanVel = data["meanVelocity"]
sop     = data["sumofsquares"]

refMeanInstVel = 0.018701958
refMeanMeanVel = 0.019027866
refMeanSOP = 0.0059312982

errorMeanInstVel = np.linalg.norm(refMeanInstVel - np.mean(instVel[:,:]))
errorMeanMeanVel = np.linalg.norm(refMeanMeanVel - np.mean(meanVel[:,:]))
errorMeanSOP     = np.linalg.norm(refMeanSOP - np.mean(sop[:,:]))

# Reference values
print(np.mean(instVel[:,:]), np.mean(meanVel[:,:]), np.mean(sop[:,:]))

succes = True
epsilon = 1e-6
if(errorMeanInstVel > epsilon):
    print('Computed error on the mean instantaneaous velocity ', 'errorMeanInstVel, is higher than threshold of ', epsilon)
    succes = False
    
if(errorMeanMeanVel > epsilon):
    print('Computed error on the mean mean velocity ', 'errorMeanMeanVel, is higher than threshold of ', epsilon)
    succes = False
    
if(errorMeanSOP > epsilon):
    print('Computed error on the mean SOP ', 'errorMeanSOP, is higher than threshold of ', epsilon)    
    succes = False   
   
if(succes):
    exit_code = 0
else:
    exit_code = 1

sy.exit(exit_code)
