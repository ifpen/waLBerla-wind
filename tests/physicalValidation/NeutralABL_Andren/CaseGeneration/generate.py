# https://www.sciencedirect.com/science/article/pii/S016761052300185X?ref=pdf_download&fr=RR-2&rr=9097a0df4dacd57c
#
import numpy as np
import os
import shutil
import fileinput
#
## Python libraries
import numpy as np  # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library
import matplotlib  # Plotting library
from mpl_toolkits.axes_grid1 import make_axes_locatable
   
outDir = "NeutralABL_Andren_Files"
if not os.path.exists(outDir):
	os.makedirs(outDir)  

heightPhy      = 1500                # Physical, in (m)
widthPhy       = 1290
lengthPhy      = 1290

heightLU       = 30                  # 

wind           = 10.0                # Physical, in (m/s), Velocity at height z=900m
refHeightPhy   = 1500.
refHeight      = refHeightPhy/heightPhy*heightLU

machNumber     = 0.1                  # Non-dimensional
viscosity      = 1.82e-05             # Physical, in (N.m/kg) 
density        = 1.0                 # Physical, in (kg/m3)

domainLength_LU  = lengthPhy / heightPhy * heightLU
domainWidth_LU   = widthPhy  / heightPhy * heightLU
domainHeight_LU  = heightLU  

periodicShift = int(7./10. * heightLU)

z0             = 0.1
z0_over_h      = z0 / refHeightPhy

# Compute the domain size
domSize      = [domainLength_LU, domainWidth_LU, domainHeight_LU] 

# Estimate uLBM based on the Mach number
soundSpeedLBM = 1./np.sqrt(3.)
uLBM = machNumber * soundSpeedLBM
C_l  = heightPhy/heightLU
C_u  = wind / uLBM
C_t  = C_l / C_u


# Estimate the total simulation time based on the time needed for a particle to go through the domain 3 times at u_Infinity
#timeToTravelDomain = int(domSize[0] / uLBM)
# Since we use a domain 6 time longer, this should be sufficient. To be checked
#vtkEvery           = int(5.*timeToTravelDomain)
#timeSimulation     = 200*timeToTravelDomain

#coriolisFrequency = 1.e-4
##coriolisPeriod = 1./ coriolisFrequency # / C_t #/ C_t # C_t converts to physical units, thus 1/C_t converts to LU
##timeSimulation = 36.*coriolisPeriod 
#timeSimulationPhy  = 3.*30./coriolisFrequency 
#vtkEveryPhy        = 3.*6./coriolisFrequency
#timeSimulation     = timeSimulationPhy / C_t
#vtkEvery           = vtkEveryPhy / C_t

timeSimulation  = 70.*3600. / C_t
vtkEvery        = timeSimulation / 2.
print(timeSimulation)

#print("Simulation time, vtkEvery: ", timeSimulationPhy/3600., vtkEveryPhy/3600.)

AABBout  = [domainWidth_LU, domainWidth_LU, domainHeight_LU]

print(timeSimulation, vtkEvery)

# Write the .prm file
new_case = outDir+'/NeutralABL_Andren_Files_'+str(int(heightLU))+'cells.prm'
shutil.copy('refCase.prm', new_case)

for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$LOADSNAP', str(0)).rstrip())  
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$STORESNAP', str(0)).rstrip()) 
    
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$TURBINES', str("")).rstrip()) 
    
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$xDom', str(round(domSize[0]))).rstrip())  
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$yDom', str(round(domSize[1]))).rstrip())  
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$zDom', str(round(domSize[2]))).rstrip())      

for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$z0', str(z0_over_h)).rstrip())     
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$refHeight', str(refHeight)).rstrip())   
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$samplingHeight', str(1)).rstrip())       
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$periodicShift', str(periodicShift)).rstrip())     
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$geoWind', str(wind)).rstrip())  

for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$lLBM', str(heightLU)).rstrip())  
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$uLBM', str(uLBM)).rstrip())  

for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$nstepsLBM', str(timeSimulation)).rstrip())         
    
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$lSI', str(heightPhy)).rstrip())  
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$uSI', str(wind)).rstrip())                   
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$viscositySI', str(viscosity)).rstrip())  
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$densitySI', str(density)).rstrip())        

for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$Precursor_VTK_Frequency', str(int(vtkEvery))).rstrip())
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$Precursor_VTK_OUT', str("Precursor_VTK_Out_v2_"+str(heightLU))).rstrip())    
    
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$Concurrent_VTK_Frequency', str(int(vtkEvery))).rstrip())
for line in fileinput.input( new_case, inplace=True):
    print (  line.replace('$AABBx', str(int(AABBout[0]))).rstrip())
for line in fileinput.input( new_case, inplace=True):
    print (  line.replace('$AABBy', str(int(AABBout[1]))).rstrip())
for line in fileinput.input( new_case, inplace=True):
    print (  line.replace('$AABBz', str(int(AABBout[2]))).rstrip())

for line in fileinput.input( new_case, inplace=True):
    print (  line.replace('$minAABBx', str(int(0))).rstrip())
for line in fileinput.input( new_case, inplace=True):
    print (  line.replace('$minAABBy', str(int(0))).rstrip())
for line in fileinput.input( new_case, inplace=True):
    print (  line.replace('$minAABBz', str(int(0))).rstrip())

for line in fileinput.input( new_case, inplace=True):
    print (  line.replace('$maxAABBx', str(int(AABBout[0]))).rstrip())
for line in fileinput.input( new_case, inplace=True):
    print (  line.replace('$maxAABBy', str(int(AABBout[1]))).rstrip())
for line in fileinput.input( new_case, inplace=True):
    print (  line.replace('$maxAABBz', str(int(AABBout[2]))).rstrip())

lineOutput=""
for line in fileinput.input( new_case, inplace=True):      
    print (  line.replace('$LINE_OUTPUT', str(lineOutput)).rstrip())  

print("Mesh size: ", (domSize[0]*domSize[1]*domSize[2])*1e-6, " million nodes.")
