import pyvista
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from collections import OrderedDict
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib

from scipy.interpolate import interp1d

import sys

scale = 1.
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'axes.titlesize' : 24*scale})
matplotlib.rcParams.update({'axes.labelsize' : 24*scale})
matplotlib.rcParams.update({'lines.linewidth' : 3})
matplotlib.rcParams.update({'lines.markersize' : 8})
matplotlib.rcParams.update({'xtick.labelsize' : 20*scale})
matplotlib.rcParams.update({'ytick.labelsize' : 20*scale})
matplotlib.rcParams.update({'legend.fontsize' : 16*scale})

def logProfile(z, uRef, z0):
    z0 = 0.1/1500.*40.
    return uRef/0.42*np.log(z/z0)

delta          = 1500 # Domain height in (m)

velocityLU = 0.05773502691896259
velocitySI = 10.
toPhysical = velocitySI/velocityLU
G = velocitySI

files = ['../NeutralABL_Andren_Files/Precursor_VTK_Out_v2_30/field_writer_436476/simulation_step_872952.pvti']# , '../NeutralABL_Andren_Files/Precursor_VTK_Out_v2_50/field_writer_727461/simulation_step_1454922.pvti']
nSamples = ['436476']#, '727461']
deltaLUs        = [30]#, 50]

shapes = ['-', '--', '-.', ':']

f, ax = plt.subplots(ncols=6, figsize=(18,4), sharey=True, constrained_layout=True)
fLM, axLM = plt.subplots(ncols=1, figsize=(5.5,5), constrained_layout=True)


for deltaLU, nSample, sh, file in zip(deltaLUs, nSamples, shapes, files):
    
    data = pyvista.get_reader(file).read()
    centers = vtk_to_numpy(data.cell_centers().GetPoints().GetData())
    vel = data["meanVelocity"]
    sop = data["sumofsquares"] / float(nSample)

    zpos = np.arange(0.5, deltaLU + 0.5)
    ninj = deltaLU

    # Sample vertical profiles and average horizontally
    velocities, stresses = [], []

    for i in range(ninj):
        for j in range(ninj):
            points = np.column_stack((np.full(len(zpos), i), np.full(len(zpos), j), zpos, ))
            idx = data.find_closest_cell(points)
            velocities.append(vel[idx])
            stresses.append(sop[idx])

    meanVel = np.mean(velocities, axis=0)
    meanSOP = np.mean(stresses, axis=0)

    zLBM = centers[idx, 2] / deltaLU
    velH = np.hypot(meanVel[:, 0], meanVel[:, 1])

    indices = np.where((zLBM >= 0.1) & (zLBM <= 0.5))[0]
    poptInflow, _ = curve_fit(
        logProfile, centers[idx, 2][indices], vel[idx, 0][indices],
        p0=[0.4, 1e-3],
    )
    uStarLBM = poptInflow[0]
    uStar = uStarLBM * toPhysical

    # Profiles to plot: (values, axis, label, output column)
    profiles = [
        (velH,                  0, r'$\sqrt{\overline{u}^2+\overline{v}^2}/G$ $(-)$', 1),
        (meanSOP[:, 0],         1, r"$\overline{u'u'}/u_*^2$ $(-)$",                 2),
        (meanSOP[:, 4],         2, r"$\overline{v'v'}/u_*^2$ $(-)$",                 3),
        (meanSOP[:, 8],         3, r"$\overline{w'w'}/u_*^2$ $(-)$",                 4),
        (meanSOP[:, 2],         4, r"$\overline{u'w'}/u_*^2$ $(-)$",                 5),
        (meanSOP[:, 5],         5, r"$\overline{v'w'}/u_*^2$ $(-)$",                 6),
    ]

    wlbWind_results = np.column_stack((
        zLBM, velH, meanSOP[:, 0], meanSOP[:, 4], meanSOP[:, 8], meanSOP[:, 2], meanSOP[:, 5],
    ))

    for values, axis, xlabel, outCol in profiles:
        scale = G if axis == 0 else uStarLBM**2
        ax[axis].plot(values / scale * (toPhysical if axis == 0 else 1),
                      zLBM, sh,
                      label='waLBerla-wind')
        ax[axis].set_xlabel(xlabel)

    axLM.plot(zLBM, velH / uStarLBM, '-x', label='waLBerla-wind')

z0 = 0.1
kappa = 0.41
z = np.arange(delta/1000., delta, delta/1000.)
logLaw = uStar / kappa * np.log(z/z0)
ax[0].plot   (logLaw/G, z/delta, 'k--', label='Log-law')
axLM.plot   (z/delta, logLaw/uStar, 'k--', label='Log-law') # LOGLAW
axLM.legend()
axLM.set_xlabel(r'$z/\delta$')
axLM.set_ylabel(r'$u/u_*$')

us_PLB_32  = 0.429
us_Senocak = 0.413580692631579

# Load Senocak data
data = np.genfromtxt('data_ProLB_ABL/Neutral/Senocak.csv',delimiter=',')
u_senocak = data[:, 0] * us_Senocak / G
z_senocak = data[:, 1] / delta

# Load ProLB data
data = np.genfromtxt('data_ProLB_ABL/Neutral/mean_dx32.csv',delimiter=',')
u_prolb = data[:, 1] / G
z_prolb = data[:, 0] / delta

# Interpolate both datasets onto a common vertical grid
z_common = np.linspace(max(np.min(z_senocak), np.min(z_prolb)),min(np.max(z_senocak), np.max(z_prolb)),200)
u_senocak_interp = np.interp(z_common, z_senocak, u_senocak)
u_prolb_interp = np.interp(z_common, z_prolb, u_prolb)

# Compute the minimum and maximum
u_min = np.minimum(u_senocak_interp, u_prolb_interp)
u_max = np.maximum(u_senocak_interp, u_prolb_interp)

# Plot the shaded region
ax[0].fill_betweenx(z_common,u_min,u_max,color='steelblue',alpha=0.3,label='Participant range')

# uPrime uPrime ###################################################################################
authors = ['AndrenMoeng', 'Nieuwstadt', 'MasonBrownNoBisect', 'SchumannCraf']
symbols = ['o', 'v', 's', 'D']
variables = [('upup', 1), ('vpvp', 2), ('wpwp', 3), ('upwp', 4), ('vpwp', 5),]

for var, ax_idx in variables:

    quantities = []
    heights = []

    # Load data from all participants
    for author in authors:
        dataTotal = np.genfromtxt(f'data_Scan/interpolated_{author}_{var}_total.dat')
        dataSGS = np.genfromtxt(f'data_Scan/interpolated_{author}_{var}_sgs.dat')
        quantity = dataTotal[:, 1] - dataSGS[:, 1]
        quantities.append(quantity)
        heights.append(dataTotal[:, 0])

    # Assume all participants use the same vertical coordinates
    heights = heights[0]
    quantities = np.array(quantities)

    # Compute minimum and maximum across participants
    q_min = np.min(quantities, axis=0)
    q_max = np.max(quantities, axis=0)

    # Plot the shaded envelope
    ax[ax_idx].fill_betweenx(heights,q_min,q_max,color='steelblue',alpha=0.3,label='Participant range')
        
handles, labels = ax[0].get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
leg = f.legend(by_label.values(), by_label.keys(),loc='upper center', ncol=3,bbox_to_anchor=(.5, 1.15),bbox_transform=f.transFigure)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.5)
leg.set_in_layout(True)

ax[0].set_ylabel(r'$z/D$ $(-)$')

axLM.set_xscale('log')

f.tight_layout()
f.savefig('profiles.png', format='png', bbox_inches='tight', dpi=250)

fLM.savefig('logLawMismatch.png', format='png', bbox_inches='tight', dpi=250)

# Regenerates the reference data
#np.savetxt('walberlaWind_Reference.dat', wlbWind_results)

referenceData = np.genfromtxt('walberlaWind_Reference.dat')
error = np.linalg.norm(wlbWind_results-referenceData)
print('Error: ', error, '%')

succes = True
epsilon = 2e-3
if(error > epsilon):
    print('Computed error on the mean profiles (', error, ') is higher than threshold of ', epsilon)
    succes = False

if(succes):
    exit_code = 0
else:
    exit_code = 1

sys.exit(exit_code)
