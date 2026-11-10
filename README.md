# waLBerla-wind

waLBerla-wind is an extension of [waLBerla](https://gitos.rrze.fau.de/cs10/walberla/walberla) for wind energy applications.

# Getting started

Being an extension of waLBerla, waLBerla-wind comes with the same minumum requirements: a C++-17-compliant compiler, an MPI library, and the [CMake](http://www.cmake.org/) build system. If you want to run simulations on GPUs, you need an additional CUDA installation.  
Lastly, as waLBerla-wind relies heavily on code generation, you will also need a Python installation. For the required Python version, please check in [pystencils](https://gitos.rrze.fau.de/cs10/pycodegen/pystencils) or [lbmpy](https://gitos.rrze.fau.de/cs10/pycodegen/lbmpy). At the time of writing, this is Python 3.12.

## Preparation

To successfully build waLBerla-wind, you will need to download waLBerla from [gitlab](https://gitos.rrze.fau.de/cs10/walberla/walberla) and install pystencils and lbmpy [gitlab](https://gitos.rrze.fau.de/cs10/pycodegen). Hints for the installation can be found in their corresponding README file in gitlab. 

For this version of waBerla-wind, please use release/1.4 for both pystencils and lbmpy.

## CMake

In the following, we will assume that you create your build folder within the waLBerla-wind directory. 
If you prefer to have an external build folder, please adapt the paths adequately.  

The first step is to create the build folder and enter it via
`mkdir build` and `cd build`.

In the build folder, you call CMake to build your project. For this, you must add certain options with `-D(Option)=(Value)`. The recommended minimal setup is
```cmake
cmake .. -DWALBERLA_DIR=/path/to/walberla/repository \
         -DPython_ROOT_DIR=/path/to/python/installation \
         -DWALBERLA_BUILD_WITH_CODEGEN=ON \
         -DWALBERLA_BUILD_WITH_PYTHON=ON \
```
Note that the `Python_ROOT_DIR` option is not necessary if you use the system's default Python installation.  

If you want to speed up your CPU simulation with vectorisation, please add
```cmake 
-DWALBERLA_OPTIMIZE_FOR_LOCALHOST=ON
```

If you want to enable GPU support, you should add the following options
```cmake 
-DWALBERLA_BUILD_WITH_CUDA=ON 
-DCMAKE_CUDA_ARCHITECTURES=XX
```
where `XX` is your GPU's compute capability (for NVIDIA GPUs, you can look it up [here](https://developer.nvidia.com/cuda-gpus)). Setting the `CMAKE_CUDA_ARCHITECTURES` is optional but recommended.  

For more potentially interesting available options, you can either use the `ccmake` tool that you call in your build folder as `ccmake .`, or you can check waLBerla's documentation.  

## Compiling your application

Currently, waLBerla-wind provides several basis applications for CPU and GPU, all of which can be found in the `apps` directory in waLBerla-wind.  
To compile an application, please call `make -j N` (`N` corresponds to the number of processes you want to compile with) either in `build/apps` to build ALL applications or `build/apps/{APP_NAME}` to compile only the applications of your choice.  
As standard applications, we provide applications on uniform grids for CPU and GPU (`LBM_CPU` and `LBM_GPU`); a CPU version with mesh refinement (`LBM_CPU_AMR`; the GPU version is planned to be released soon); and applications with a concurrent method for turbulent inflow (`LBM_CPU_CP` and `LBM_GPU_CP`).  
In addition, we provide an app generator and the corresponding generated basis apps that will soon replace the standard apps. Therefore, we recommend to use the generated versions for any development.

### App generation 

With the development of more and more features, the standard apps become loaded with potentially unused code if those features are not needed in the simulation. Not only makes this the application unreadable, it also increases the memory consumption and reduces the performance of the simulation.  
We, therefore, implemented an app generator where the user can choose exactly which features they want to use and which parts of the code can be removed.  
In the default generated applications `Generated_{hardware}_{grid}`, all features are included, resulting in a rather heavy setup.  

To define your own setup, you need to modify `walberla-wind/apps/AppGenerator/generate_apps.py`. Please ONLY modify parts of the code included between the `USER_DEFINED CONFIGURATION` comments! 
In the following, we will provide an overview of the different options.

- `turbine_communication`: If the wind turbines in the simulation are located close to block boundaries, a communication routine is required to ensure that the turbine data (position, forces, etc) are always up-to-date, even if a part of the turbine does not reside on the same process. In general, you should ALWAYS include this feature. Only if you can 100% ensure that no turbine requires this communication (as they are far enough away from a block boundary and do not move), you can turn this option off. 
- `turbine_output`: the turbine implementation comes with turbine-specific outputs, e.g., the position of the discretisation points, normal and tangential forces along the blade etc. As always, output is costly and should only be included if the output is really needed for the evaluation routine.
- `vtk_output`: When no VTK output of fields is required (for benchmarking runs, or because only turbine output is necessary), this option can be disabled.
- `field_map`: This list includes all fields quantities that should be calculated for VTK output. If you are not interested in a specific quantity, it is worth not including it in this list. Like this, the fields are never allocated, saving memory, nor calculated, increasing performance.  
  Note that some fields are included per default, as they are needed for internal routines. These are the PDF, density, velocity, force and flag fields. You can still exclude them from VTK output dynamically.  
  Also, if you include an averaged quantity, its instantaneous counterpart is automatically included for the Welford algorithm.
- `with_snapshot`: The snapshot feature allows the user to make snapshots of the simulation with a user-defined frequency, or to start the simulation from a snapshot. 
- `generate_wfb`: The wall-function bounce (WFB) requires the creation of a dedicated field and therefore consumes memory, even if it is not applied. If you know that you model walls only with no-slip or other boundary conditions, you can deactivate the generation of the WFB with this option.
- `generate_control`: This option adds the controllers for torque and pitch to the application. Note that you can still decide at runtime not to use the controllers (but the controller routine will be entered).
- `streaming_pattern`: For now, only the pull streaming pattern is supported. Do not change it, please.

Once you made all the desired adaptions to the generation script, go to `build/apps/AppGeneration` and call `make`. This make will update all files in the `Generated_{hardware}_{grid}` source folders.  
Afterwards, you can enter the build folder of the desired generated application, e.g., `build/apps/GeneratedApp_cpu_uniform` and call `make -j N` again. The application will be compiled and ready for use.  

NOTE: If you ever happen to call `make clean` in either `build/apps` or `build/apps/Generated_{hardware}_{grid}`, this will unfortunately also clean the `CMakeLists.txt` file in the source directory of the generated app(s). The next time you will call `cmake`, this will result in an error. Until we found a workaround, please proceed as follows:  
Comment out the files concerning the generated app(s) in `apps/CMakeLists.txt`, call `cmake`, and build the `AppGenerator` target again. This will recreate the missing `CMakeLists.txt` file(s).  
Once this is done, you can comment in the lines in `apps/CMakeLists.txt` again and proceed as normal.

## Running an application

Once your desired application is successfully built, you can run the executable either serially or with `mpirun`, in both cases providing the parameter file as argument.

# Please cite us
If you use waLBerla-wind in a publication, please cite the following articles.

### Overview
- Schottenhamml H, Anciaux Sedrakian A, Blondel F, Köstler H, Rüde U. waLBerla-wind: A lattice-Boltzmann-based high-performance flow solver for wind energy applications. Concurrency Computat Pract Exper. 2024; 36(16):e8117. doi: [10.1002/cpe.8117](https://doi.org/10.1002/cpe.8117)

### Validation cases
- DTU 10MW wind tunnel case; comparison with the vortex solver Castor and OpenFOAM-based software SOWFA  
  Helen Schottenhamml et al 2022 J. Phys.: Conf. Ser. 2265 022027
  DOI [10.1088/1742-6596/2265/2/022027](https://doi.org/10.1088/1742-6596/2265/2/022027)
- NewMexico wind tunnel test case; comparison with the vortex solver Castor   
Schottenhamml H, Anciaux Sedrakian A, Blondel F, Köstler H, Rüde U. waLBerla-wind: A lattice-Boltzmann-based high-performance flow solver for wind energy applications. Concurrency Computat Pract Exper. 2024; 36(16):e8117. doi: [10.1002/cpe.8117](https://doi.org/10.1002/cpe.8117)

If in addition, you want to cite waLBerla, please check in [waLBerla's README file](https://i10git.cs.fau.de/walberla/walberla) for references.

# License
waLBerla-wind is licensed under [GPLv3](COPYING.txt).
