
waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( airfoil* )
waLBerla_link_files_to_builddir( *.blade )
waLBerla_link_files_to_builddir( *.tower )
waLBerla_link_files_to_builddir( *.ctrl )
waLBerla_link_files_to_builddir( *.wd )

waLBerla_generate_target_from_python(
        NAME TurbineApp_cpu_uniform_Generation
        FILE LBMGeneration.py
        OUT_FILES
        waLBerlaWind_KernelInfo.h
        waLBerlaWind_SweepCollection.cpp waLBerlaWind_SweepCollection.h
        waLBerlaWind_StorageSpecification.cpp waLBerlaWind_StorageSpecification.h
        waLBerlaWind_BoundaryCollection.h
        waLBerlaWind_NoSlip.cpp waLBerlaWind_NoSlip.h
        waLBerlaWind_FreeSlip.cpp waLBerlaWind_FreeSlip.h
        waLBerlaWind_WFB.cpp waLBerlaWind_WFB.h
        waLBerlaWind_UniformUBB.cpp waLBerlaWind_UniformUBB.h
        waLBerlaWind_LogLawUBB.cpp waLBerlaWind_LogLawUBB.h
        waLBerlaWind_Outflow.cpp waLBerlaWind_Outflow.h
        waLBerlaWind_DensityPackInfo.cpp waLBerlaWind_DensityPackInfo.h
        waLBerlaWind_VelocityPackInfo.cpp waLBerlaWind_VelocityPackInfo.h
        waLBerlaWind_WelfordOutput.cpp waLBerlaWind_WelfordOutput.h
        waLBerlaWind_SoSResetter.cpp waLBerlaWind_SoSResetter.h
        waLBerlaWind_SoCResetter.cpp waLBerlaWind_SoCResetter.h
        waLBerlaWind_WelfordEddyViscosity.cpp waLBerlaWind_WelfordEddyViscosity.h
        waLBerlaWind_PdfSetter.cpp waLBerlaWind_PdfSetter.h
        waLBerlaWind_MacroGetter.cpp waLBerlaWind_MacroGetter.h
        waLBerlaWind_StrainRateWriter.cpp waLBerlaWind_StrainRateWriter.h
        waLBerlaWind_WelfordStrainRate.cpp waLBerlaWind_WelfordStrainRate.h
        FlowDriverCollection.h
        Drivers.h Drivers.cpp
)

turbine_core_codegen_dependencies()

add_executable( TurbineApp_cpu_uniform main.cpp )
target_link_libraries( TurbineApp_cpu_uniform
        PRIVATE walberla::core walberla::blockforest walberla::domain_decomposition walberla::field walberla::lbm
                walberla::timeloop 
                TurbineApp_cpu_uniform_Generation
                domain component turbine_topology force_model
                wind_turbine_core

)