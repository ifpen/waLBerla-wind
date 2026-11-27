
waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( airfoil* )
waLBerla_link_files_to_builddir( *.blade )
waLBerla_link_files_to_builddir( *.tower )
waLBerla_link_files_to_builddir( *.ctrl )
waLBerla_link_files_to_builddir( *.wd )

waLBerla_generate_target_from_python(
        NAME TurbineApp_gpu_static_refinement_Generation
        FILE LBMGeneration.py
        OUT_FILES
        waLBerlaWind_KernelInfo.h
        waLBerlaWind_SweepCollection.cu waLBerlaWind_SweepCollection.h
        waLBerlaWind_StorageSpecification.cu waLBerlaWind_StorageSpecification.h
        waLBerlaWind_BoundaryCollection.h
        waLBerlaWind_NoSlip.cu waLBerlaWind_NoSlip.h
        waLBerlaWind_FreeSlip.cu waLBerlaWind_FreeSlip.h
        waLBerlaWind_WFB.cu waLBerlaWind_WFB.h
        waLBerlaWind_UniformUBB.cu waLBerlaWind_UniformUBB.h
        waLBerlaWind_LogLawUBB.cu waLBerlaWind_LogLawUBB.h
        waLBerlaWind_Outflow.cu waLBerlaWind_Outflow.h
        waLBerlaWind_DensityPackInfo.cu waLBerlaWind_DensityPackInfo.h
        waLBerlaWind_VelocityPackInfo.cu waLBerlaWind_VelocityPackInfo.h
        waLBerlaWind_WelfordOutput.cu waLBerlaWind_WelfordOutput.h
        waLBerlaWind_SoSResetter.cu waLBerlaWind_SoSResetter.h
        waLBerlaWind_SoCResetter.cu waLBerlaWind_SoCResetter.h
        waLBerlaWind_WelfordEddyViscosity.cu waLBerlaWind_WelfordEddyViscosity.h
        waLBerlaWind_PdfSetter.cpp waLBerlaWind_PdfSetter.h
        waLBerlaWind_MacroGetter.cpp waLBerlaWind_MacroGetter.h
        waLBerlaWind_StrainRateWriter.cu waLBerlaWind_StrainRateWriter.h
        waLBerlaWind_WelfordStrainRate.cu waLBerlaWind_WelfordStrainRate.h
        FlowDriverCollection.h
        Drivers.h Drivers.cu
)

turbine_core_codegen_dependencies()

add_executable( TurbineApp_gpu_static_refinement main.cu )
target_link_libraries( TurbineApp_gpu_static_refinement
        PRIVATE walberla::core walberla::blockforest walberla::domain_decomposition walberla::field walberla::lbm
                walberla::timeloop walberla::gpu
                TurbineApp_gpu_static_refinement_Generation
                domain component turbine_topology force_model
                wind_turbine_core

)