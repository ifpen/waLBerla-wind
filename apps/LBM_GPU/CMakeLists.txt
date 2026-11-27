
waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( airfoil_data )
waLBerla_link_files_to_builddir( *.blade )
waLBerla_link_files_to_builddir( *.tower )
waLBerla_link_files_to_builddir( *.ctrl )
waLBerla_link_files_to_builddir( *.wd )

if(WALBERLA_BUILD_WITH_HIP)
  set(generate_extension hip)
  set(gpu_extension hip)
elseif(WALBERLA_BUILD_WITH_CUDA)
  set(generate_extension cu)
  set(gpu_extension cu)
endif()

waLBerla_generate_target_from_python(
        NAME WindTurbineCustomLatticeModelGPU_generated
        FILE WindTurbineLBMGeneration.py
        OUT_FILES
        waLBerlaWind_KernelInfo.h
        waLBerlaWind_CollectionSweepCollection.${generate_extension} waLBerlaWind_CollectionSweepCollection.h
        waLBerlaWind_CollectionStorageSpecification.${generate_extension} waLBerlaWind_CollectionStorageSpecification.h
        waLBerlaWind_CollectionBoundaryCollection.h
        waLBerlaWind_NoSlip.${generate_extension} waLBerlaWind_NoSlip.h
        waLBerlaWind_FreeSlip.${generate_extension} waLBerlaWind_FreeSlip.h
        waLBerlaWind_WFB.${generate_extension} waLBerlaWind_WFB.h
        waLBerlaWind_UniformUBB.${generate_extension} waLBerlaWind_UniformUBB.h
        waLBerlaWind_LogLawUBB.${generate_extension} waLBerlaWind_LogLawUBB.h
        waLBerlaWind_Outflow.${generate_extension} waLBerlaWind_Outflow.h
        waLBerlaWind_SoSResetter.${generate_extension} waLBerlaWind_SoSResetter.h
        waLBerlaWind_WelfordVelocity.${generate_extension} waLBerlaWind_WelfordVelocity.h
        waLBerlaWind_MacroSetter.cpp waLBerlaWind_MacroSetter.h
        waLBerlaWind_MacroGetter.cpp waLBerlaWind_MacroGetter.h
        waLBerlaWind_PdfPackInfo.${generate_extension} waLBerlaWind_PdfPackInfo.h
        waLBerlaWind_EddyViscosityWriter.${generate_extension} waLBerlaWind_EddyViscosityWriter.h
        waLBerlaWind_StrainRateWriter.${generate_extension} waLBerlaWind_StrainRateWriter.h
        FlowDriverCollection.h
        Drivers.h Drivers.${generate_extension}
)

turbine_core_codegen_dependencies()

add_executable( TurbineLBM_GPU main.${gpu_extension} )
target_link_libraries( TurbineLBM_GPU
        PRIVATE walberla::core walberla::blockforest walberla::domain_decomposition walberla::field walberla::gpu
                walberla::lbm walberla::lbm_generated walberla::timeloop
		WindTurbineCustomLatticeModelGPU_generated
                domain component external_forces turbine_topology force_model
                wind_turbine_core walberla_helper
)
