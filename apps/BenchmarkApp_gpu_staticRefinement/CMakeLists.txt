
waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( airfoil_data )
waLBerla_link_files_to_builddir( *.blade )
waLBerla_link_files_to_builddir( *.tower )
waLBerla_link_files_to_builddir( *.ctrl )
waLBerla_link_files_to_builddir( *.wd )

if(WALBERLA_BUILD_WITH_HIP)
  set(source_extension hip)
  set_source_files_properties(main.cu PROPERTIES LANGUAGE HIP)
elseif(WALBERLA_BUILD_WITH_CUDA)
  set(source_extension cu)
endif()

waLBerla_generate_target_from_python(
        NAME BenchmarkApp_gpu_static_refinement_Generation
        FILE LBMGeneration.py
        OUT_FILES
        waLBerlaWind_KernelInfo.h
        waLBerlaWind_SweepCollection.${source_extension} waLBerlaWind_SweepCollection.h
        waLBerlaWind_StorageSpecification.${source_extension} waLBerlaWind_StorageSpecification.h
        waLBerlaWind_BoundaryCollection.h
        waLBerlaWind_NoSlip.${source_extension} waLBerlaWind_NoSlip.h
        waLBerlaWind_FreeSlip.${source_extension} waLBerlaWind_FreeSlip.h
        waLBerlaWind_WFB.${source_extension} waLBerlaWind_WFB.h
        waLBerlaWind_UniformUBB.${source_extension} waLBerlaWind_UniformUBB.h
        waLBerlaWind_LogLawUBB.${source_extension} waLBerlaWind_LogLawUBB.h
        waLBerlaWind_Outflow.${source_extension} waLBerlaWind_Outflow.h
        waLBerlaWind_DensityPackInfo.${source_extension} waLBerlaWind_DensityPackInfo.h
        waLBerlaWind_VelocityPackInfo.${source_extension} waLBerlaWind_VelocityPackInfo.h
        waLBerlaWind_PdfSetter.cpp waLBerlaWind_PdfSetter.h
        waLBerlaWind_MacroGetter_EVEN.cpp waLBerlaWind_MacroGetter_EVEN.h
        waLBerlaWind_MacroGetter_ODD.cpp waLBerlaWind_MacroGetter_ODD.h
        FlowDriverCollection.h
        Drivers.h Drivers.${source_extension}
)

turbine_core_codegen_dependencies()

add_executable( BenchmarkApp_gpu_static_refinement main.cu )
target_link_libraries( BenchmarkApp_gpu_static_refinement
        PRIVATE walberla::core walberla::blockforest walberla::domain_decomposition walberla::field walberla::lbm
                walberla::timeloop walberla::gpu
                BenchmarkApp_gpu_static_refinement_Generation
                domain component turbine_topology force_model
                wind_turbine_core

)