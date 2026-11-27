
waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( airfoil_data )
waLBerla_link_files_to_builddir( *.blade )
waLBerla_link_files_to_builddir( *.tower )
waLBerla_link_files_to_builddir( *.ctrl )
waLBerla_link_files_to_builddir( *.wd )

waLBerla_generate_target_from_python(
        NAME WindTurbineCustomLatticeModel_generated
        FILE WindTurbineLBMGeneration.py
        OUT_FILES
        waLBerlaWind_KernelInfo.h
        waLBerlaWind_CollectionSweepCollection.cpp waLBerlaWind_CollectionSweepCollection.h
        waLBerlaWind_CollectionStorageSpecification.cpp waLBerlaWind_CollectionStorageSpecification.h
        waLBerlaWind_CollectionBoundaryCollection.h
        waLBerlaWind_NoSlip.cpp waLBerlaWind_NoSlip.h
        waLBerlaWind_FreeSlip.cpp waLBerlaWind_FreeSlip.h
        waLBerlaWind_WFB.cpp waLBerlaWind_WFB.h
        waLBerlaWind_UniformUBB.cpp waLBerlaWind_UniformUBB.h
        waLBerlaWind_LogLawUBB.cpp waLBerlaWind_LogLawUBB.h
        waLBerlaWind_Outflow.cpp waLBerlaWind_Outflow.h
        waLBerlaWind_WelfordVelocity.cpp waLBerlaWind_WelfordVelocity.h
        waLBerlaWind_SoSResetter.cpp waLBerlaWind_SoSResetter.h
        waLBerlaWind_MacroSetter.cpp waLBerlaWind_MacroSetter.h
        waLBerlaWind_MacroGetter.cpp waLBerlaWind_MacroGetter.h
        waLBerlaWind_EddyViscosityWriter.cpp waLBerlaWind_EddyViscosityWriter.h
        waLBerlaWind_StrainRateWriter.cpp waLBerlaWind_StrainRateWriter.h
        FlowDriverCollection.h
        Drivers.h Drivers.cpp
)

turbine_core_codegen_dependencies()

add_executable( TurbineLBM_CPU main.cpp )

target_link_libraries(
    TurbineLBM_CPU
        PRIVATE walberla::core walberla::blockforest walberla::domain_decomposition walberla::field walberla::lbm
                walberla::timeloop
                WindTurbineCustomLatticeModel_generated
                domain component turbine_topology force_model
                wind_turbine_core
)
