include(${CMAKE_CURRENT_LIST_DIR}/AppGenerator/codegen_dependencies.cmake)

add_subdirectory(AppGenerator EXCLUDE_FROM_ALL)

add_subdirectory(LBM_CPU)
add_subdirectory(LBM_CPU_CP)
add_subdirectory(LBM_CPU_AMR)

add_subdirectory(GeneratedApp_cpu_uniform)
add_subdirectory(GeneratedApp_cpu_staticRefinement)

if ( WALBERLA_BUILD_WITH_GPU_SUPPORT)
add_subdirectory(LBM_GPU)
add_subdirectory(LBM_GPU_CP)

add_subdirectory(GeneratedApp_gpu_uniform)
add_subdirectory(GeneratedApp_gpu_staticRefinement)
add_subdirectory(BenchmarkApp_gpu_staticRefinement)
endif()

