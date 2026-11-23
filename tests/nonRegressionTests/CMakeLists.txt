add_test(NAME nonRegressionTests_LBM_CPU
    COMMAND bash run.sh $<TARGET_FILE:TurbineLBM_CPU>
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/LBM_CPU_test)
set_tests_properties(nonRegressionTests_LBM_CPU PROPERTIES LABELS "walberlawind::nonRegressionTests::CPU")

add_test(NAME nonRegressionTests_LBM_CPU_CP
    COMMAND bash run.sh $<TARGET_FILE:TurbineLBM_CPU_CP>
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/LBM_CPU_CP_test)
set_tests_properties(nonRegressionTests_LBM_CPU_CP PROPERTIES LABELS "walberlawind::nonRegressionTests::CPU")

if(WALBERLA_BUILD_WITH_CUDA)
    add_test(NAME nonRegressionTests_LBM_GPU
        COMMAND bash run.sh $<TARGET_FILE:TurbineLBM_GPU>
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/LBM_GPU_test)
    set_tests_properties(nonRegressionTests_LBM_GPU PROPERTIES LABELS "walberlawind::nonRegressionTests::GPU")

    add_test(NAME nonRegressionTests_LBM_GPU_CP
        COMMAND bash run.sh $<TARGET_FILE:TurbineLBM_GPU_CP>
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/LBM_GPU_CP_test)
    set_tests_properties(nonRegressionTests_LBM_GPU_CP PROPERTIES LABELS "walberlawind::nonRegressionTests::GPU")
endif()
