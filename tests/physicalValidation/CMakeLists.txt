add_test(NAME physicalValidation_NeutralABL_Andren
    COMMAND bash run.sh $<TARGET_FILE:TurbineLBM_CPU>
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/NeutralABL_Andren)
set_tests_properties(physicalValidation_NeutralABL_Andren PROPERTIES LABELS "walberlawind::physicalValidation::CPU")
