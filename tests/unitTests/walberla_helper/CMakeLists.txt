#################
### BlockInfo ###
#################

waLBerla_add_test_executable( BlockInfoTest blockforest/BlockInfoTest.cpp)
target_link_libraries( BlockInfoTest walberla::core walberla::blockforest )
waLBerla_execute_test( NAME BlockInfoTest )


#############
### Field ###
#############

waLBerla_add_test_executable( TurbineCoreFieldTest field/TurbineCoreFieldTest.cpp)
target_link_libraries( TurbineCoreFieldTest walberla::core walberla::field walberla_helper )
waLBerla_execute_test( NAME TurbineCoreFieldTest )

########################################
### Macroscopic Variable Calculation ###
########################################

waLBerla_add_test_executable( TurbineCoreMacroscopicVariableCalculatorTest field/TurbineCoreMacroscopicVariableCalculatorTest.cpp)
target_link_libraries( TurbineCoreMacroscopicVariableCalculatorTest walberla::core walberla::field walberla_helper )
waLBerla_execute_test( NAME TurbineCoreMacroscopicVariableCalculatorTest )

#####################
### Interpolators ###
#####################

waLBerla_add_test_executable( TurbineCoreFieldInterpolationTest field/interpolators/TurbineCoreFieldInterpolationTest.cpp)
target_link_libraries( TurbineCoreFieldInterpolationTest walberla::core walberla::field walberla_helper)
waLBerla_execute_test( NAME TurbineCoreFieldInterpolationTest )

waLBerla_add_test_executable( TurbineCoreMacroscopicFieldInterpolationTest field/interpolators/TurbineCoreMacroscopicFieldInterpolationTest.cpp)
target_link_libraries( TurbineCoreMacroscopicFieldInterpolationTest walberla::core walberla::field walberla_helper)
waLBerla_execute_test( NAME TurbineCoreMacroscopicFieldInterpolationTest )

####################
### Distributors ###
####################

waLBerla_add_test_executable( TurbineCoreFieldDistributorTest field/distributors/TurbineCoreFieldDistributorTest.cpp)
target_link_libraries( TurbineCoreFieldDistributorTest walberla::core walberla::field walberla_helper)
waLBerla_execute_test( NAME TurbineCoreFieldDistributorTest )