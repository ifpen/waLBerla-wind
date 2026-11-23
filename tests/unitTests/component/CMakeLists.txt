
#####################
### ComponentType ###
#####################

waLBerla_add_test_executable( ComponentTypeTest ComponentTypeTest.cpp)
target_link_libraries( ComponentTypeTest component )
waLBerla_execute_test( NAME ComponentTypeTest )