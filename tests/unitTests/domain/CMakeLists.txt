############################
### Boundary setup tests ###
############################

waLBerla_add_test_executable( BoundarySetupPeriodicityTest BoundarySetupPeriodicityTest.cpp)
target_link_libraries( BoundarySetupPeriodicityTest walberla::core )
waLBerla_execute_test( NAME BoundarySetupPeriodicityTest )

waLBerla_add_test_executable( BoundarySetupFieldFillerTest BoundarySetupFieldFillerTest.cpp)
target_link_libraries( BoundarySetupFieldFillerTest walberla::boundary walberla::core walberla::domain_decomposition walberla::field walberla::geometry )
waLBerla_execute_test( NAME BoundarySetupFieldFillerTest )

##########################
### Domain setup tests ###
##########################

waLBerla_add_test_executable( DomainSetupDomainDecompositionTest DomainSetupDomainDecompositionTest.cpp)
target_link_libraries( DomainSetupDomainDecompositionTest walberla::blockforest walberla::core walberla::domain_decomposition )
waLBerla_execute_test( NAME DomainSetupDomainDecompositionTest )
