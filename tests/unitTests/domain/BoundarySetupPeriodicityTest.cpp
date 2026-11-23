
#include <core/Environment.h>
#include <core/debug/TestSubsystem.h>

#include "domain/BoundarySetup.h"

namespace turbine_core::boundary_setup_periodicity_test {

    void testPeriodicity( const EnvironmentSetup::Type environmentType,
                          const InflowSetup::Type inflowType,
                          const OutflowSetup::Type outflowType ) {

        /// setup config for boundary setup

        walberla::Config::Block boundaryBlock("Boundaries");

        // setups
        boundaryBlock.addParameter( "setup",       EnvironmentSetup::toString(environmentType));
        boundaryBlock.addParameter( "inflowType",  InflowSetup::toString(inflowType));
        boundaryBlock.addParameter( "outflowType", OutflowSetup::toString(outflowType));
        // dummy inflow/outflow
        boundaryBlock.addParameter( "inflowVelocity", "<0.05,0.0,0.0>" );
        boundaryBlock.addParameter( "outflowPressure", "1.0" );

        const domain::BoundarySetup boundarySetup( walberla::Config::BlockHandle{&boundaryBlock} );

        const auto periodicity = boundarySetup.periodicity();

        switch (boundarySetup.environmentSetup()) {
            case EnvironmentSetup::Periodic :
                WALBERLA_ASSERT_EQUAL(periodicity, walberla::Vector3<bool>(1,1,1), "Periodic setup did not yield periodic domain.")
                break;
            default:
                //TODO what about other y direction?
                if( boundarySetup.inflowType() == InflowSetup::Periodic ) {
                    WALBERLA_ASSERT_EQUAL(periodicity[0], 1, "Periodicity in x-direction not set despite of periodic inflow/outflow.")
                    WALBERLA_ASSERT_EQUAL(periodicity[2], 0, "Periodicity in z-direction set despite of non-periodic setup.")
                }
        }
    }


    int main( int argc, char** argv ) {

        walberla::mpi::Environment mpiEnv(argc, argv);
        walberla::debug::enterTestMode();

        // fully periodic -> needs to work even with false inflow/outflow input
        testPeriodicity( EnvironmentSetup::Periodic, InflowSetup::InflowUniform, OutflowSetup::Outflow );

        // open with periodic inflow/outflow
        testPeriodicity(EnvironmentSetup::Open, InflowSetup::Periodic, OutflowSetup::Periodic );

        // open with non-periodic inflow/outflow
        testPeriodicity(EnvironmentSetup::Open, InflowSetup::InflowUniform, OutflowSetup::Outflow);

        // tunnel with periodic inflow/outflow
        testPeriodicity(EnvironmentSetup::Tunnel, InflowSetup::Periodic, OutflowSetup::Periodic );

        // tunnel with non-periodic inflow/outflow
        testPeriodicity(EnvironmentSetup::Tunnel, InflowSetup::InflowUniform, OutflowSetup::Outflow );

        return EXIT_SUCCESS;

    }


} // namespace turbine_core::boundary_setup_periodicity_test

int main( int argc, char ** argv ) {
    return turbine_core::boundary_setup_periodicity_test::main(argc, argv);
}