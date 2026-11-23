
#include <core/Environment.h>
#include <core/debug/TestSubsystem.h>
#include <core/config/Config.h>
#include <stencil/D3Q27.h>
#include <blockforest/Initialization.h>
#include <field/AddToStorage.h>
#include <lbm/all.h>

#include "domain/EnvironmentSetup.h"
#include "domain/BoundarySetup.h"
#include "domain/BoundaryHandling.h"
#include "wind_turbine_core/WalberlaDataTypes.h"

namespace turbine_core::boundary_setup_field_filler_test {

    using flag_t = walberla::uint8_t;
    using FlagField_T = walberla::FlagField<flag_t>;

    constexpr uint_t nGhostLayer{1};

    using CollisionModel_T = walberla::lbm::collision_model::SRT;
    using LatticeModel_T = walberla::lbm::D3Q27<CollisionModel_T,true>;
    constexpr real_t omega{1.8};

    using PdfField_T = walberla::lbm::PdfField<LatticeModel_T>;

    using Stencil_T = walberla::stencil::D3Q27;

    using NoSlip_T = walberla::lbm::NoSlip< LatticeModel_T, flag_t >;
    using FreeSlip_T = walberla::lbm::FreeSlip< LatticeModel_T, FlagField_T >;
    using SimpleUBB_T = walberla::lbm::SimpleUBB< LatticeModel_T, flag_t >;
    using SimplePressure_T = walberla::lbm::SimplePressure< LatticeModel_T, flag_t >;

    void testConfig( const walberla::Config::BlockHandle & config, const std::shared_ptr<walberla::StructuredBlockForest> & blocks, const BlockDataID & pdfFieldID,
                     const walberla::FlagUID FluidFlagUID, const walberla::FlagUID NoSlipFlagUID, const walberla::FlagUID WFBFlagUID, const walberla::FlagUID SymmetryFlagUID,
                     const walberla::FlagUID UniformInflowFlagUID, const walberla::FlagUID LogLawFlagUID, const walberla::FlagUID OutflowFlagUID ) {

        /// set up and fill flag fields

        BlockDataID configFlagFieldID = walberla::field::addFlagFieldToStorage<FlagField_T>(blocks, "flag field config", nGhostLayer);
        BlockDataID setterFlagFieldID = walberla::field::addFlagFieldToStorage<FlagField_T>(blocks, "flag field setter", nGhostLayer);

        const domain::BoundarySetup boundarySetup{config};

        using BH = boundary::DummyBoundaryHandling<Stencil_T, PdfField_T, FlagField_T, NoSlip_T>;
        BlockDataID boundaryHandlingID = blocks->addBlockData( std::make_shared< BH >(setterFlagFieldID, pdfFieldID ),
                                                               "boundary handling" );

        boundarySetup.fillFlagFieldFromConfig<FlagField_T>(blocks, configFlagFieldID, FluidFlagUID, NoSlipFlagUID, WFBFlagUID, SymmetryFlagUID, UniformInflowFlagUID,
            LogLawFlagUID, OutflowFlagUID);
        auto setter = boundarySetup.fillFlagFieldFromSetter<BH::BoundaryHandling>(blocks, boundaryHandlingID, nGhostLayer, FluidFlagUID, NoSlipFlagUID, WFBFlagUID, SymmetryFlagUID, UniformInflowFlagUID,
            LogLawFlagUID, OutflowFlagUID);

        /// test equality of fields -> assume config one is correct as waLBerla only functionalities

        for (auto & block : *blocks) {

            auto * configFlagField = block.getData<FlagField_T>(configFlagFieldID);
            auto * setterFlagField = block.getData<FlagField_T>(setterFlagFieldID);

            WALBERLA_ASSERT_EQUAL(configFlagField->xyzSizeWithGhostLayer(), setterFlagField->xyzSizeWithGhostLayer(), "Flag field must be of same size.")

            const auto ci = configFlagField->xyzSizeWithGhostLayer();

            for (const auto & cell : ci) {

                auto configFluidFlag = configFlagField->getOrRegisterFlag(FluidFlagUID);
                auto configNoSlipFlag = configFlagField->getOrRegisterFlag(NoSlipFlagUID);
                auto configFreeSlipFlag = configFlagField->getOrRegisterFlag(SymmetryFlagUID);

                auto setterFluidFlag = setterFlagField->getOrRegisterFlag(FluidFlagUID);
                auto setterNoSlipFlag = setterFlagField->getOrRegisterFlag(NoSlipFlagUID);
                auto setterFreeSlipFlag = setterFlagField->getOrRegisterFlag(SymmetryFlagUID);

                const std::string debugString{"In cell [" + std::to_string(cell.x()) + "," + std::to_string(cell.y()) + "," + std::to_string(cell.z()) + "] for setup " + EnvironmentSetup::toString(boundarySetup.environmentSetup()) + " (inflow: " + InflowSetup::toString(boundarySetup.inflowType()) + ", outflow: " + OutflowSetup::toString(boundarySetup.outflowType()) + ")"};

                WALBERLA_ASSERT( !(configFlagField->isFlagSet(cell, configFluidFlag) ^ setterFlagField->isFlagSet(cell, setterFluidFlag)),
                                 "Fluid flag inconsistency : config(" << configFlagField->isFlagSet(cell, configFluidFlag) << ") vs. Setter(" <<
                                                                         setterFlagField->isFlagSet(cell, setterFluidFlag) << ")\n" << debugString )
                WALBERLA_ASSERT( !(configFlagField->isFlagSet(cell, configNoSlipFlag) ^ setterFlagField->isFlagSet(cell, setterNoSlipFlag)),
                                 "NoSlip flag inconsistency : config(" << configFlagField->isFlagSet(cell, configNoSlipFlag) << ") vs. Setter(" <<
                                                                      setterFlagField->isFlagSet(cell, setterNoSlipFlag) << ")\n" << debugString )
                WALBERLA_ASSERT( !(configFlagField->isFlagSet(cell, configFreeSlipFlag) ^ setterFlagField->isFlagSet(cell, setterFreeSlipFlag)),
                                 "FreeSlip flag inconsistency : config(" << configFlagField->isFlagSet(cell, configFreeSlipFlag) << ") vs. Setter(" <<
                                                                      setterFlagField->isFlagSet(cell, setterFreeSlipFlag) << ")\n" << debugString )

            }

        }

        // delete data and add in new test to ensure empty flag field in the beginning
        blocks->clearBlockData(configFlagFieldID);
        blocks->clearBlockData(setterFlagFieldID);
        blocks->clearBlockData(boundaryHandlingID);

    }

    int main( int argc, char ** argv ) {

        walberla::mpi::Environment mpiEnv(argc, argv);
        walberla::debug::enterTestMode();

        /// set up test domain

        const uint_t numberOfBlocksInDirection = 2;
        const uint_t numberOfCellsPerBlockInDirection = 4;
        const real_t dx{1};

        // block storage
        auto blocks = walberla::blockforest::createUniformBlockGrid( numberOfBlocksInDirection, numberOfBlocksInDirection, numberOfBlocksInDirection,
                                                                     numberOfCellsPerBlockInDirection, numberOfCellsPerBlockInDirection, numberOfCellsPerBlockInDirection,
                                                                     dx, 0, false, false,
                                                                     false, false, false,
                                                                     false );

        LatticeModel_T latticeModel{omega};
        BlockDataID pdfFieldID = walberla::lbm::addPdfFieldToStorage(blocks, "pdf field", latticeModel);

        /// create flags
        const walberla::FlagUID FluidFlagUID("fluid");
        const walberla::FlagUID NoSlipFlagUID("no_slip");
        const walberla::FlagUID WFBFlagUID("no_slip");
        const walberla::FlagUID SymmetryFlagUID("no_slip");
        const walberla::FlagUID UniformInflowFlagUID("no_slip");
        const walberla::FlagUID LogLawInflowFlagUID("no_slip");
        const walberla::FlagUID OutflowFlagUID("no_slip");


        /// test periodic setup

        walberla::Config::Block boundaryBlockPeriodic("Boundaries");
        {
            boundaryBlockPeriodic.addParameter("setup", EnvironmentSetup::toString(EnvironmentSetup::Periodic));
            boundaryBlockPeriodic.addParameter("inflowType", InflowSetup::toString(InflowSetup::Periodic));
            boundaryBlockPeriodic.addParameter("outflowType", OutflowSetup::toString(OutflowSetup::Periodic));
        }

        testConfig( walberla::Config::BlockHandle{&boundaryBlockPeriodic}, blocks, pdfFieldID,
            FluidFlagUID, NoSlipFlagUID, WFBFlagUID, SymmetryFlagUID, UniformInflowFlagUID, LogLawInflowFlagUID, OutflowFlagUID );

        /// test tunnel setup with periodic inflow/outflow

        walberla::Config::Block boundaryBlockTunnelPeriodic("Boundaries");
        {
            boundaryBlockTunnelPeriodic.addParameter("setup", EnvironmentSetup::toString(EnvironmentSetup::Tunnel));
            boundaryBlockTunnelPeriodic.addParameter("inflowType", InflowSetup::toString(InflowSetup::Periodic));
            boundaryBlockTunnelPeriodic.addParameter("outflowType", OutflowSetup::toString(OutflowSetup::Periodic));
        }

        testConfig( walberla::Config::BlockHandle{&boundaryBlockTunnelPeriodic}, blocks, pdfFieldID,
            FluidFlagUID, NoSlipFlagUID, WFBFlagUID, SymmetryFlagUID, UniformInflowFlagUID, LogLawInflowFlagUID, OutflowFlagUID );

        /// test tunnel setup with non-periodic inflow/outflow

        walberla::Config::Block boundaryBlockTunnelNonPeriodic("Boundaries");
        {
            boundaryBlockTunnelNonPeriodic.addParameter("setup", EnvironmentSetup::toString(EnvironmentSetup::Tunnel));
            boundaryBlockTunnelNonPeriodic.addParameter("inflowType", InflowSetup::toString(InflowSetup::InflowUniform));
            boundaryBlockTunnelNonPeriodic.addParameter("outflowType", OutflowSetup::toString(OutflowSetup::Outflow));
            boundaryBlockTunnelNonPeriodic.addParameter( "inflowVelocity", "<0.05,0.0,0.0>" );
            boundaryBlockTunnelNonPeriodic.addParameter( "outflowPressure", "1.0" );
        }

        testConfig( walberla::Config::BlockHandle{&boundaryBlockTunnelNonPeriodic}, blocks, pdfFieldID,
            FluidFlagUID, NoSlipFlagUID, WFBFlagUID, SymmetryFlagUID, UniformInflowFlagUID, LogLawInflowFlagUID, OutflowFlagUID );


        /// test open setup with periodic inflow/outflow

        walberla::Config::Block boundaryBlockOpenPeriodic("Boundaries");
        {
            boundaryBlockOpenPeriodic.addParameter( "setup", EnvironmentSetup::toString(EnvironmentSetup::Open));
            boundaryBlockOpenPeriodic.addParameter( "inflowType", InflowSetup::toString(InflowSetup::Periodic));
            boundaryBlockOpenPeriodic.addParameter( "outflowType", OutflowSetup::toString(OutflowSetup::Periodic));
        }


        /// test open setup with non-periodic inflow/outflow

        walberla::Config::Block boundaryBlockOpenNonPeriodic("Boundaries");
        {
            boundaryBlockOpenNonPeriodic.addParameter( "setup", EnvironmentSetup::toString(EnvironmentSetup::Open));
            boundaryBlockOpenNonPeriodic.addParameter( "inflowType", InflowSetup::toString(InflowSetup::InflowUniform));
            boundaryBlockOpenNonPeriodic.addParameter( "outflowType", OutflowSetup::toString(OutflowSetup::Outflow));
            boundaryBlockOpenNonPeriodic.addParameter( "inflowVelocity", "<0.05,0.0,0.0>" );
            boundaryBlockOpenNonPeriodic.addParameter( "outflowPressure", "1.0" );
        }

        testConfig( walberla::Config::BlockHandle{&boundaryBlockOpenNonPeriodic}, blocks, pdfFieldID,
            FluidFlagUID, NoSlipFlagUID, WFBFlagUID, SymmetryFlagUID, UniformInflowFlagUID, LogLawInflowFlagUID, OutflowFlagUID );

        return EXIT_SUCCESS;
    }

} // namespace turbine_core::boundary_setup_field_filler_test

int main( int argc, char ** argv ) {
    return turbine_core::boundary_setup_field_filler_test::main(argc,argv);
}