
#include <core/Environment.h>
#include <core/debug/TestSubsystem.h>

#include "domain/DomainSetup.h"

namespace turbine_core::domain_setup_domain_decomposition {

    void testMinimalNBlocks( const domain::DomainSetup & setup ) {

        // at least one block per dimension
        const auto nBlocks = setup.numberOfCoarseBlocksPerDirection_;
        for( uint_t d = 0; d < 3; ++d ) {
            WALBERLA_ASSERT_GREATER_EQUAL(nBlocks[d], uint_t(1), "No blocks in direction " << d << ".")
        }

    }

    void testDomainAdjustment( const domain::DomainSetup & setup,
                               const walberla::Vector3<uint_t> & initialCellsPerBlock,
                               const walberla::Vector3<uint_t> & initialDomainSize ) {

        // check that domain did not increase/decrease too much
        for( uint_t d = 0; d < 3; ++d ) {
            const auto diff = std::abs(int(setup.domainSize_[d]) - int(initialDomainSize[d]));
            WALBERLA_ASSERT_LESS( diff, initialCellsPerBlock[d] * (1<<setup.numLevels_), "Domain size in direction " << d << " was changed more than initial block size in this direction." )
        }

        // if parameters already fit, do not change domain
        for( uint_t d = 0; d < 3; ++d ) {

            if( initialDomainSize[d] % initialCellsPerBlock[d] == 0 ) {
                WALBERLA_ASSERT_EQUAL(initialDomainSize[d], setup.domainSize_[d], "Domain size in direction " << d << " changed even though parameters fit perfectly.")
            }

        }

    }

    void testFixedCellsPerBlock( const domain::DomainSetup & setup, const walberla::Vector3<uint_t> & initialCellsPerBlock ) {

        WALBERLA_ASSERT_EQUAL( setup.fixedCellsPerBlock_, true )

        for( uint_t d = 0; d < 3; ++d ) {
            WALBERLA_ASSERT_EQUAL( initialCellsPerBlock[d], setup.cellsPerBlock_[d], "CellsPerBlock in direction " << d << " changed for fixed cells." )
        }

    }

    void testVariableCellsPerBlock( const domain::DomainSetup & setup, const walberla::Vector3<uint_t> & maxCellsPerBlock ) {

        WALBERLA_ASSERT_EQUAL( setup.fixedCellsPerBlock_, false )
        //TODO how to test properly?!

    }

    int main( int argc, char ** argv ) {

        walberla::mpi::Environment mpiEnv(argc, argv);
        walberla::debug::enterTestMode();

        /// TESTS 0 - domainSize < cellsPerBlock

        { // fixed cells per block
            auto globalConfig = std::make_shared<walberla::Config>();
            auto globalBlock = globalConfig->getWritableGlobalBlock();
            auto configTest = globalBlock.createBlock("DomainSetup");
            configTest.addParameter("domainSize", "<1,1,1>");
            configTest.addParameter("cellsPerBlock", "<64,64,64>");
            configTest.addParameter("domainAdjustmentThreshold", "0");
            configTest.addParameter("numLevels", "0");

            domain::DomainSetup setup { globalConfig, walberla::Vector3<bool>() };
            setup.computeDomainDecomposition();

            testMinimalNBlocks(setup);
            testDomainAdjustment(setup,walberla::Vector3<uint_t>(64,64,64),walberla::Vector3<uint_t>(1,1,1));
            testFixedCellsPerBlock(setup, walberla::Vector3<uint_t>(64,64,64));
        }

        // THIS TEST CASE IS SUPPOSED TO FAIL DUE TO BAD USER INPUT -> do not execute!!
        /* { // variable cells per block
            walberla::Config::Block configTest("DomainSetup");
            configTest.addParameter("domainSize", "<1,1,1>");
            configTest.addParameter("maxCellsPerBlock", "<64,64,64>");
            configTest.addParameter("domainAdjustmentThreshold", "0");
            configTest.addParameter("numLevels", "0");

            domain::DomainSetup setup { walberla::Config::BlockHandle(&configTest), walberla::Vector3<bool>() };
            setup.computeDomainDecomposition();

            testMinimalNBlocks(setup);
            testDomainAdjustment(setup,walberla::Vector3<uint_t>(64,64,64),walberla::Vector3<uint_t>(1,1,1));
            testVariableCellsPerBlock(setup, walberla::Vector3<uint_t>(64,64,64));
        } */

        const std::vector<uint_t> cellsPerBlock{16, 32, 64};
        const std::vector<uint_t> domainSize{250, 118, 65};

        /// TEST SERIES 1 - fixed cells per block

        for( uint_t i = 0; i < 3; ++i ) {

            const uint_t x = i, y = (i+1)%3, z = (i+2)%3;

            for( const auto & dat : {"0.0", "0.33333", "0.66666", "1.0"} ) {

                for( const auto & level : {"0", "1", "3"} ) {

                    auto globalConfig = std::make_shared<walberla::Config>();
                    auto globalBlock = globalConfig->getWritableGlobalBlock();
                    auto configTest = globalBlock.createBlock("DomainSetup");

                    std::ostringstream domainString, cellString;
                    domainString << "<" << domainSize[x] << "," << domainSize[y] << "," << domainSize[z] << ">";
                    cellString   << "<" << cellsPerBlock[x] << "," << cellsPerBlock[y] << "," << cellsPerBlock[z] << ">";

                    configTest.addParameter("domainSize", domainString.str());
                    configTest.addParameter("cellsPerBlock", cellString.str());
                    configTest.addParameter("domainAdjustmentThreshold", dat);
                    configTest.addParameter("numLevels", level);

                    domain::DomainSetup setup1{globalConfig,walberla::Vector3<bool>()};
                    setup1.computeDomainDecomposition();

                    testMinimalNBlocks(setup1);
                    testDomainAdjustment(setup1,
                                         walberla::Vector3<uint_t>(cellsPerBlock[x], cellsPerBlock[y], cellsPerBlock[z]),
                                         walberla::Vector3<uint_t>(domainSize[x], domainSize[y], domainSize[z]));
                    testFixedCellsPerBlock(setup1, walberla::Vector3<uint_t>(cellsPerBlock[x], cellsPerBlock[y], cellsPerBlock[z]));

                }
            }

        }


        /// TEST SERIES 2 - variable cells per block

        for( uint_t i = 0; i < 3; ++i ) {

            const uint_t x = i, y = (i+1)%3, z = (i+2)%3;

            for( const auto & dat : {"0.0", "0.33333", "0.66666", "1.0"} ) {

                for( const auto & level : {"0", "1", "3"} ) {

                    auto globalConfig = std::make_shared<walberla::Config>();
                    auto globalBlock = globalConfig->getWritableGlobalBlock();
                    auto configTest = globalBlock.createBlock("DomainSetup");

                    std::ostringstream domainString, cellString;
                    domainString << "<" << domainSize[x] << "," << domainSize[y] << "," << domainSize[z] << ">";
                    cellString   << "<" << cellsPerBlock[x] << "," << cellsPerBlock[y] << "," << cellsPerBlock[z] << ">";

                    configTest.addParameter("domainSize", domainString.str());
                    configTest.addParameter("maxCellsPerBlock", cellString.str());
                    configTest.addParameter("domainAdjustmentThreshold", dat);
                    configTest.addParameter("numLevels", level);

                    domain::DomainSetup setup1{globalConfig,walberla::Vector3<bool>()};
                    setup1.computeDomainDecomposition();

                    testMinimalNBlocks(setup1);
                    testDomainAdjustment(setup1,
                                         walberla::Vector3<uint_t>(cellsPerBlock[x], cellsPerBlock[y], cellsPerBlock[z]),
                                         walberla::Vector3<uint_t>(domainSize[x], domainSize[y], domainSize[z]));
                    testVariableCellsPerBlock(setup1, walberla::Vector3<uint_t>(cellsPerBlock[x], cellsPerBlock[y], cellsPerBlock[z]));

                }
            }

        }

        return EXIT_SUCCESS;

    }

} // namespace turbine_core::domain_setup_domain_decomposition

int main( int argc, char ** argv ) {
    return turbine_core::domain_setup_domain_decomposition::main(argc, argv);
}

