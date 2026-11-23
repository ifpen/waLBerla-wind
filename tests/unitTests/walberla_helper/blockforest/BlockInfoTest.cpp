
#include <cstdlib>
#include <random>

#include <core/Environment.h>
#include <core/debug/TestSubsystem.h>
#include <core/DataTypes.h>

#include <blockforest/Initialization.h>

#include "walberla_helper/blockforest/BlockInfo.h"

namespace turbine_core {

    namespace blockinfo_test {

        using namespace walberla;

        void getBlockLocalCellTest( std::shared_ptr<walberla::blockforest::StructuredBlockForest> & blocks ) {

            auto nCellsX = blocks->getNumberOfXCells();
            auto nCellsY = blocks->getNumberOfYCells();
            auto nCellsZ = blocks->getNumberOfZCells();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> disX(real_t(0), real_t(nCellsX));
            std::uniform_real_distribution<> disY(real_t(0), real_t(nCellsY));
            std::uniform_real_distribution<> disZ(real_t(0), real_t(nCellsZ));

            for( auto it = blocks->begin(); it != blocks->end(); ++it ) {

                blockforest::BlockInfo info{it.get(), blocks};

                auto & block = *(it.get());

                // test random points in domain
                for( uint_t i = 0; i < 100; ++i ) {
                    Vector3<real_t> point{disX(gen), disY(gen), disZ(gen)};

                    auto standardCell = blocks->getBlockLocalCell(block, point[0], point[1], point[2]);
                    auto customCell = info.getBlockLocalCell(point);

                    WALBERLA_ASSERT_EQUAL(standardCell.x(), customCell[0])
                    WALBERLA_ASSERT_EQUAL(standardCell.y(), customCell[1])
                    WALBERLA_ASSERT_EQUAL(standardCell.z(), customCell[2])

                    auto standardCenter = blocks->getBlockLocalCellCenter(block, standardCell);
                    auto customCenter = info.getBlockLocalCellCenter(customCell);

                    WALBERLA_ASSERT_FLOAT_EQUAL(standardCenter[0], customCenter[0])
                    WALBERLA_ASSERT_FLOAT_EQUAL(standardCenter[1], customCenter[1])
                    WALBERLA_ASSERT_FLOAT_EQUAL(standardCenter[2], customCenter[2])
                }
            }

        }

        int main(int argc, char ** argv) {
            Environment walberlaEnv(argc, argv);
            debug::enterTestMode();

            const uint_t numberOfBlocksInDirection = 2;
            const uint_t numberOfCellsPerBlockInDirection = 10;
            const auto dx = real_t(0.4);

            auto blocks = walberla::blockforest::createUniformBlockGrid( numberOfBlocksInDirection, numberOfBlocksInDirection, numberOfBlocksInDirection,
                                                               numberOfCellsPerBlockInDirection, numberOfCellsPerBlockInDirection, numberOfCellsPerBlockInDirection,
                                                               dx, 0, false, false,
                                                               false, false, false,
                                                               false );

            getBlockLocalCellTest(blocks);

            return EXIT_SUCCESS;
        }

    }
}

int main( int argc, char** argv ) {
    return turbine_core::blockinfo_test::main(argc, argv);
}