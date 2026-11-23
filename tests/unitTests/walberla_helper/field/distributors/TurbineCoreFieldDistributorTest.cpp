
#include <blockforest/all.h>

#include <core/debug/TestSubsystem.h>
#include <core/mpi/Environment.h>
#include <core/math/all.h>

#include <field/AddToStorage.h>
#include <field/FlagField.h>
#include <field/GhostLayerField.h>
#include <field/distributors/all.h>

#include <vector>

#include "walberla_helper/IndexVectorCreator.h"
#include "walberla_helper/blockforest/BlockInfo.h"
#include "walberla_helper/field/Field.h"
#include "walberla_helper/field/distributors/NearestNeighbourFieldDistributor.h"
#include "walberla_helper/field/distributors/KernelFieldDistributor.h"
#include "walberla_helper/field/projector_kernels/SmoothedDiracDeltaKernel.h"

#include "wind_turbine_core/WalberlaDataTypes.h"

namespace turbine_core {

    namespace distribution_tests {

        const uint_t FieldGhostLayers( 1 );

        using flag_t = walberla::uint8_t;
        using FlagField_T = walberla::FlagField<flag_t>;

        typedef walberla::GhostLayerField< real_t, 1>                    ScalarField_T;
        typedef walberla::GhostLayerField< walberla::Vector3<real_t>, 1> Vec3Field_T;
        typedef walberla::GhostLayerField< real_t, 3>                    MultiComponentField_T;

        const walberla::FlagUID Domain_Flag ( "domain" );
        const walberla::FlagUID Boundary_Flag ( "boundary" );


        void initFlagField( FlagField_T * field, walberla::IBlock * const ) {
            auto domainFlag = field->getOrRegisterFlag( Domain_Flag );
            WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ( field, field->addFlag( x, y, z, domainFlag ); );
        }

        void resetScalarFields( const std::shared_ptr<walberla::StructuredBlockStorage> & blocks,
                                const BlockDataID & scalarFieldID, const BlockDataID & scalarFieldTCID ) {
            for( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt )
            {
                auto sField = blockIt->getData<ScalarField_T>( scalarFieldID );
                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(sField,
                                                                 sField->get(x,y,z) = real_t(0);
                );

                auto sFieldTC = blockIt->getData<ScalarField_T>( scalarFieldTCID );
                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(sFieldTC,
                                                                 sFieldTC->get(x,y,z) = real_t(0);
                );
            }
        }

        void resetVectorFields( const std::shared_ptr<walberla::StructuredBlockStorage> & blocks,
                                const BlockDataID & vectorFieldID, const BlockDataID & vectorFieldTCID ) {
            for( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt )
            {
                auto vField = blockIt->getData<Vec3Field_T>( vectorFieldID );
                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(vField,
                                                                 vField->get(x,y,z) = walberla::Vector3<real_t>(real_t(0));
                );

                auto vFieldTC = blockIt->getData<Vec3Field_T>( vectorFieldTCID );
                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(vFieldTC,
                                                                 vFieldTC->get(x,y,z) = walberla::Vector3<real_t>(real_t(0));
                );
            }
        }

        void resetMultiCompFields( const std::shared_ptr<walberla::StructuredBlockStorage> & blocks,
                                   const BlockDataID & multiComponentFieldID, const BlockDataID & multiComponentFieldTCID ) {
            for( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt )
            {
                auto mField = blockIt->getData<MultiComponentField_T>( multiComponentFieldID );
                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(mField,
                                                                 mField->get(x,y,z,0) = real_t(0);
                                                                         mField->get(x,y,z,1) = real_t(0);
                                                                         mField->get(x,y,z,2) = real_t(0);
                );

                auto mFieldTC = blockIt->getData<MultiComponentField_T>( multiComponentFieldTCID );
                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(mFieldTC,
                                                                 mFieldTC->get(x,y,z,0) = real_t(0);
                                                                         mFieldTC->get(x,y,z,1) = real_t(0);
                                                                         mFieldTC->get(x,y,z,2) = real_t(0);
                );
            }
        }

        void setBoundaryFlags( const std::shared_ptr<walberla::StructuredBlockStorage> & blocks,
                               const BlockDataID & flagFieldID ) {
            for( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt )
            {
                auto flagField = blockIt->getData<FlagField_T>( flagFieldID );
                auto domainFlag = flagField->getOrRegisterFlag( Domain_Flag );
                auto boundaryFlag = flagField->getOrRegisterFlag( Boundary_Flag );
                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(flagField,
                                                                 if( x == 2)
                                                                 {
                                                                     flagField->removeFlag(x,y,z,domainFlag);
                                                                     flagField->addFlag(x,y,z,boundaryFlag);
                                                                 }
                );
            }
        }

        void testNearestNeighborDistributor( const std::shared_ptr<walberla::StructuredBlockForest> & blocks, const BlockDataID & flagFieldID,
                                             const BlockDataID & scalarFieldID, const BlockDataID & vectorFieldID, const BlockDataID & multiComponentFieldID,
                                             const BlockDataID & scalarFieldTCID, const BlockDataID & vectorFieldTCID, const BlockDataID & multiComponentFieldTCID )
        {
            // distributors
            typedef walberla::field::NearestNeighborDistributor<ScalarField_T, FlagField_T>         ScalarDistributor_T;
            typedef walberla::field::NearestNeighborDistributor<Vec3Field_T, FlagField_T>           Vec3Distributor_T;
            typedef walberla::field::NearestNeighborDistributor<MultiComponentField_T, FlagField_T> MultiComponentDistributor_T;
            BlockDataID scalarDistributorID         = walberla::field::addDistributor< ScalarDistributor_T, FlagField_T >( blocks, scalarFieldID, flagFieldID, Domain_Flag );
            BlockDataID vectorDistributorID         = walberla::field::addDistributor< Vec3Distributor_T, FlagField_T >( blocks, vectorFieldID, flagFieldID, Domain_Flag );
            BlockDataID multiComponentDistributorID = walberla::field::addDistributor< MultiComponentDistributor_T, FlagField_T >( blocks, multiComponentFieldID, flagFieldID, Domain_Flag );

            // check scalar distribution
            {
                walberla::Vector3<real_t> distributionPoint(real_t(1.9), real_t(2.1), real_t(2.1));
                real_t distributionValue{100};
                auto containingBlock = blocks->getBlock(distributionPoint);
                if( containingBlock != nullptr )
                {
                    // walberla
                    auto distPtr = containingBlock->getData<ScalarDistributor_T>(scalarDistributorID);
                    distPtr->distribute(distributionPoint, &distributionValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo{ containingBlock, blocks };
                    field::Field<real_t> scalarFieldTC(containingBlock->getData<ScalarField_T>(scalarFieldTCID));

                    projectors::NearestNeighbourFieldDistributor<field::Field<real_t>> customDistributor{blockInfo, &scalarFieldTC};
                    customDistributor.distribute(distributionPoint, &distributionValue);

                    // compare fields
                    auto scalarField = containingBlock->getData<ScalarField_T>(scalarFieldID);
                    WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(scalarField,
                                                                     WALBERLA_ASSERT_FLOAT_EQUAL(scalarField->get(x,y,z,0), scalarFieldTC.get(x,y,z,0), "Custom NearestNeighbourDistributor fails for scalar distribution in interior."))

                }

                // reset
                resetScalarFields(blocks, scalarFieldID, scalarFieldTCID);
            }

            // check vector distribution
            {
                walberla::Vector3<real_t> distributionPoint(real_t(5.4),real_t(2.1),real_t(3.2));
                walberla::Vector3<real_t> distributionValue(real_t(100), real_t(-10), real_t(1));
                auto containingBlock = blocks->getBlock(distributionPoint);
                if( containingBlock != nullptr )
                {
                    // walberla
                    auto distPtr = containingBlock->getData<Vec3Distributor_T>(vectorDistributorID);
                    distPtr->distribute(distributionPoint, &distributionValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo{ containingBlock, blocks };
                    field::Field<walberla::Vector3<real_t>> vectorFieldTC(containingBlock->getData<Vec3Field_T>(vectorFieldTCID));

                    projectors::NearestNeighbourFieldDistributor<field::Field<walberla::Vector3<real_t>>> customDistributor{blockInfo, &vectorFieldTC};
                    customDistributor.distribute(distributionPoint, &distributionValue);

                    // compare fields
                    auto vectorField = containingBlock->getData<Vec3Field_T>(vectorFieldID);
                    WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(vectorField,
                                                                     WALBERLA_ASSERT_FLOAT_EQUAL(vectorField->get(x,y,z), vectorFieldTC.get(x,y,z), "Custom NearestNeighbourDistributor fails for vector distribution in interior."))

                }

                // reset
                resetVectorFields(blocks, vectorFieldID, vectorFieldTCID);
            }

            // check multi component distribution
            {
                walberla::Vector3<real_t> distributionPoint(real_t(4.4),real_t(2.1),real_t(3.2));
                std::vector<real_t> distributionValue(3);
                distributionValue[0] = real_t(100);
                distributionValue[1] = real_t(-10);
                distributionValue[2] = real_t(1);
                auto containingBlock = blocks->getBlock(distributionPoint);
                if( containingBlock != nullptr )
                {
                    // walberla
                    auto distPtr = containingBlock->getData<MultiComponentDistributor_T>(multiComponentDistributorID);
                    distPtr->distribute(distributionPoint, distributionValue.begin());

                    // turbine core
                    blockforest::BlockInfo blockInfo{ containingBlock, blocks };
                    field::Field<real_t> multiComponentFieldTC(containingBlock->getData<MultiComponentField_T>(multiComponentFieldTCID));

                    projectors::NearestNeighbourFieldDistributor<field::Field<real_t>> customDistributor{blockInfo, &multiComponentFieldTC};
                    customDistributor.distribute(distributionPoint, distributionValue.data());

                    // compare fields
                    auto multiComponentField = containingBlock->getData<MultiComponentField_T>(multiComponentFieldID);
                    WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(multiComponentField,
                                                                        WALBERLA_ASSERT_FLOAT_EQUAL(multiComponentField->get(x,y,z,0), multiComponentFieldTC.get(x,y,z,0), "Custom NearestNeighbourDistributor fails for multi component distribution in interior (index 0).")
                                                                        WALBERLA_ASSERT_FLOAT_EQUAL(multiComponentField->get(x,y,z,1), multiComponentFieldTC.get(x,y,z,1), "Custom NearestNeighbourDistributor fails for multi component distribution in interior (index 1).")
                                                                        WALBERLA_ASSERT_FLOAT_EQUAL(multiComponentField->get(x,y,z,2), multiComponentFieldTC.get(x,y,z,2), "Custom NearestNeighbourDistributor fails for multi component distribution in interior (index 2).")
                    )

                }

                // reset
                resetMultiCompFields(blocks, multiComponentFieldID, multiComponentFieldTCID);
            }
        }


        template<typename Kernel_T>
        void testKernelDistributor( const std::shared_ptr<walberla::StructuredBlockForest> & blocks, const BlockDataID & flagFieldID,
                                    const BlockDataID & scalarFieldID, const BlockDataID & vectorFieldID, const BlockDataID & multiComponentFieldID,
                                    const BlockDataID & scalarFieldTCID, const BlockDataID & vectorFieldTCID, const BlockDataID & multiComponentFieldTCID )
        {
            // distributors
            typedef walberla::field::KernelDistributor<ScalarField_T, FlagField_T>         ScalarDistributor_T;
            typedef walberla::field::KernelDistributor<Vec3Field_T, FlagField_T>           Vec3Distributor_T;
            typedef walberla::field::KernelDistributor<MultiComponentField_T, FlagField_T> MultiComponentDistributor_T;
            BlockDataID scalarDistributorID         = walberla::field::addDistributor< ScalarDistributor_T, FlagField_T >( blocks, scalarFieldID, flagFieldID, Domain_Flag );
            BlockDataID vectorDistributorID         = walberla::field::addDistributor< Vec3Distributor_T, FlagField_T >( blocks, vectorFieldID, flagFieldID, Domain_Flag );
            BlockDataID multiComponentDistributorID = walberla::field::addDistributor< MultiComponentDistributor_T, FlagField_T >( blocks, multiComponentFieldID, flagFieldID, Domain_Flag );

            // check scalar distribution
            {
                walberla::Vector3<real_t> distributionPoint(real_t(1.9), real_t(2.1), real_t(2.1));
                real_t distributionValue{100};
                auto containingBlock = blocks->getBlock(distributionPoint);
                if( containingBlock != nullptr ) {

                    // walberla
                    auto distPtr = containingBlock->getData<ScalarDistributor_T>(scalarDistributorID);
                    distPtr->distribute(distributionPoint, &distributionValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo{ containingBlock, blocks };
                    field::Field<real_t> scalarFieldTC(containingBlock->getData<ScalarField_T>(scalarFieldTCID));

                    projectors::KernelFieldDistributor<field::Field<real_t>, Kernel_T> customDistributor{blockInfo, &scalarFieldTC};
                    customDistributor.distribute(distributionPoint, &distributionValue);

                    // compare fields
                    auto scalarField = containingBlock->getData<ScalarField_T>(scalarFieldID);
                    WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(scalarField,
                                                                     WALBERLA_ASSERT_FLOAT_EQUAL(scalarField->get(x,y,z,0), scalarFieldTC.get(x,y,z,0), "Custom KernelDistributor fails for scalar distribution in interior. "))

                }

                // reset
                resetScalarFields(blocks, scalarFieldID, scalarFieldTCID);
            }

            // check vector distribution
            {
                walberla::Vector3<real_t> distributionPoint(real_t(5.4),real_t(2.1),real_t(3.2));
                walberla::Vector3<real_t> distributionValue(real_t(100), real_t(-10), real_t(1));
                auto containingBlock = blocks->getBlock(distributionPoint);
                if( containingBlock != nullptr )
                {
                    // walberla
                    auto distPtr = containingBlock->getData<Vec3Distributor_T>(vectorDistributorID);
                    distPtr->distribute(distributionPoint, &distributionValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo{ containingBlock, blocks };
                    field::Field<walberla::Vector3<real_t>> vectorFieldTC(containingBlock->getData<Vec3Field_T>(vectorFieldTCID));

                    projectors::KernelFieldDistributor<field::Field<walberla::Vector3<real_t>>, Kernel_T> customDistributor{blockInfo, &vectorFieldTC};
                    customDistributor.distribute(distributionPoint, &distributionValue);

                    // compare fields
                    auto vectorField = containingBlock->getData<Vec3Field_T>(vectorFieldID);
                    WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(vectorField,
                                                                     WALBERLA_ASSERT_FLOAT_EQUAL(vectorField->get(x,y,z), vectorFieldTC.get(x,y,z), "Custom KernelDistributor fails for vector distribution in interior."))


                }

                // reset
                resetVectorFields(blocks, vectorFieldID, vectorFieldTCID);
            }

            // check multi component distribution
            {
                walberla::Vector3<real_t> distributionPoint(real_t(4.4),real_t(2.1),real_t(3.2));
                std::vector<real_t> distributionValue(3);
                distributionValue[0] = real_t(100);
                distributionValue[1] = real_t(-10);
                distributionValue[2] = real_t(1);
                auto containingBlock = blocks->getBlock(distributionPoint);
                if( containingBlock != nullptr )
                {
                    // walberla
                    auto distPtr = containingBlock->getData<MultiComponentDistributor_T>(multiComponentDistributorID);
                    distPtr->distribute(distributionPoint, distributionValue.begin());

                    // turbine core
                    blockforest::BlockInfo blockInfo{ containingBlock, blocks };
                    field::Field<real_t> multiComponentFieldTC(containingBlock->getData<MultiComponentField_T>(multiComponentFieldTCID));

                    projectors::KernelFieldDistributor<field::Field<real_t>, Kernel_T> customDistributor{blockInfo, &multiComponentFieldTC};
                    customDistributor.distribute(distributionPoint, distributionValue.data());

                    // compare fields
                    auto multiComponentField = containingBlock->getData<MultiComponentField_T>(multiComponentFieldID);
                    WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(multiComponentField,
                                                                     WALBERLA_ASSERT_FLOAT_EQUAL(multiComponentField->get(x,y,z,0), multiComponentFieldTC.get(x,y,z,0), "Custom KernelDistributor fails for multi component distribution in interior (index 0).")
                                                                     WALBERLA_ASSERT_FLOAT_EQUAL(multiComponentField->get(x,y,z,1), multiComponentFieldTC.get(x,y,z,1), "Custom KernelDistributor fails for multi component distribution in interior (index 1).")
                                                                     WALBERLA_ASSERT_FLOAT_EQUAL(multiComponentField->get(x,y,z,2), multiComponentFieldTC.get(x,y,z,2), "Custom KernelDistributor fails for multi component distribution in interior (index 2).")
                    )

                }

                // reset
                resetMultiCompFields(blocks, multiComponentFieldID, multiComponentFieldTCID);
            }
        }


        void testNearestNeighborDistributorAtBoundary( const std::shared_ptr<walberla::StructuredBlockForest> & blocks,
                                                       const BlockDataID & flagFieldID,
                                                       const BlockDataID & scalarFieldID, const BlockDataID & scalarFieldTCID )
        {
            // distributor
            typedef walberla::field::NearestNeighborDistributor<ScalarField_T, FlagField_T> ScalarDistributor_T;
            BlockDataID scalarDistributorID = walberla::field::addDistributor<ScalarDistributor_T, FlagField_T>(blocks, scalarFieldID, flagFieldID, Domain_Flag);

            // check scalar interpolation close to boundary
            {
                walberla::Vector3<real_t> distributionPoint(real_t(1.9), real_t(2.1), real_t(2.1));
                real_t distributionValue{100};
                auto containingBlock = blocks->getBlock(distributionPoint);
                if (containingBlock != nullptr) {
                    // walberla
                    auto distPtr = containingBlock->getData<ScalarDistributor_T>(scalarDistributorID);
                    distPtr->distribute(distributionPoint, &distributionValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo{ containingBlock, blocks };
                    field::Field<real_t> scalarFieldTC(containingBlock->getData<ScalarField_T>(scalarFieldTCID));

                    projectors::NearestNeighbourFieldDistributor<field::Field<real_t>> customDistributor{blockInfo, &scalarFieldTC};
                    customDistributor.distribute(distributionPoint, &distributionValue);

                    // compare fields
                    auto scalarField = containingBlock->getData<ScalarField_T>(scalarFieldID);
                    WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(scalarField,
                                                                     WALBERLA_ASSERT_FLOAT_EQUAL(scalarField->get(x,y,z,0), scalarFieldTC.get(x,y,z,0), "Custom NearestNeighbourDistributor fails for scalar distribution close to boundary."))

                }

                // reset
                resetScalarFields(blocks, scalarFieldID, scalarFieldTCID);
            }

            // check scalar interpolation inside boundary
            {
                walberla::Vector3<real_t> distributionPoint(real_t(2.7), real_t(2.1), real_t(1.1));
                real_t distributionValue(real_t(100));
                auto containingBlock = blocks->getBlock(distributionPoint);
                if (containingBlock != nullptr) {
                    // walberla
                    auto distPtr = containingBlock->getData<ScalarDistributor_T>(scalarDistributorID);
                    distPtr->distribute(distributionPoint, &distributionValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo{ containingBlock, blocks };
                    field::Field<real_t> scalarFieldTC(containingBlock->getData<ScalarField_T>(scalarFieldTCID));

                    projectors::NearestNeighbourFieldDistributor<field::Field<real_t>> customDistributor{blockInfo, &scalarFieldTC};
                    customDistributor.distribute(distributionPoint, &distributionValue);

                    // compare fields
                    auto scalarField = containingBlock->getData<ScalarField_T>(scalarFieldID);
                    WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(scalarField,
                                                                     WALBERLA_ASSERT_FLOAT_EQUAL(scalarField->get(x,y,z,0), scalarFieldTC.get(x,y,z,0), "Custom NearestNeighbourDistributor fails for scalar distribution inside boundary."))

                }

                // reset
                resetScalarFields(blocks, scalarFieldID, scalarFieldTCID);
            }
        }

        template<typename Kernel_T>
        void testKernelDistributorAtBoundary( const std::shared_ptr<walberla::StructuredBlockForest> & blocks,
                                              const BlockDataID & flagFieldID,
                                              const BlockDataID & scalarFieldID, const BlockDataID & scalarFieldTCID )
        {
            // distributor
            typedef walberla::field::KernelDistributor<ScalarField_T, FlagField_T> ScalarDistributor_T;
            BlockDataID scalarDistributorID = walberla::field::addDistributor<ScalarDistributor_T, FlagField_T>(blocks, scalarFieldID, flagFieldID, Domain_Flag);

            // check scalar interpolation close to boundary
            {
                walberla::Vector3<real_t> distributionPoint(real_t(1.9), real_t(2.1), real_t(2.1));
                real_t distributionValue{100};
                auto containingBlock = blocks->getBlock(distributionPoint);
                if (containingBlock != nullptr) {
                    // walberla
                    auto distPtr = containingBlock->getData<ScalarDistributor_T>(scalarDistributorID);
                    distPtr->distribute(distributionPoint, &distributionValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo{ containingBlock, blocks };
                    field::Field<real_t> scalarFieldTC(containingBlock->getData<ScalarField_T>(scalarFieldTCID));

                    projectors::KernelFieldDistributor<field::Field<real_t>, Kernel_T> customDistributor{blockInfo, &scalarFieldTC};
                    customDistributor.distribute(distributionPoint, &distributionValue);

                    // compare fields
                    auto scalarField = containingBlock->getData<ScalarField_T>(scalarFieldID);
                    WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(scalarField,
                                                                     WALBERLA_ASSERT_FLOAT_EQUAL(scalarField->get(x,y,z,0), scalarFieldTC.get(x,y,z,0), "Custom KernelDistributor fails for scalar distribution close to boundary. "))

                }

                // reset
                resetScalarFields(blocks, scalarFieldID, scalarFieldTCID);
            }

            // check scalar interpolation inside boundary
            {
                walberla::Vector3<real_t> distributionPoint(real_t(2.7), real_t(2.1), real_t(1.1));
                real_t distributionValue{100};
                auto containingBlock = blocks->getBlock(distributionPoint);
                if (containingBlock != nullptr) {
                    // walberla
                    auto distPtr = containingBlock->getData<ScalarDistributor_T>(scalarDistributorID);
                    distPtr->distribute(distributionPoint, &distributionValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo{ containingBlock, blocks };
                    field::Field<real_t> scalarFieldTC(containingBlock->getData<ScalarField_T>(scalarFieldTCID));

                    projectors::KernelFieldDistributor<field::Field<real_t>, Kernel_T> customDistributor{blockInfo, &scalarFieldTC};
                    customDistributor.distribute(distributionPoint, &distributionValue);

                    // compare fields
                    auto scalarField = containingBlock->getData<ScalarField_T>(scalarFieldID);
                    WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(scalarField,
                                                                     WALBERLA_ASSERT_FLOAT_EQUAL(scalarField->get(x,y,z,0), scalarFieldTC.get(x,y,z,0), "Custom KernelDistributor fails for scalar distribution inside boundary. "))

                }

                // reset
                resetScalarFields(blocks, scalarFieldID, scalarFieldTCID);
            }
        }


        int main(int argc, char **argv) {

            walberla::mpi::Environment mpiEnv(argc, argv);
            walberla::debug::enterTestMode();

            const uint_t numberOfBlocksInDirection = 2;
            const uint_t numberOfCellsPerBlockInDirection = 4;
            const real_t dx {1};

            // block storage
            auto blocks = walberla::blockforest::createUniformBlockGrid( numberOfBlocksInDirection, numberOfBlocksInDirection, numberOfBlocksInDirection,
                                                                         numberOfCellsPerBlockInDirection, numberOfCellsPerBlockInDirection, numberOfCellsPerBlockInDirection,
                                                                         dx, 0, false, false,
                                                                         false, false, false,
                                                                         false );
            // flag field
            BlockDataID flagFieldID = walberla::field::addFlagFieldToStorage<FlagField_T>( blocks, "flag field", FieldGhostLayers, false, initFlagField );

            // data fields for walberla distributor
            BlockDataID scalarFieldID         = walberla::field::addToStorage< ScalarField_T >( blocks, "scalar field", real_t(0), walberla::field::fzyx, FieldGhostLayers );
            BlockDataID vectorFieldID         = walberla::field::addToStorage< Vec3Field_T >( blocks, "vec3 field", walberla::Vector3<real_t>(real_t(0)), walberla::field::fzyx, FieldGhostLayers );
            BlockDataID multiComponentFieldID = walberla::field::addToStorage< MultiComponentField_T >( blocks, "multi component field", real_t(0), walberla::field::fzyx, FieldGhostLayers );

            // data fields for turbine core distributor
            BlockDataID scalarFieldTCID         = walberla::field::addCloneToStorage< ScalarField_T >( blocks, scalarFieldID, "scalar field TC" );
            BlockDataID vectorFieldTCID         = walberla::field::addCloneToStorage< Vec3Field_T >( blocks, vectorFieldID, "vec3 field TC" );
            BlockDataID multiComponentFieldTCID = walberla::field::addCloneToStorage< MultiComponentField_T >( blocks, multiComponentFieldID, "multi component field TC" );

            // test all distributors with domain flags everywhere, i.e. without special boundary treatment necessary
            testNearestNeighborDistributor(blocks, flagFieldID,
                                           scalarFieldID, vectorFieldID, multiComponentFieldID,
                                           scalarFieldTCID, vectorFieldTCID, multiComponentFieldTCID);
            testKernelDistributor<projectors::SmoothedDiracDeltaKernel>(blocks, flagFieldID,
                                  scalarFieldID, vectorFieldID, multiComponentFieldID,
                                  scalarFieldTCID, vectorFieldTCID, multiComponentFieldTCID);

            // set some boundary flags in flag field and invalidate the corresponding scalar field values
            setBoundaryFlags(blocks, flagFieldID );

            // test all distributors' behaviour close to boundary cells
            testNearestNeighborDistributorAtBoundary(blocks, flagFieldID, scalarFieldID, scalarFieldTCID);
            testKernelDistributorAtBoundary<projectors::SmoothedDiracDeltaKernel>(blocks, flagFieldID, scalarFieldID, scalarFieldTCID);

            return 0;
        }

    } // namespace field_distribution_tests

} // namespace turbine_core

int main( int argc, char **argv ){
    turbine_core::distribution_tests::main(argc, argv);
}