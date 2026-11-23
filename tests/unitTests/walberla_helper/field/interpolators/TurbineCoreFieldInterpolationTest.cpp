
#include <blockforest/all.h>

#include <core/debug/TestSubsystem.h>
#include <core/mpi/Environment.h>
#include <core/math/Vector3.h>

#include <field/AddToStorage.h>
#include <field/FlagField.h>
#include <field/GhostLayerField.h>
#include <field/interpolators/all.h>

#include <vector>
#include <random>

#include "walberla_helper/field/Field.h"
#include "walberla_helper/field/interpolators/NearestNeighbourFieldInterpolator.h"
#include "walberla_helper/field/interpolators/TrilinearFieldInterpolator.h"

namespace turbine_core {

    namespace field_interpolation_tests {

        using namespace walberla;

        const uint_t FieldGhostLayers( 1 );

        using flag_t = walberla::uint8_t;
        using FlagField_T = FlagField<flag_t>;

        typedef GhostLayerField< real_t, 1>          ScalarField_T;
        typedef GhostLayerField< walberla::Vector3<real_t>, 1> Vec3Field_T;
        typedef GhostLayerField< real_t, 3>          MultiComponentField_T;

        const FlagUID Domain_Flag ( "domain" );
        const FlagUID Boundary_Flag ( "boundary" );

        void initFlagField( FlagField_T * field, IBlock * const /*block*/ )
        {
            auto domainFlag = field->getOrRegisterFlag( Domain_Flag );
            WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ( field, field->addFlag( x, y, z, domainFlag ); );
        }

        void initScalarField( const shared_ptr<StructuredBlockStorage> & blocks, const BlockDataID & scalarFieldID ) {

            std::random_device rd;
            std::mt19937 e2(rd());
            std::uniform_real_distribution<real_t> dist(real_t(0), real_t(10));

            for( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt )
            {
                auto field = blockIt->getData<ScalarField_T>( scalarFieldID );
                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(field,
                                                                 field->get(x,y,z) = dist(e2);
                );
            }
        }

        void initVectorField( const shared_ptr<StructuredBlockStorage> & blocks, const BlockDataID & vectorFieldID ) {

            std::random_device rd;
            std::mt19937 e2(rd());
            std::uniform_real_distribution<real_t> dist(real_t(0), real_t(10));

            for( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt )
            {
                auto field = blockIt->getData<Vec3Field_T>( vectorFieldID );
                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(field,
                                                                 field->get(x,y,z) = walberla::Vector3<real_t>(dist(e2), dist(e2), dist(e2));
                );
            }
        }

        void initMultiComponentField( const shared_ptr<StructuredBlockStorage> & blocks, const BlockDataID & multiComponentFieldID ) {
            std::random_device rd;
            std::mt19937 e2(rd());
            std::uniform_real_distribution<real_t> dist(real_t(0), real_t(10));

            for( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt )
            {
                auto field = blockIt->getData<MultiComponentField_T>( multiComponentFieldID );
                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(field,
                                                                 field->get(x,y,z,0) = dist(e2);
                                                                         field->get(x,y,z,1) = dist(e2);
                                                                         field->get(x,y,z,2) = dist(e2);
                );
            }
        }

        void setBoundaryFlags( const shared_ptr<StructuredBlockStorage> & blocks,
                               const BlockDataID & flagFieldID, const BlockDataID & scalarFieldID )
        {
            for( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt )
            {
                auto flagField = blockIt->getData<FlagField_T>( flagFieldID );
                auto valueField = blockIt->getData<ScalarField_T>( scalarFieldID );
                auto domainFlag = flagField->getOrRegisterFlag( Domain_Flag );
                auto boundaryFlag = flagField->getOrRegisterFlag( Boundary_Flag );
                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(flagField,
                                                                 if( x == 2)
                                                                 {
                                                                     flagField->removeFlag(x,y,z,domainFlag);
                                                                     flagField->addFlag(x,y,z,boundaryFlag);
                                                                     valueField->get(x,y,z) = std::numeric_limits<real_t>::max();
                                                                 }
                );
            }
        }

        void testNearestNeighborFieldInterpolator( const shared_ptr<StructuredBlockForest> & blocks, const BlockDataID & flagFieldID,
                                                   const BlockDataID & scalarFieldID, const BlockDataID & vectorFieldID, const BlockDataID & multiComponentFieldID ) {

            // field interpolators
            typedef walberla::field::NearestNeighborFieldInterpolator<ScalarField_T, FlagField_T>         ScalarFieldInterpolator_T;
            typedef walberla::field::NearestNeighborFieldInterpolator<Vec3Field_T, FlagField_T>           Vec3FieldInterpolator_T;
            typedef walberla::field::NearestNeighborFieldInterpolator<MultiComponentField_T, FlagField_T> MultiComponentFieldInterpolator_T;
            BlockDataID scalarFieldInterpolatorID         = walberla::field::addFieldInterpolator< ScalarFieldInterpolator_T, FlagField_T >( blocks, scalarFieldID, flagFieldID, Domain_Flag );
            BlockDataID vectorFieldInterpolatorID         = walberla::field::addFieldInterpolator< Vec3FieldInterpolator_T, FlagField_T >( blocks, vectorFieldID, flagFieldID, Domain_Flag );
            BlockDataID multiComponentFieldInterpolatorID = walberla::field::addFieldInterpolator< MultiComponentFieldInterpolator_T, FlagField_T >( blocks, multiComponentFieldID, flagFieldID, Domain_Flag );

            // check scalar interpolation
            {
                walberla::Vector3<real_t> interpolationPoint(real_t(1.9), real_t(2.1), real_t(2.1));
                auto containingBlock = blocks->getBlock(interpolationPoint);
                if( containingBlock != nullptr ) {
                    // walberla
                    real_t interpolatedValue{0.0};
                    auto interPtr = containingBlock->getData<ScalarFieldInterpolator_T>(scalarFieldInterpolatorID);
                    interPtr->get(interpolationPoint, &interpolatedValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo(containingBlock, blocks);
                    field::Field<real_t> scalarField(containingBlock->getData<ScalarField_T>(scalarFieldID));

                    real_t interpolatedValueTC{0};
                    projectors::NearestNeighbourFieldInterpolator<field::Field<real_t>> customInterpolator(blockInfo, &scalarField);
                    customInterpolator.get(Vector3<real_t>(interpolationPoint), &interpolatedValueTC);

                    // compare
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue, interpolatedValueTC, "Custom NearestNeighborFieldInterpolator: Scalar interpolation failed");
                }
            }

            // check vector interpolation
            {
                walberla::Vector3<real_t> interpolationPoint(real_t(5.4),real_t(2.1),real_t(3.2));
                auto containingBlock = blocks->getBlock( interpolationPoint );
                if( containingBlock != nullptr ) {
                    // walberla
                    Vector3<real_t> interpolatedValue(real_t(0));
                    auto interPtr = containingBlock->getData<Vec3FieldInterpolator_T>(vectorFieldInterpolatorID);
                    interPtr->get(interpolationPoint, &interpolatedValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo(containingBlock, blocks);
                    field::Field<walberla::Vector3<real_t>> vectorField(containingBlock->getData<Vec3Field_T>(vectorFieldID));

                    Vector3<real_t> interpolatedValueTC(real_t(0));
                    projectors::NearestNeighbourFieldInterpolator<field::Field<walberla::Vector3<real_t>>> customInterpolator(blockInfo, &vectorField);
                    customInterpolator.get(Vector3<real_t>(interpolationPoint), &interpolatedValueTC);

                    // compare
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[0], interpolatedValueTC[0], "Custom NearestNeighborFieldInterpolator: Vec3[0] interpolation failed");
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[1], interpolatedValueTC[1], "Custom NearestNeighborFieldInterpolator: Vec3[1] interpolation failed");
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[2], interpolatedValueTC[2], "Custom NearestNeighborFieldInterpolator: Vec3[2] interpolation failed");
                }
            }

            // check multi component interpolation
            {
                walberla::Vector3<real_t> interpolationPoint(real_t(4.4),real_t(2.1),real_t(3.2));
                auto containingBlock = blocks->getBlock( interpolationPoint );
                if( containingBlock != nullptr ) {
                    // walberla
                    std::vector<real_t> interpolatedValue(3, real_t(0));
                    auto interPtr = containingBlock->getData<MultiComponentFieldInterpolator_T>(multiComponentFieldInterpolatorID);
                    interPtr->get(interpolationPoint, interpolatedValue.begin());

                    // turbine core
                    blockforest::BlockInfo blockInfo(containingBlock, blocks);
                    field::Field<real_t> multiComponentField(containingBlock->getData<MultiComponentField_T>(multiComponentFieldID));

                    std::vector<real_t> interpolatedValueTC{0};
                    projectors::NearestNeighbourFieldInterpolator<field::Field<real_t>> customInterpolator(blockInfo, &multiComponentField);
                    customInterpolator.get(Vector3<real_t>(interpolationPoint), interpolatedValueTC.begin());

                    // compare
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[0], interpolatedValueTC[0], "Custom NearestNeighborFieldInterpolator: Multi Component[0] interpolation failed");
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[1], interpolatedValueTC[1], "Custom NearestNeighborFieldInterpolator: Multi Component[1] interpolation failed");
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[2], interpolatedValueTC[2], "Custom NearestNeighborFieldInterpolator: Multi Component[2] interpolation failed");
                }
            }
        }

        void testTrilinearFieldInterpolator( const shared_ptr<StructuredBlockForest> & blocks, const BlockDataID & flagFieldID,
                                             const BlockDataID & scalarFieldID, const BlockDataID & vectorFieldID, const BlockDataID & multiComponentFieldID )
        {
            // field interpolators
            typedef walberla::field::TrilinearFieldInterpolator<ScalarField_T, FlagField_T>         ScalarFieldInterpolator_T;
            typedef walberla::field::TrilinearFieldInterpolator<Vec3Field_T, FlagField_T>           Vec3FieldInterpolator_T;
            typedef walberla::field::TrilinearFieldInterpolator<MultiComponentField_T, FlagField_T> MultiComponentFieldInterpolator_T;
            BlockDataID scalarFieldInterpolatorID         = walberla::field::addFieldInterpolator< ScalarFieldInterpolator_T, FlagField_T >( blocks, scalarFieldID, flagFieldID, Domain_Flag );
            BlockDataID vectorFieldInterpolatorID         = walberla::field::addFieldInterpolator< Vec3FieldInterpolator_T, FlagField_T >( blocks, vectorFieldID, flagFieldID, Domain_Flag );
            BlockDataID multiComponentFieldInterpolatorID = walberla::field::addFieldInterpolator< MultiComponentFieldInterpolator_T, FlagField_T >( blocks, multiComponentFieldID, flagFieldID, Domain_Flag );

            // check scalar interpolation
            {
                walberla::Vector3<real_t> interpolationPoint(real_t(1.9), real_t(2.1), real_t(2.1));
                auto containingBlock = blocks->getBlock(interpolationPoint);
                if( containingBlock != nullptr ) {

                    // walberla
                    real_t interpolatedValue{0};
                    auto interPtr = containingBlock->getData<ScalarFieldInterpolator_T>(scalarFieldInterpolatorID);
                    interPtr->get(interpolationPoint, &interpolatedValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo(containingBlock, blocks);
                    field::Field<real_t> scalarField(containingBlock->getData<ScalarField_T>(scalarFieldID));

                    real_t interpolatedValueTC{0};
                    projectors::TrilinearFieldInterpolator<field::Field<real_t>> customInterpolator(blockInfo, &scalarField);
                    customInterpolator.get(Vector3<real_t>(interpolationPoint), &interpolatedValueTC);

                    // compare
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue, interpolatedValueTC, "Custom TrilinearFieldInterpolator: Scalar interpolation failed");

                }
            }

            // check vector interpolation
            {
                walberla::Vector3<real_t> interpolationPoint(real_t(5.4),real_t(2.1),real_t(3.2));
                auto containingBlock = blocks->getBlock( interpolationPoint );
                if( containingBlock != nullptr ) {

                    // walberla
                    walberla::Vector3<real_t> interpolatedValue(real_t(0));
                    auto interPtr = containingBlock->getData<Vec3FieldInterpolator_T>(vectorFieldInterpolatorID);
                    interPtr->get(interpolationPoint, &interpolatedValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo(containingBlock, blocks);
                    field::Field<walberla::Vector3<real_t>> vectorField(containingBlock->getData<Vec3Field_T>(vectorFieldID));

                    Vector3<real_t> interpolatedValueTC(real_t(0));
                    projectors::TrilinearFieldInterpolator<field::Field<walberla::Vector3<real_t>>> customInterpolator(blockInfo, &vectorField);
                    customInterpolator.get(Vector3<real_t>(interpolationPoint), &interpolatedValueTC);

                    // compare
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[0], interpolatedValueTC[0], "Custom TrilinearFieldInterpolator: Vec3[0] interpolation failed");
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[1], interpolatedValueTC[1], "Custom TrilinearFieldInterpolator: Vec3[1] interpolation failed");
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[2], interpolatedValueTC[2], "Custom TrilinearFieldInterpolator: Vec3[2] interpolation failed");

                }
            }

            // check multi component interpolation
            {
                walberla::Vector3<real_t> interpolationPoint(real_t(4.4),real_t(2.1),real_t(3.2));
                auto containingBlock = blocks->getBlock( interpolationPoint );
                if( containingBlock != nullptr ) {

                    // walberla
                    std::vector<real_t> interpolatedValue(3, real_t(0));
                    auto interPtr = containingBlock->getData<MultiComponentFieldInterpolator_T>(multiComponentFieldInterpolatorID);
                    interPtr->get(interpolationPoint, interpolatedValue.begin());

                    // turbine core
                    blockforest::BlockInfo blockInfo(containingBlock, blocks);
                    field::Field<real_t> multiComponentField(containingBlock->getData<MultiComponentField_T>(multiComponentFieldID));

                    std::vector<real_t> interpolatedValueTC(3, real_t(0));
                    projectors::TrilinearFieldInterpolator<field::Field<real_t>> customInterpolator(blockInfo, &multiComponentField);
                    customInterpolator.get(Vector3<real_t>(interpolationPoint), interpolatedValueTC.begin());

                    // compare
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[0], interpolatedValueTC[0], "Custom TrilinearFieldInterpolator: Multi Component[0] interpolation failed");
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[1], interpolatedValueTC[1], "Custom TrilinearFieldInterpolator: Multi Component[1] interpolation failed");
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[2], interpolatedValueTC[2], "Custom TrilinearFieldInterpolator: Multi Component[2] interpolation failed");

                }
            }
        }
/*

        void testKernelFieldInterpolator( const shared_ptr<StructuredBlockStorage> & blocks, const BlockDataID & flagFieldID,
                                          const BlockDataID & scalarFieldID, const BlockDataID & vectorFieldID, const BlockDataID & multiComponentFieldID )
        {
            // field interpolators
            typedef field::KernelFieldInterpolator<ScalarField_T, FlagField_T>         ScalarFieldInterpolator_T;
            typedef field::KernelFieldInterpolator<Vec3Field_T, FlagField_T>           Vec3FieldInterpolator_T;
            typedef field::KernelFieldInterpolator<MultiComponentField_T, FlagField_T> MultiComponentFieldInterpolator_T;
            BlockDataID scalarFieldInterpolatorID         = field::addFieldInterpolator< ScalarFieldInterpolator_T, FlagField_T >( blocks, scalarFieldID, flagFieldID, Domain_Flag );
            BlockDataID vectorFieldInterpolatorID         = field::addFieldInterpolator< Vec3FieldInterpolator_T, FlagField_T >( blocks, vectorFieldID, flagFieldID, Domain_Flag );
            BlockDataID multiComponentFieldInterpolatorID = field::addFieldInterpolator< MultiComponentFieldInterpolator_T, FlagField_T >( blocks, multiComponentFieldID, flagFieldID, Domain_Flag );

            // check scalar interpolation
            {
                Vector3<real_t> interpolationPoint(real_t(1.9), real_t(2.1), real_t(2.1));
                auto containingBlockID = blocks->getBlock(interpolationPoint);
                if( containingBlockID != nullptr )
                {
                    real_t interpolatedValue{0};
                    auto interPtr = containingBlockID->getData<ScalarFieldInterpolator_T>(scalarFieldInterpolatorID);
                    interPtr->get(interpolationPoint, &interpolatedValue);
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue, real_t(1.4), "KernelFieldInterpolator: Scalar interpolation failed");
                }
            }

            // check vector interpolation
            {
                Vector3<real_t> interpolationPoint(real_t(5.4),real_t(2.1),real_t(3.2));
                auto containingBlockID = blocks->getBlock( interpolationPoint );
                if( containingBlockID != nullptr ) {
                    Vector3<real_t> interpolatedValue(real_t(0));
                    auto interPtr = containingBlockID->getData<Vec3FieldInterpolator_T>(vectorFieldInterpolatorID);
                    interPtr->get(interpolationPoint, &interpolatedValue);
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[0], real_t(4.9), "KernelFieldInterpolator: Vec3[0] interpolation failed");
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[1], real_t(2.0), "KernelFieldInterpolator: Vec3[1] interpolation failed");
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[2], real_t(1.6), "KernelFieldInterpolator: Vec3[2] interpolation failed");
                }
            }

            // check multi component interpolation
            {
                Vector3<real_t> interpolationPoint(real_t(4.4),real_t(2.1),real_t(3.2));
                auto containingBlockID = blocks->getBlock( interpolationPoint );
                if( containingBlockID != nullptr ) {
                    std::vector<real_t> interpolatedValue(3, real_t(0));
                    auto interPtr = containingBlockID->getData<MultiComponentFieldInterpolator_T>(multiComponentFieldInterpolatorID);
                    interPtr->get(interpolationPoint, interpolatedValue.begin());
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[0], real_t(3.9), "KernelFieldInterpolator: Multi Component[0] interpolation failed");
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[1], real_t(2.0), "KernelFieldInterpolator: Multi Component[1] interpolation failed");
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue[2], real_t(1.6), "KernelFieldInterpolator: Multi Component[2] interpolation failed");
                }
            }
        }
*/

        void testNearestNeighborFieldInterpolatorAtBoundary( const shared_ptr<StructuredBlockForest> & blocks,
                                                             const BlockDataID & flagFieldID, const BlockDataID & scalarFieldID) {
            // field interpolators
            typedef walberla::field::NearestNeighborFieldInterpolator<ScalarField_T, FlagField_T> ScalarFieldInterpolator_T;
            BlockDataID scalarFieldInterpolatorID = walberla::field::addFieldInterpolator<ScalarFieldInterpolator_T, FlagField_T>(blocks, scalarFieldID, flagFieldID, Domain_Flag);

            // check scalar interpolation close to boundary
            {
                walberla::Vector3<real_t> interpolationPoint(real_t(1.9), real_t(2.1), real_t(2.1));
                auto containingBlock = blocks->getBlock(interpolationPoint);
                if (containingBlock != nullptr) {
                    // walberla
                    real_t interpolatedValue{0};
                    auto interPtr = containingBlock->getData<ScalarFieldInterpolator_T>(scalarFieldInterpolatorID);
                    interPtr->get(interpolationPoint, &interpolatedValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo(containingBlock, blocks);
                    field::Field<real_t> scalarField(containingBlock->getData<ScalarField_T>(scalarFieldID));

                    real_t interpolatedValueTC{0};
                    projectors::NearestNeighbourFieldInterpolator<field::Field<real_t>> customInterpolator(blockInfo, &scalarField);
                    customInterpolator.get(Vector3<real_t>(interpolationPoint), &interpolatedValueTC);

                    // compare
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue, interpolatedValueTC,
                                               "Custom NearestNeighborFieldInterpolator: Scalar interpolation near boundary failed");
                }
            }

            // check scalar interpolation inside boundary
            {
                walberla::Vector3<real_t> interpolationPoint(real_t(2.7), real_t(2.1), real_t(1.1));
                auto containingBlock = blocks->getBlock(interpolationPoint);
                if (containingBlock != nullptr) {
                    // walberla
                    real_t interpolatedValue{0};
                    auto interPtr = containingBlock->getData<ScalarFieldInterpolator_T>(scalarFieldInterpolatorID);
                    interPtr->get(interpolationPoint, &interpolatedValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo(containingBlock, blocks);
                    field::Field<real_t> scalarField(containingBlock->getData<ScalarField_T>(scalarFieldID));

                    real_t interpolatedValueTC{0};
                    projectors::NearestNeighbourFieldInterpolator<field::Field<real_t>> customInterpolator(blockInfo, &scalarField);
                    customInterpolator.get(Vector3<real_t>(interpolationPoint), &interpolatedValueTC);

                    // compare
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue, interpolatedValueTC,
                                               "NearestNeighborFieldInterpolator: Scalar interpolation inside boundary failed");
                }
            }
        }


        void testTrilinearFieldInterpolatorAtBoundary( const shared_ptr<StructuredBlockForest> & blocks,
                                                       const BlockDataID & flagFieldID, const BlockDataID & scalarFieldID ) {
            // field interpolators
            typedef walberla::field::TrilinearFieldInterpolator<ScalarField_T, FlagField_T> ScalarFieldInterpolator_T;
            BlockDataID scalarFieldInterpolatorID = walberla::field::addFieldInterpolator<ScalarFieldInterpolator_T, FlagField_T>(blocks, scalarFieldID, flagFieldID, Domain_Flag);

            // check scalar interpolation close to boundary
            {
                walberla::Vector3<real_t> interpolationPoint(real_t(1.9), real_t(2.1), real_t(2.1));
                auto containingBlock = blocks->getBlock(interpolationPoint);
                if (containingBlock != nullptr) {
                    //walberla
                    real_t interpolatedValue{0};
                    auto interPtr = containingBlock->getData<ScalarFieldInterpolator_T>(scalarFieldInterpolatorID);
                    interPtr->get(interpolationPoint, &interpolatedValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo(containingBlock, blocks);
                    field::Field<real_t> scalarField(containingBlock->getData<ScalarField_T>(scalarFieldID));

                    real_t interpolatedValueTC{0};
                    projectors::TrilinearFieldInterpolator<field::Field<real_t>> customInterpolator(blockInfo, &scalarField);
                    customInterpolator.get(Vector3<real_t>(interpolationPoint), &interpolatedValueTC);

                    // compare
                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue, interpolatedValueTC, "Custom TrilinearFieldInterpolator: Scalar interpolation near boundary failed");

                }
            }

            // check scalar interpolation inside boundary
            {
                walberla::Vector3<real_t> interpolationPoint(real_t(2.7), real_t(2.1), real_t(1.1));
                auto containingBlock = blocks->getBlock(interpolationPoint);
                if (containingBlock != nullptr) {
                    // walberla
                    real_t interpolatedValue{0};
                    auto interPtr = containingBlock->getData<ScalarFieldInterpolator_T>(scalarFieldInterpolatorID);
                    interPtr->get(interpolationPoint, &interpolatedValue);

                    // turbine core
                    blockforest::BlockInfo blockInfo(containingBlock, blocks);
                    field::Field<real_t> scalarField(containingBlock->getData<ScalarField_T>(scalarFieldID));

                    real_t interpolatedValueTC{0};
                    projectors::TrilinearFieldInterpolator<field::Field<real_t>> customInterpolator(blockInfo, &scalarField);
                    customInterpolator.get(Vector3<real_t>(interpolationPoint), &interpolatedValueTC);

                    WALBERLA_CHECK_FLOAT_EQUAL(interpolatedValue, interpolatedValueTC,
                                               "Custom TrilinearFieldInterpolator: Scalar interpolation inside boundary failed");
                }
            }
        }

/*

        void testKernelFieldInterpolatorAtBoundary( const shared_ptr<StructuredBlockStorage> & blocks,
                                                    const BlockDataID & flagFieldID, const BlockDataID & scalarFieldID ) {
            // field interpolators
            typedef field::KernelFieldInterpolator<ScalarField_T, FlagField_T> ScalarFieldInterpolator_T;
            BlockDataID scalarFieldInterpolatorID = field::addFieldInterpolator<ScalarFieldInterpolator_T, FlagField_T>(blocks, scalarFieldID, flagFieldID, Domain_Flag);

            // check scalar interpolation close to boundary
            {
                Vector3<real_t> interpolationPoint(real_t(1.9), real_t(2.1), real_t(2.1));
                auto containingBlockID = blocks->getBlock(interpolationPoint);
                if (containingBlockID != nullptr) {
                    real_t interpolatedValue{0};
                    auto interPtr = containingBlockID->getData<ScalarFieldInterpolator_T>(scalarFieldInterpolatorID);
                    interPtr->get(interpolationPoint, &interpolatedValue);
                    // kernel interpolation can not extrapolate values from the available ones (see comments in KernelFieldInterpolator.h)
                    // it will thus yield a value between the available ones, which are 0 and 1
                    WALBERLA_CHECK(interpolatedValue < real_t(1),
                                   "KernelFieldInterpolator: Scalar interpolation near boundary failed");
                    WALBERLA_CHECK(interpolatedValue > real_t(0),
                                   "KernelFieldInterpolator: Scalar interpolation near boundary failed");
                }
            }

            // check scalar interpolation inside boundary
            {
                Vector3<real_t> interpolationPoint(real_t(2.7), real_t(2.1), real_t(1.1));
                auto containingBlockID = blocks->getBlock(interpolationPoint);
                if (containingBlockID != nullptr) {
                    real_t interpolatedValue{0};
                    auto interPtr = containingBlockID->getData<ScalarFieldInterpolator_T>(scalarFieldInterpolatorID);
                    interPtr->get(interpolationPoint, &interpolatedValue);
                    // values has to be between the available ones, i.e. 1 and 3
                    WALBERLA_CHECK(interpolatedValue > real_t(1),
                                   "KernelFieldInterpolator: Scalar interpolation inside boundary failed");
                    WALBERLA_CHECK(interpolatedValue < real_t(3),
                                   "KernelFieldInterpolator: Scalar interpolation inside boundary failed");
                }
            }
        }
*/

        int main(int argc, char **argv) {

            mpi::Environment mpiEnv(argc, argv);
            debug::enterTestMode();

            const uint_t numberOfBlocksInDirection = 2;
            const uint_t numberOfCellsPerBlockInDirection = 4;
            const real_t dx{1};

            // block storage
            auto blocks = walberla::blockforest::createUniformBlockGrid( numberOfBlocksInDirection, numberOfBlocksInDirection, numberOfBlocksInDirection,
                                                                         numberOfCellsPerBlockInDirection, numberOfCellsPerBlockInDirection, numberOfCellsPerBlockInDirection,
                                                                         dx, 0, false, false,
                                                                         false, false, false,
                                                                         false );
            // flag field
            BlockDataID flagFieldID = walberla::field::addFlagFieldToStorage<FlagField_T>( blocks, "flag field", FieldGhostLayers, false, initFlagField );

            // data fields
            BlockDataID scalarFieldID         = walberla::field::addToStorage< ScalarField_T >( blocks, "scalar field", real_t(0), walberla::field::fzyx, FieldGhostLayers );
            BlockDataID vectorFieldID         = walberla::field::addToStorage< Vec3Field_T >( blocks, "vec3 field", walberla::Vector3<real_t>(real_t(0)), walberla::field::fzyx, FieldGhostLayers );
            BlockDataID multiComponentFieldID = walberla::field::addToStorage< MultiComponentField_T >( blocks, "multi component field", real_t(0), walberla::field::fzyx, FieldGhostLayers );

            initScalarField(blocks, scalarFieldID);
            initVectorField(blocks, vectorFieldID );
            initMultiComponentField(blocks, multiComponentFieldID );

            // test all interpolators with domain flags everywhere, i.e. without special boundary treatment necessary
            testNearestNeighborFieldInterpolator(blocks, flagFieldID, scalarFieldID, vectorFieldID, multiComponentFieldID);
            testTrilinearFieldInterpolator(blocks, flagFieldID, scalarFieldID, vectorFieldID, multiComponentFieldID);
//            testKernelFieldInterpolator(blocks, flagFieldID, scalarFieldID, vectorFieldID, multiComponentFieldID);

            // set some boundary flags in flag field and invalidate the corresponding scalar field values
            setBoundaryFlags(blocks, flagFieldID, scalarFieldID);

            // test all interpolators' behavior close to boundary cells
            testNearestNeighborFieldInterpolatorAtBoundary(blocks, flagFieldID, scalarFieldID);
            testTrilinearFieldInterpolatorAtBoundary(blocks, flagFieldID, scalarFieldID);
//            testKernelFieldInterpolatorAtBoundary(blocks, flagFieldID, scalarFieldID);

            return 0;
        }

    } // namespace field_interpolation_tests
}

int main( int argc, char **argv ){
    turbine_core::field_interpolation_tests::main(argc, argv);
}