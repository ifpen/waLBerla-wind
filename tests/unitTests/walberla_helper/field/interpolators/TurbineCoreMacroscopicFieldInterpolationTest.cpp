
#include <blockforest/all.h>

#include <core/debug/TestSubsystem.h>
#include <core/mpi/Environment.h>
#include <core/math/Vector3.h>

#include <field/AddToStorage.h>
#include <field/FlagField.h>
#include <field/GhostLayerField.h>
#include <field/interpolators/all.h>

#include <lbm/lattice_model/all.h>
#include <lbm/field/AddToStorage.h>

#include <random>

#include "walberla_helper/field/Field.h"
#include "walberla_helper/field/interpolators/TrilinearFieldInterpolator.h"

namespace turbine_core {

    namespace field_interpolation_tests {

        using namespace walberla;

        const uint_t FieldGhostLayers( 1 );

        using flag_t = walberla::uint8_t;
        using FlagField_T = FlagField<flag_t>;

        typedef GhostLayerField< real_t, 1>          ScalarField_T;
        typedef GhostLayerField< walberla::Vector3<real_t>, 1> VectorField_T;
        typedef GhostLayerField< real_t, 3>          MultiComponentField_T;

        typedef walberla::field::TrilinearFieldInterpolator<ScalarField_T, FlagField_T> ScalarFieldInterpolator_T;
        typedef walberla::field::TrilinearFieldInterpolator<VectorField_T, FlagField_T> Vec3FieldInterpolator_T;

        const FlagUID Domain_Flag ( "domain" );
        const FlagUID Boundary_Flag ( "boundary" );

        using CollisionModel_T = walberla::lbm::collision_model::SRT;
        using ForceModel_T = walberla::lbm::force_model::GuoField<MultiComponentField_T>;

        template<typename LM_T>
        using PdfField_T = walberla::lbm::PdfField< LM_T >;

        constexpr real_t omega{1.8};

        void initFlagField( FlagField_T * field, IBlock * const /*block*/ ) {
            auto domainFlag = field->getOrRegisterFlag( Domain_Flag );
            WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ( field, field->addFlag( x, y, z, domainFlag ); );
        }

        void initForceVectorField( const std::shared_ptr<walberla::blockforest::StructuredBlockForest> & forest, const BlockDataID & forceFieldID ) {

            std::random_device rd;
            std::mt19937 e2(rd());
            std::uniform_real_distribution<real_t> dist(real_t(-0.1), real_t(0.1));

            for( auto it = forest->begin(); it != forest->end(); ++it ) {

                auto forceField = it->getData<VectorField_T>(forceFieldID);

                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(forceField,
                                                                 forceField->get(x,y,z) = walberla::Vector3<real_t>(dist(e2), dist(e2), dist(e2));
                )
            }
        }

        void initForceMulticomponentField( const std::shared_ptr<walberla::blockforest::StructuredBlockForest> & forest, const BlockDataID & forceFieldID ) {

            std::random_device rd;
            std::mt19937 e2(rd());
            std::uniform_real_distribution<real_t> dist(real_t(-5), real_t(5));

            for( auto it = forest->begin(); it != forest->end(); ++it ) {

                auto forceField = it->getData<MultiComponentField_T>(forceFieldID);

                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(forceField,
                                                                 forceField->get(x,y,z,0) = dist(e2);
                                                                         forceField->get(x,y,z,1) = dist(e2);
                                                                         forceField->get(x,y,z,2) = dist(e2);
                )
            }
        }

        template< typename LatticeModel_T >
        void initPDFField( const std::shared_ptr<walberla::blockforest::StructuredBlockForest> & forest, const BlockDataID & pdfFieldID ) {

            std::random_device rd;
            std::mt19937 e2(rd());
            std::uniform_real_distribution<real_t> distRho(real_t(0.9995), real_t(1.0005));
            std::uniform_real_distribution<real_t> distU(real_t(-0.1), real_t(0.1));

            for( auto it = forest->begin(); it != forest->end(); ++it ) {

                auto pdfField = it->getData<PdfField_T<LatticeModel_T>>(pdfFieldID);

                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(pdfField,
                                                                 pdfField->setDensityAndVelocity(x,y,z,
//                                                                           walberla::Vector3<real_t>(0.05, 0, 0), real_t(1));
                                                                           walberla::Vector3<real_t>(distU(e2), distU(e2), distU(e2)), distRho(e2));
                )
            }
        }

        void setBoundaryFlags( const shared_ptr<StructuredBlockStorage> & blocks,
                               const BlockDataID & flagFieldID, const BlockDataID & scalarFieldID ) {

            for( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt )
            {
                auto flagField = blockIt->getData<FlagField_T>( flagFieldID );
                auto valueField = blockIt->getData<ScalarField_T>( scalarFieldID );
                auto domainFlag = flagField->getOrRegisterFlag( Domain_Flag );
                auto boundaryFlag = flagField->getOrRegisterFlag( Boundary_Flag );
                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(flagField,
                                                                 if( x == 2) {
                                                                     flagField->removeFlag(x,y,z,domainFlag);
                                                                     flagField->addFlag(x,y,z,boundaryFlag);
                                                                     valueField->get(x,y,z) = std::numeric_limits<real_t>::max();
                                                                 }
                );
            }
        }

        template<typename LatticeModel_T>
        void testTrilinearMacroscopicFieldInterpolator( const shared_ptr<StructuredBlockForest> & blocks, const BlockDataID &,
                                                        const BlockDataID & densityFieldID, const BlockDataID & velocityFieldID, const BlockDataID & densityInterpolatorID, const BlockDataID & velocityInterpolatorID,
                                                        const BlockDataID & , const BlockDataID & forceMulticomponentFieldID ) {

            const std::string stencilName{LatticeModel_T::Stencil::NAME};

            // check multi component interpolation
            {
                LatticeModel_T latticeModel{CollisionModel_T (omega), ForceModel_T (forceMulticomponentFieldID)};
                auto pdfFieldID = walberla::lbm::addPdfFieldToStorage(blocks, "pdfField " + stencilName, latticeModel,
                                                                      walberla::Vector3<real_t>(0.05), real_t(1), FieldGhostLayers, walberla::field::Layout::fzyx);
                initPDFField<LatticeModel_T>(blocks, pdfFieldID);

                for( auto it = blocks->begin(); it != blocks->end(); ++it ) {

                    // walberla - calculate macroscopic variables
                    auto pdfField = it->getData<PdfField_T<LatticeModel_T>>(pdfFieldID);
                    auto densityField = it->getData<ScalarField_T>(densityFieldID);
                    auto velocityField = it->getData<VectorField_T>(velocityFieldID);

                    auto forceField = it->getData<MultiComponentField_T>(forceMulticomponentFieldID);
                    WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(pdfField,
                                                                     walberla::Vector3<real_t> velocity{};
                                                                     auto density = pdfField->getDensityAndVelocity(velocity, x, y, z);
                                                                     densityField->get(x,y,z) = density;
                                                                     velocityField->get(x,y,z) = velocity;
                    )

                    auto densityInterpolator = it->getData<ScalarFieldInterpolator_T>(densityInterpolatorID);
                    auto velocityInterpolator = it->getData<Vec3FieldInterpolator_T>(velocityInterpolatorID);

                    // turbine core
                    blockforest::BlockInfo blockInfo(&(*it), blocks);

//                    auto forceField = it->getData<MultiComponentField_T>(forceMulticomponentFieldID);

                    field::Field<real_t> pdfFieldTC {pdfField};
                    field::Field<real_t> forceFieldTC {forceField};

                    projectors::TrilinearMacroscopicFieldInterpolator<typename LatticeModel_T::Stencil, field::Field<real_t>, field::Field<real_t>, false, LatticeModel_T::compressible> customInterpolator(
                            blockInfo, &pdfFieldTC, &forceFieldTC);

                    WALBERLA_FOR_ALL_CELLS_XYZ(pdfField,

                                               if(!blockInfo.getAABB().contains(x,y,z)) {
                                                   continue;
                                               }

                                               // walberla
                                               real_t density{};
                                               walberla::Vector3<real_t> velocity(real_t(0));
                                               densityInterpolator->get(x, y, z, &density);
                                               velocityInterpolator->get(x, y, z, &velocity);

                                               // turbine core
                                               real_t interpolatedValueTC[4]{};
                                               customInterpolator.get(x, y, z, interpolatedValueTC);

                                               // compare
                                               WALBERLA_CHECK_FLOAT_EQUAL(density, interpolatedValueTC[0],
                                                                          "TrilinearMacroFieldInterpolator: density interpolation failed");
                                               WALBERLA_CHECK_FLOAT_EQUAL(velocity[0], interpolatedValueTC[1],
                                                                          "TrilinearMacroFieldInterpolator: velocity[0] interpolation failed");
                                               WALBERLA_CHECK_FLOAT_EQUAL(velocity[1], interpolatedValueTC[2],
                                                                          "TrilinearMacroFieldInterpolator: velocity[1] interpolation failed");
                                               WALBERLA_CHECK_FLOAT_EQUAL(velocity[2], interpolatedValueTC[3],
                                                                          "TrilinearMacroFieldInterpolator: velocity[2] interpolation failed");
                    )


                }
            }
        }

/*
        template<typename LatticeModel_T>
        void testTrilinearMacroscopicFieldInterpolatorAtBoundary( const shared_ptr<StructuredBlockForest> & blocks,
                                                                  const BlockDataID & flagFieldID,
                                                                  const BlockDataID & scalarFieldID ) {
            // field interpolators
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
        }*/


        int main(int argc, char **argv) {

            mpi::Environment mpiEnv(argc, argv);
            debug::enterTestMode();

            const uint_t numberOfBlocksInDirection = 1;
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

            // force fields
            BlockDataID forceVectorFieldID = walberla::field::addToStorage< VectorField_T >( blocks, "force vector field", walberla::Vector3<real_t>(0), walberla::field::fzyx, FieldGhostLayers );
            BlockDataID forceMulticomponentFieldID = walberla::field::addToStorage< MultiComponentField_T >( blocks, "force multicomponent field", real_t(0), walberla::field::fzyx, FieldGhostLayers );

            // density and velocity field for waLBerla
            BlockDataID densityFieldID = walberla::field::addToStorage< ScalarField_T >( blocks, "density field", real_c(0), walberla::field::Layout::fzyx, FieldGhostLayers );
            BlockDataID velocityFieldID = walberla::field::addToStorage< VectorField_T >( blocks, "velocity field", walberla::math::Vector3<real_t>(0), walberla::field::Layout::fzyx, FieldGhostLayers );

            // walberla interpolators
            BlockDataID densityInterpolatorID = walberla::field::addFieldInterpolator< ScalarFieldInterpolator_T, FlagField_T >( blocks, densityFieldID, flagFieldID, Domain_Flag );
            BlockDataID velocityInterpolatorID = walberla::field::addFieldInterpolator< Vec3FieldInterpolator_T, FlagField_T >( blocks, velocityFieldID, flagFieldID, Domain_Flag );

            initForceVectorField(blocks, forceVectorFieldID);
            initForceMulticomponentField(blocks, forceMulticomponentFieldID);

            // test all interpolators with domain flags everywhere, i.e. without special boundary treatment necessary
            testTrilinearMacroscopicFieldInterpolator<walberla::lbm::D3Q19<CollisionModel_T,false,ForceModel_T>>(blocks, flagFieldID, densityFieldID, velocityFieldID, densityInterpolatorID, velocityInterpolatorID, forceVectorFieldID, forceMulticomponentFieldID);
            testTrilinearMacroscopicFieldInterpolator<walberla::lbm::D3Q19<CollisionModel_T,true,ForceModel_T>>(blocks, flagFieldID, densityFieldID, velocityFieldID, densityInterpolatorID, velocityInterpolatorID, forceVectorFieldID, forceMulticomponentFieldID);
            testTrilinearMacroscopicFieldInterpolator<walberla::lbm::D3Q27<CollisionModel_T,false,ForceModel_T>>(blocks, flagFieldID, densityFieldID, velocityFieldID, densityInterpolatorID, velocityInterpolatorID, forceVectorFieldID, forceMulticomponentFieldID);
            testTrilinearMacroscopicFieldInterpolator<walberla::lbm::D3Q27<CollisionModel_T,true,ForceModel_T>>(blocks, flagFieldID, densityFieldID, velocityFieldID, densityInterpolatorID, velocityInterpolatorID, forceVectorFieldID, forceMulticomponentFieldID);

            // set some boundary flags in flag field and invalidate the corresponding scalar field values
//            setBoundaryFlags(blocks, flagFieldID, scalarFieldID);

            // test all interpolators' behavior close to boundary cells
//            testTrilinearMacroscopicFieldInterpolatorAtBoundary(blocks, flagFieldID, scalarFieldID);

            return 0;
        }

    } // namespace field_interpolation_tests
}

int main( int argc, char **argv ){
    turbine_core::field_interpolation_tests::main(argc, argv);
}