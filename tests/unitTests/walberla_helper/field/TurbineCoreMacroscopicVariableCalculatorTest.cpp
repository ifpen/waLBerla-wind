
#include <memory>

#include <core/Environment.h>
#include <core/debug/TestSubsystem.h>

#include <blockforest/Initialization.h>

#include <field/AddToStorage.h>
#include <field/Layout.h>

#include <lbm/field/PdfField.h>
#include <lbm/field/AddToStorage.h>
#include <lbm/lattice_model/all.h>

#include "walberla_helper/field/MacroscopicVariableCalculator.h"
#include "wind_turbine_core/WalberlaDataTypes.h"

namespace turbine_core {

    namespace macroscopic_variable_calculator_test {

        using ScalarField_T = walberla::GhostLayerField<real_t, 1>;
        using VectorField_T = walberla::GhostLayerField<real_t, 3>;

        using CollisionModel_T = walberla::lbm::collision_model::SRT;
        using ForceModel_T = walberla::lbm::force_model::GuoField<VectorField_T>;

        template<typename LM_T>
        using PdfField_T = walberla::lbm::PdfField< LM_T >;

        constexpr real_t omega{1.8};

        uint_t FieldGhostLayers{1};

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
                                                                            walberla::Vector3<real_t>(distU(e2), distU(e2), distU(e2)), distRho(e2));
                )
            }
        }

        void initForceField( const std::shared_ptr<walberla::blockforest::StructuredBlockForest> & forest, const BlockDataID & forceFieldID ) {

            std::random_device rd;
            std::mt19937 e2(rd());
            std::uniform_real_distribution<real_t> dist(real_t(-0.1), real_t(0.1));

            for( auto it = forest->begin(); it != forest->end(); ++it ) {

                auto forceField = it->getData<VectorField_T>(forceFieldID);

                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(forceField,
                                                                 forceField->get(x,y,z,0) = dist(e2);
                                                                 forceField->get(x,y,z,1) = dist(e2);
                                                                 forceField->get(x,y,z,2) = dist(e2);
                )
            }
        }

        template< typename LatticeModel_T >
        void testMacroscopicValueCalculator( const std::shared_ptr<walberla::StructuredBlockForest> & forest, const BlockDataID & forceFieldID ) {

            stencil::initStencilData<typename LatticeModel_T::Stencil>();
            const std::string stencilName{LatticeModel_T::Stencil::NAME};

            LatticeModel_T latticeModel{CollisionModel_T (omega), walberla::lbm::force_model::GuoField<VectorField_T>(forceFieldID)};
            auto pdfFieldID = walberla::lbm::addPdfFieldToStorage(forest, "pdf field " + stencilName, latticeModel, FieldGhostLayers, walberla::field::Layout::fzyx);

            initPDFField<LatticeModel_T>(forest, pdfFieldID);

            // create calculator
            field::DensityAndVelocityCalculator<typename LatticeModel_T::Stencil,PdfField_T<LatticeModel_T>, VectorField_T, LatticeModel_T::compressible> calculatorTC;

            // calculate macroscopic values
            for( auto it = forest->begin(); it != forest->end(); ++it ) {

                // get fields
                auto pdfField = it->getData<PdfField_T<LatticeModel_T>>(pdfFieldID);
                auto forceField = it->getData<VectorField_T>(forceFieldID);

                calculatorTC.setPdfField(pdfField);
                calculatorTC.setForceField(forceField);

                WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(pdfField,
                                                                 // calculate walberla
                                                                 walberla::Vector3<real_t> velocity{};
                                                                 auto density = pdfField->getDensityAndVelocity(velocity, x, y, z);

                                                                 // calculate turbine core
                                                                 Vector3<real_t> velocityTC{};
                                                                 real_t densityTC{};
                                                                 calculatorTC.get(x,y,z,densityTC,velocityTC);

                                                                 // check for equality
                                                                 WALBERLA_ASSERT_FLOAT_EQUAL(density, densityTC, "Custom macroscopic variable calculator failed for density. Stencil = " << stencilName << ". ")
                                                                 WALBERLA_ASSERT_FLOAT_EQUAL(velocity[0], velocityTC[0], "Custom macroscopic variable calculator failed for velocity[0]. Stencil = " << stencilName << ". ")
                                                                 WALBERLA_ASSERT_FLOAT_EQUAL(velocity[1], velocityTC[1], "Custom macroscopic variable calculator failed for velocity[1]. Stencil = " << stencilName << ". ")
                                                                 WALBERLA_ASSERT_FLOAT_EQUAL(velocity[2], velocityTC[2], "Custom macroscopic variable calculator failed for velocity[2]. Stencil = " << stencilName << ". ")
                )

            }

        }

        int main( int argc, char** argv ) {

            walberla::mpi::Environment mpiEnv(argc, argv);
            walberla::debug::enterTestMode();

            const uint_t numberOfBlocksInDirection = 2;
            const uint_t numberOfCellsPerBlockInDirection = 4;
            const real_t dx{1};

            // block storage
            auto blocks = walberla::blockforest::createUniformBlockGrid( numberOfBlocksInDirection, numberOfBlocksInDirection, numberOfBlocksInDirection,
                                                                         numberOfCellsPerBlockInDirection, numberOfCellsPerBlockInDirection, numberOfCellsPerBlockInDirection,
                                                                         dx, 0, false, false,
                                                                         false, false, false,
                                                                         false );

            auto forceFieldID = walberla::field::addToStorage<VectorField_T>(blocks, "force field", real_t(0), walberla::field::Layout::fzyx, FieldGhostLayers);

            // initialise force field with random data
            initForceField(blocks, forceFieldID);

            testMacroscopicValueCalculator<walberla::lbm::D2Q9<CollisionModel_T,false,ForceModel_T>>(blocks, forceFieldID);
            testMacroscopicValueCalculator<walberla::lbm::D2Q9<CollisionModel_T,true,ForceModel_T>>(blocks, forceFieldID);
            testMacroscopicValueCalculator<walberla::lbm::D3Q19<CollisionModel_T,false,ForceModel_T>>(blocks, forceFieldID);
            testMacroscopicValueCalculator<walberla::lbm::D3Q19<CollisionModel_T,true,ForceModel_T>>(blocks, forceFieldID);
            testMacroscopicValueCalculator<walberla::lbm::D3Q27<CollisionModel_T,false,ForceModel_T>>(blocks, forceFieldID);
            testMacroscopicValueCalculator<walberla::lbm::D3Q27<CollisionModel_T,true,ForceModel_T>>(blocks, forceFieldID);

            return EXIT_SUCCESS;
        }

    }

}

int main( int argc, char** argv ) {
    return turbine_core::macroscopic_variable_calculator_test::main(argc, argv);
}
