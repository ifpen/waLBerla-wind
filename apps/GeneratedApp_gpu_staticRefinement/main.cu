#include <iostream>
#include <cmath>
#include <vector>
#include <functional>

#include <blockforest/Initialization.h>
#include <core/all.h>
#include <core/Environment.h>
#include <core/mpi/all.h>
#include <domain_decomposition/all.h>
#include <field/all.h>
#include <geometry/all.h>
#include <lbm/all.h>
#include <lbm/vtk/all.h>
#include <lbm_generated/field/AddToStorage.h>
#include <timeloop/all.h>

#include "conversion/Conversion.h"
#include "domain/BoundaryHandling.h"
#include "domain/BoundarySetter.h"
#include "domain/BoundarySetup.h"
#include "domain/DomainInitialiser.h"
#include "domain/DomainSetup.h"
#include "farm_creator/WindFarm.h"
#include "farm_creator/ConfigTurbineCreator.h"
#include "output/Logging.h"
#include "output/TurbineOutput.h"
#include "walberla_helper/field/all.h"
#include "wind_turbine_core/ProjectDefines.h"

#include "waLBerlaWind_KernelInfo.h"

namespace turbine_core {

    static const uint_t fieldGhostLayers =  2 ;

    // define different interpolators and distributors

    template<class Kernel_T>
    using KernelFieldInterpolator_T = projectors::KernelFieldInterpolator<field::Field<real_t>, Kernel_T>;

    template<class Kernel_T>
    using KernelDistributor_T = projectors::KernelFieldDistributor<field::Field<real_t>, Kernel_T>;

    using InterpolationKernel_T = projectors::SmoothedDiracDeltaKernel;
    using DistributionKernel_T = projectors::SmoothedDiracDeltaKernel;

    using WindFarm_T = WindFarm<topology::gpu::Tree, creator::ConfigTurbineCreator,
            ScalarField_T, VectorField_T, VectorField_T,
            KernelFieldInterpolator_T<InterpolationKernel_T>,
            KernelDistributor_T<DistributionKernel_T>>;

    using TurbineOutput_T = output::TurbineOutput<
            FlagField_T, ScalarField_T, VectorField_T, SecondOrderTensorField_T, ThirdOrderTensorField_T,
            PdfField_T, Stencil_T, StorageSpecification_T::zeroCenteredPDFs, StorageSpecification_T::compressible
    >;

    template<typename Type_T>
    using GPUField_T = walberla::gpu::GPUField<Type_T>;

    int main(int argc, char** argv) {

        walberla::Environment walberlaEnv(argc, argv);
        walberla::gpu::selectDeviceBasedOnMpiRank();

        walberla::logging::Logging::instance()->setLogLevel(walberla::logging::Logging::INFO);

        /// general simulation parameters
        auto globalConfig = walberlaEnv.config();
        auto parameters = globalConfig->getOneBlock("Parameters");

        uint_t timesteps = parameters.getParameter<uint_t>("timesteps", uint_t(10)); ++timesteps;
        const real_t remainingTimeLoggerFrequency = parameters.getParameter<real_t>("remainingTimeLoggerFrequency", real_t(3.0)); // in seconds

        const real_t length_SI = parameters.getParameter<real_t>("length_SI");
        const real_t length_LU = parameters.getParameter<real_t>("length_LU");
        const real_t velocity_SI = parameters.getParameter<real_t>("velocity_SI");
        const real_t velocity_LU = parameters.getParameter<real_t>("velocity_LU");
        const real_t viscosity_SI = parameters.getParameter<real_t>("viscosity_SI");
        const real_t density_SI = parameters.getParameter<real_t>("density_SI");

        Conversion::calculateConversionFactors(length_SI, length_LU, velocity_SI, velocity_LU, density_SI);
        Conversion::print();

//        projectors::GaussianFunction5<>::setSigma(real_t(7.07106781));

        /// parameter conversion
        real_t viscosity_LU = (viscosity_SI / density_SI) * Conversion::C_t() / Conversion::C_l() / Conversion::C_l();

        WALBERLA_CHECK_GREATER(viscosity_LU, real_t(0.0), "Negative LU velocity - Did you calculate the conversion factors?")

        // calculate omega on coarsest level
        const real_t omega = walberla::lbm::collision_model::omegaFromViscosity(viscosity_LU);
        WALBERLA_LOG_INFO_ON_ROOT("omega = " << omega)

        /// TURBINE

        auto windFarmConfig = globalConfig->getBlock("WindFarm");
        const std::string turbineOutputDirectory = windFarmConfig.getParameter<std::string>("outputFolder", "");
        const uint_t forceOutputStart = windFarmConfig.getParameter<uint_t>("startingTimeStepForces", 0);
        const uint_t forceOutputFrequency = windFarmConfig.getParameter<uint_t>("frequencyOutputForces", 1);

        WindFarm_T farm(turbineOutputDirectory, windFarmConfig);
        farm.createGeometry();
        farm.calculateTurbineAABBs();

        /// BLOCKFOREST

        WALBERLA_LOG_INFO_ON_ROOT("Creating blockforest...")

        auto boundariesConfig = globalConfig->getOneBlock( "Boundaries" );
        domain::BoundarySetup boundarySetup(boundariesConfig);

        if (boundarySetup.inflowType() == InflowSetup::ShiftedPeriodic && codegen::KernelInfo::inplace) {
            WALBERLA_ABORT("For the moment, shifted periodicity is not compatible with in-place streaming patterns. "
                     "Please change either inflow/outflow type or streaming pattern.")
        }

        domain::DomainSetup domainSetup ( globalConfig, boundarySetup.periodicity() );
        
        WALBERLA_ASSERT(!domainSetup.snapshotHandler_->loadSnapshot, "Restart is currently not supported for non-uniform grids.")

        refinement::StaticRefinementHandler<WindFarm_T> staticRefinementHandler{ globalConfig->getOneBlock("DomainSetup"), &farm, boundarySetup.environmentSetup() };
        auto blocks = domainSetup.createStructuredBlockForest(staticRefinementHandler, fieldGhostLayers);

        farm.setBlockForest(blocks);
        farm.initialiseForceAndControlModel();

        /// FIELDS

        WALBERLA_LOG_INFO_ON_ROOT("Creating fields...")

        //layout - DO NOT CHANGE
        auto layout = codegen::KernelInfo::layout;
        const StorageSpecification_T storageSpec{};

        std::map<output::Fields::Types, BlockDataID> fieldMap {};
        uint_t initialWelfordCounter = 0;

        auto allocator = std::make_shared< walberla::gpu::HostFieldAllocator<real_t> >(); // use pinned memory allocator for faster CPU-GPU memory transfers
        BlockDataID forceFieldCpuID;
        BlockDataID meanVelocityFieldCpuID;
        BlockDataID pdfFieldCpuID;
        BlockDataID sumOfCubesFieldCpuID;
        BlockDataID sumOfSquaresFieldCpuID;

        if(domainSetup.snapshotHandler_->loadSnapshot) {

        	auto & ssh = domainSetup.snapshotHandler_;
        	
        	auto vectorFieldDataHandling = std::make_shared<field::FlattenedFieldHandling<VectorField_T>> (blocks, fieldGhostLayers, layout, allocator);
        	const auto forceFileName = std::string(std::filesystem::path(ssh->baseFolder) / std::filesystem::path(ssh->forceFile).c_str());
        	
        	forceFieldCpuID = blocks->loadBlockData(forceFileName, vectorFieldDataHandling, "force field CPU");
        	
        	auto pdfFieldDataHandling = std::make_shared< walberla::lbm_generated::internal::PdfFieldHandling< StorageSpecification_T > >( blocks, storageSpec, fieldGhostLayers, layout, allocator );
        	
        	const auto pdfFileName = std::string(std::filesystem::path(ssh->baseFolder) / std::filesystem::path(ssh->pdfFile).c_str());
        	pdfFieldCpuID = blocks->loadBlockData(pdfFileName, pdfFieldDataHandling, "pdf field CPU");
        	

        	if(!ssh->meanVelFile.empty()) {
        		const auto meanVelFileName = std::string(std::filesystem::path(ssh->baseFolder) / std::filesystem::path(ssh->meanVelFile).c_str());
        		initialWelfordCounter = ssh->welfordCounter;

        		meanVelocityFieldCpuID = blocks->loadBlockData(meanVelFileName, vectorFieldDataHandling, "mean velocity field CPU");

        		if(!ssh->sosFile.empty()) {
        			auto secondOrderTensorFieldDataHandling = std::make_shared<field::FlattenedFieldHandling<SecondOrderTensorField_T>> (blocks, fieldGhostLayers, layout, allocator);
        			const auto sosFileName = std::string(std::filesystem::path(ssh->baseFolder) / std::filesystem::path(ssh->sosFile).c_str());
        			sumOfSquaresFieldCpuID = blocks->loadBlockData(sosFileName, secondOrderTensorFieldDataHandling, "sum of squares field");
        		}
        		if(!ssh->socFile.empty()) {
        			auto thirdOrderTensorFieldDataHandling = std::make_shared<field::FlattenedFieldHandling<ThirdOrderTensorField_T>> (blocks, fieldGhostLayers, layout, allocator);
        			const auto socFileName = std::string(std::filesystem::path(ssh->baseFolder) / std::filesystem::path(ssh->socFile).c_str());
        			sumOfCubesFieldCpuID = blocks->loadBlockData(socFileName, thirdOrderTensorFieldDataHandling, "sum of cubes field CPU");
        		}
        	}
        } else {
        	forceFieldCpuID = walberla::field::addToStorage<VectorField_T>(blocks, "force field CPU", real_t(0), layout, fieldGhostLayers, allocator);
        	meanVelocityFieldCpuID = walberla::field::addToStorage<VectorField_T>(blocks, "mean velocity field CPU", real_t(0), layout, fieldGhostLayers, allocator);
        	pdfFieldCpuID = walberla::lbm_generated::addPdfFieldToStorage(blocks, "pdf field CPU", storageSpec, fieldGhostLayers, layout, walberla::Set<walberla::SUID>::emptySet(), walberla::Set<walberla::SUID>::emptySet(), allocator);
        	sumOfCubesFieldCpuID = walberla::field::addToStorage<ThirdOrderTensorField_T>(blocks, "sum of cubes field CPU", real_t(0), layout, fieldGhostLayers, allocator);
        	sumOfSquaresFieldCpuID = walberla::field::addToStorage<SecondOrderTensorField_T>(blocks, "sum of squares field CPU", real_t(0), layout, fieldGhostLayers, allocator);
        }

        BlockDataID densityFieldCpuID = walberla::field::addToStorage<ScalarField_T>(blocks, "density field CPU", real_t(1), layout, fieldGhostLayers, allocator);
        BlockDataID eddyViscosityFieldCpuID = walberla::field::addToStorage<ScalarField_T>(blocks, "eddy viscosity field CPU", real_t(0), layout, fieldGhostLayers, allocator);
        BlockDataID flagFieldID = walberla::field::addFlagFieldToStorage<FlagField_T>(blocks, "flag field", fieldGhostLayers);
        BlockDataID meanEddyViscosityFieldCpuID = walberla::field::addToStorage<ScalarField_T>(blocks, "mean eddy viscosity field CPU", real_t(0), layout, fieldGhostLayers, allocator);
        BlockDataID meanStrainRateFieldCpuID = walberla::field::addToStorage<SecondOrderTensorField_T>(blocks, "mean strain rate field CPU", real_t(0), layout, fieldGhostLayers, allocator);
        BlockDataID omegaFieldCpuID = walberla::field::addToStorage<ScalarField_T>(blocks, "omega field CPU", real_t(0), layout, fieldGhostLayers, allocator);
        BlockDataID strainRateFieldCpuID = walberla::field::addToStorage<SecondOrderTensorField_T>(blocks, "strain rate field CPU", real_t(0), layout, fieldGhostLayers, allocator);
        BlockDataID velocityFieldCpuID = walberla::field::addToStorage<VectorField_T>(blocks, "velocity field CPU", real_t(0), layout, fieldGhostLayers, allocator);


        fieldMap[output::Fields::DENSITY] = densityFieldCpuID;
        fieldMap[output::Fields::EDDY_VISCOSITY] = eddyViscosityFieldCpuID;
        fieldMap[output::Fields::FLAG] = flagFieldID;
        fieldMap[output::Fields::FORCE] = forceFieldCpuID;
        fieldMap[output::Fields::MEAN_EDDY_VISCOSITY] = meanEddyViscosityFieldCpuID;
        fieldMap[output::Fields::MEAN_STRAIN_RATE] = meanStrainRateFieldCpuID;
        fieldMap[output::Fields::MEAN_VELOCITY] = meanVelocityFieldCpuID;
        fieldMap[output::Fields::OMEGA] = omegaFieldCpuID;
        fieldMap[output::Fields::PDF] = pdfFieldCpuID;
        fieldMap[output::Fields::STRAIN_RATE] = strainRateFieldCpuID;
        fieldMap[output::Fields::SUM_OF_CUBES] = sumOfCubesFieldCpuID;
        fieldMap[output::Fields::SUM_OF_SQUARES] = sumOfSquaresFieldCpuID;
        fieldMap[output::Fields::VELOCITY] = velocityFieldCpuID;

        /// ABL calculations

        auto initParams = globalConfig->getOneBlock("Initialisation");
        const auto initType = domain::DomainInitialisation::toType(initParams.getParameter<std::string>("type"));
        const Vector3<real_t> initialVelocity = initParams.getParameter<Vector3<real_t>>("initialVelocity");

        const real_t kappa = parameters.getParameter<real_t>("kappa", real_t(0.42));
        const real_t B = parameters.getParameter<real_t>("B", real_t(5.5));
        const real_t roughnessLengthRatio = parameters.getParameter<real_t>("roughnessLengthRatio", real_t(1e-4));
        const real_t referenceHeight = parameters.getParameter<real_t>("referenceHeight_LU", real_t(-1));
        const uint32_t samplingHeight = parameters.getParameter<uint32_t>("samplingHeight_LU");
        const real_t velocityFilterWidth = parameters.getParameter<real_t>("velocityFilterWidth");
        const real_t roughnessLength = roughnessLengthRatio * referenceHeight;

        const uint_t welfordInterval = walberlaEnv.config()->getOneBlock("Output").getParameter<uint_t>("welfordInterval", uint_t(0));
        // uTau from MOST
        const real_t uTau = kappa * initialVelocity[0] / std::log(real_t(1) / roughnessLengthRatio);

        /// DOMAIN INITIALISATION

        WALBERLA_LOG_INFO_ON_ROOT("Initialise domain...")
        if(!domainSetup.snapshotHandler_->loadSnapshot) {
        			
            const real_t flowDirection = atan(initialVelocity[1] / initialVelocity[0]);
            std::unique_ptr<domain::DomainInitialisation> initialiser{};
            if (initType == domain::DomainInitialisation::UNIFORM) {
                initialiser = std::make_unique<domain::UniformInitialisation>(blocks, initialVelocity);
            } else if (initType == domain::DomainInitialisation::ASMUTH) {
                initialiser = std::make_unique<domain::AsmuthInitialisation>(blocks, flowDirection, uTau, roughnessLength, kappa, B,
                                                                             viscosity_LU, domainSetup.domainSize_, true);
            } else if (initType == domain::DomainInitialisation::LOG_LAW) {
                // asmuth without perturbation velocities
                initialiser = std::make_unique<domain::AsmuthInitialisation>(blocks, flowDirection, uTau, roughnessLength, kappa, B,
                                                                             viscosity_LU, domainSetup.domainSize_, false);
            }
            using PdfSetter_T = walberla::pystencils::waLBerlaWind_PdfSetter;
            PdfSetter_T setter(densityFieldCpuID, forceFieldCpuID, pdfFieldCpuID, velocityFieldCpuID);
            initialiser->setViaVelocityField<VectorField_T, PdfSetter_T>(velocityFieldCpuID, setter);

        	initialiser->setViaVelocityField<VectorField_T, PdfSetter_T>(meanVelocityFieldCpuID, setter);
        } else {

        	if(domainSetup.snapshotHandler_->meanVelFile.empty()) {
        		WALBERLA_ABORT("Mean velocity not loaded from snapshot but also not initialised.")
        	}

        	// initialise velocity field from pfd field if snapshot
        	walberla::pystencils::waLBerlaWind_MacroGetter getter(densityFieldCpuID, forceFieldCpuID, pdfFieldCpuID, velocityFieldCpuID);
        	    for( auto & block : *blocks ) {
                            getter(&block);
                        }
        }

        WALBERLA_LOG_INFO_ON_ROOT("Add GPU fields...")

        BlockDataID densityFieldGpuID = walberla::gpu::addGPUFieldToStorage<ScalarField_T>( blocks, densityFieldCpuID, "density field GPU", true );
        BlockDataID eddyViscosityFieldGpuID = walberla::gpu::addGPUFieldToStorage<ScalarField_T>( blocks, eddyViscosityFieldCpuID, "eddy viscosity field GPU", true );
        BlockDataID forceFieldGpuID = walberla::gpu::addGPUFieldToStorage<VectorField_T>( blocks, forceFieldCpuID, "force field GPU", true );
        BlockDataID meanEddyViscosityFieldGpuID = walberla::gpu::addGPUFieldToStorage<ScalarField_T>( blocks, meanEddyViscosityFieldCpuID, "mean eddy viscosity field GPU", true );
        BlockDataID meanStrainRateFieldGpuID = walberla::gpu::addGPUFieldToStorage<SecondOrderTensorField_T>( blocks, meanStrainRateFieldCpuID, "mean strain rate field GPU", true );
        BlockDataID meanVelocityFieldGpuID = walberla::gpu::addGPUFieldToStorage<VectorField_T>( blocks, meanVelocityFieldCpuID, "mean velocity field GPU", true );
        BlockDataID omegaFieldGpuID = walberla::gpu::addGPUFieldToStorage<ScalarField_T>( blocks, omegaFieldCpuID, "omega field GPU", true );
        BlockDataID pdfFieldGpuID = walberla::lbm_generated::addGPUPdfFieldToStorage<PdfField_T, StorageSpecification_T >(blocks, pdfFieldCpuID, storageSpec, "pdf field GPU");
        BlockDataID strainRateFieldGpuID = walberla::gpu::addGPUFieldToStorage<SecondOrderTensorField_T>( blocks, strainRateFieldCpuID, "strain rate field GPU", true );
        BlockDataID sumOfCubesFieldGpuID = walberla::gpu::addGPUFieldToStorage<ThirdOrderTensorField_T>( blocks, sumOfCubesFieldCpuID, "sum of cubes field GPU", true );
        BlockDataID sumOfSquaresFieldGpuID = walberla::gpu::addGPUFieldToStorage<SecondOrderTensorField_T>( blocks, sumOfSquaresFieldCpuID, "sum of squares field GPU", true );
        BlockDataID velocityFieldGpuID = walberla::gpu::addGPUFieldToStorage<VectorField_T>( blocks, velocityFieldCpuID, "velocity field GPU", true );
        SweepCollection_T sweepCollection( blocks, densityFieldGpuID, eddyViscosityFieldGpuID, forceFieldGpuID, omegaFieldGpuID, 
                                           pdfFieldGpuID, velocityFieldGpuID, omega );

        // for (auto& block : *blocks) {
        //     sweepCollection.initialise(&block);
        //     sweepCollection.initialise(&block, fieldGhostLayers);
        // }
        WALBERLA_MPI_BARRIER()
        WALBERLA_LOG_INFO_ON_ROOT("Initialisation done")

        /// BOUNDARIES

        WALBERLA_LOG_INFO_ON_ROOT("Setting up boundaries...")

        boundarySetup.fillFlagFieldFromConfig<FlagField_T>(blocks, flagFieldID, FluidFlagUID,
                                                           NoSlipFlagUID, WFBFlagUID, SymmetryFlagUID, UniformInflowFlagUID,
                                                           LogLawInflowFlagUID, OutflowFlagUID);

        if(boundarySetup.wallType() == WallSetup::WFB){
            WALBERLA_CHECK(referenceHeight > real_t(0), "Reference height must be given in the parameter file for wall function boundary conditions.")
            WALBERLA_CHECK(boundarySetup.environmentSetup() != EnvironmentSetup::Tunnel,
                           "Wall-function bounce is currently not supported for the tunnel environment")
            // WALBERLA_CHECK(samplingHeight >= uint_t(0), "Sampling height for wall function boundary conditions must be a positive integer.")
            WALBERLA_CHECK(samplingHeight < domainSetup.cellsPerBlock_[2], "Sampling height must be smaller than block height.")
        }
        

        const auto& inflowVelocity = boundarySetup.inflowVelocity();
        auto velocityInit = boundary::velocityInit(roughnessLength, kappa, uTau);
        BoundaryCollection_T boundaryCollection( 
        	blocks, flagFieldID, pdfFieldGpuID, FluidFlagUID, velocityFieldGpuID, 
        	samplingHeight, velocityFilterWidth,  roughnessLengthRatio * referenceHeight,
        	inflowVelocity[0], inflowVelocity[1], inflowVelocity[2],velocityFieldCpuID,  velocityInit,
        	pdfFieldCpuID
        );

        const int periodicShiftValue = boundariesConfig.getParameter<int>("periodicShiftValue", int(real_t(0.33) * real_t(domainSetup.domainSize_[2])));
        // walberla::gpu::ShiftedPeriodicityGPU<PdfGPUField_T> shiftedPeriodicity (blocks, pdfFieldGpuID, fieldGhostLayers, 0, 1, periodicShiftValue);

        /// set IDs for farm
        farm.setFieldIDs(densityFieldGpuID, velocityFieldGpuID, forceFieldGpuID);

        /// field resetter
        
        walberla::pystencils::waLBerlaWind_SoSResetter welfordOutputSosResetter(sumOfSquaresFieldGpuID);

        walberla::pystencils::waLBerlaWind_SoCResetter welfordOutputSocResetter(sumOfCubesFieldGpuID);

        /// WELFORDOUTPUT
        walberla::pystencils::waLBerlaWind_WelfordOutput welfordOutputSweep(
        	meanVelocityFieldGpuID, sumOfCubesFieldGpuID, sumOfSquaresFieldGpuID, velocityFieldGpuID, 
        	real_t(initialWelfordCounter));
        /// WELFORDEDDYVISCOSITY
        walberla::pystencils::waLBerlaWind_WelfordEddyViscosity welfordEddyViscositySweep(
        	eddyViscosityFieldGpuID, meanEddyViscosityFieldGpuID, 
        	real_t(initialWelfordCounter));
        /// WELFORDSTRAINRATE
        walberla::pystencils::waLBerlaWind_WelfordStrainRate welfordStrainRateSweep(
        	meanStrainRateFieldGpuID, strainRateFieldGpuID, 
        	real_t(initialWelfordCounter));

        /// COMMUNICATION

        WALBERLA_LOG_INFO_ON_ROOT("Set up communication...")

        // create communication for fields
        bool gpuEnabledMPI = parameters.getParameter<bool>( "gpuEnabledMPI", false );
        if(walberla::mpi::MPIManager::instance()->numProcesses() == 1)
            gpuEnabledMPI = false;

        auto communication = std::make_shared<walberla::gpu::communication::NonUniformGPUScheme<CommunicationStencil_T>>(blocks, gpuEnabledMPI);
        auto pdfPackInfo = walberla::lbm_generated::setupNonuniformGPUPdfCommunication<PdfGPUField_T>(blocks, pdfFieldGpuID);
        communication->addPackInfo( pdfPackInfo );
        auto densityPackInfo = std::make_shared<walberla::GeneratedNonUniformGPUFieldPackInfo<GPUField_T<real_t>, walberla::pystencils::waLBerlaWind_DensityPackInfo>>(densityFieldGpuID);
        communication->addPackInfo( densityPackInfo );
        auto velocityPackInfo = std::make_shared<walberla::GeneratedNonUniformGPUFieldPackInfo<GPUField_T<real_t>, walberla::pystencils::waLBerlaWind_VelocityPackInfo>>(velocityFieldGpuID);
        communication->addPackInfo( velocityPackInfo );

        /// TIMELOOP

        WALBERLA_LOG_INFO_ON_ROOT("Creating time loop...")

        #ifdef NDEBUG
        using TimingPolicy_T = walberla::timing::WcPolicy;
#else
        using TimingPolicy_T = walberla::timing::DeviceSynchronizePolicy;
#endif
        // create time loop
        auto timeloop = walberla::timeloop::SweepTimeloop<TimingPolicy_T>(blocks->getBlockStorage(), timesteps);

        farm.setTimeloop(&timeloop);

        /// driving forces
        walberla::wind::FlowDriverCollection flowDriver(blocks, &timeloop, globalConfig, domainSetup, forceFieldGpuID, velocityFieldGpuID, fieldGhostLayers);
        for(auto & block : *blocks) { flowDriver(&block); }

        uint_t maxKernelWidth = 1;
        if constexpr (std::is_same_v<WindFarm_T::Interpolator, KernelFieldInterpolator_T<InterpolationKernel_T>>) {
            maxKernelWidth = std::max({maxKernelWidth, InterpolationKernel_T::xWidth, InterpolationKernel_T::yWidth, InterpolationKernel_T::zWidth});
        }
        if constexpr (std::is_same_v<WindFarm_T::Distributor, KernelDistributor_T<DistributionKernel_T >>) {
            maxKernelWidth = std::max({maxKernelWidth, DistributionKernel_T::xWidth, DistributionKernel_T::yWidth, DistributionKernel_T::zWidth});
        }

        walberla::Config::Blocks turbineBlocks{};
        windFarmConfig.getBlocks(turbineBlocks);
        auto numTurbines = turbineBlocks.size();
        uint_t diameter = 0;
        //TODO will we ever have different models during the simulation ?
        for (auto & turbineConfig : turbineBlocks) {
            const std::string aeroModel = turbineConfig.getParameter<std::string>( "aeroModel" );
            if(aeroModel == "rotorDisk") {
                diameter = math::max(diameter, turbineConfig.getParameter<uint_t>("diameter_LU"));
            }
        }

        

        // add LBM sweep and communication to time loop
        // output
        TurbineOutput_T turbineOutput{
                blocks, &timeloop,
                globalConfig->getOneBlock("Output"), FluidFlagUID,
                fieldMap
        };

        walberla::pystencils::waLBerlaWind_StrainRateWriter strainRateWriter(forceFieldGpuID, omegaFieldGpuID, pdfFieldGpuID, strainRateFieldGpuID);

        turbineOutput.addMacroscopicFieldCalculator( [&]() {
        for( auto & block : *blocks ) {
            strainRateWriter(&block);
        }});
        turbineOutput.addMacroscopicFieldCalculator( [&]() {
            walberla::gpu::fieldCpy<ScalarField_T, GPUField_T<real_t>>( blocks, densityFieldCpuID, densityFieldGpuID );
            walberla::gpu::fieldCpy<VectorField_T, GPUField_T<real_t>>( blocks, velocityFieldCpuID, velocityFieldGpuID );
            walberla::gpu::fieldCpy<VectorField_T, GPUField_T<real_t>>( blocks, forceFieldCpuID, forceFieldGpuID );
            walberla::gpu::fieldCpy<VectorField_T, GPUField_T<real_t>>( blocks, meanVelocityFieldCpuID, meanVelocityFieldGpuID );
            walberla::gpu::fieldCpy<SecondOrderTensorField_T, GPUField_T<real_t>>( blocks, sumOfSquaresFieldCpuID, sumOfSquaresFieldGpuID );
            walberla::gpu::fieldCpy<ThirdOrderTensorField_T, GPUField_T<real_t>>( blocks, sumOfCubesFieldCpuID, sumOfCubesFieldGpuID );
            walberla::gpu::fieldCpy<ScalarField_T, GPUField_T<real_t>>( blocks, eddyViscosityFieldCpuID, eddyViscosityFieldGpuID );
            walberla::gpu::fieldCpy<SecondOrderTensorField_T, GPUField_T<real_t>>( blocks, strainRateFieldCpuID, strainRateFieldGpuID );

            gpuDeviceSynchronize();
        });

        

        auto welfordOutputLambda = [&welfordOutputSweep](walberla::IBlock * block) {
            welfordOutputSweep(block);
        };
        auto welfordEddyViscosityLambda = [&welfordEddyViscositySweep](walberla::IBlock * block) {
            welfordEddyViscositySweep(block);
        };
        auto welfordStrainRateLambda = [&welfordStrainRateSweep](walberla::IBlock * block) {
            welfordStrainRateSweep(block);
        };

        walberla::lbm_generated::BasicRecursiveTimeStepGPU<PdfGPUField_T, SweepCollection_T, BoundaryCollection_T, walberla::timeloop::SweepTimeloop<TimingPolicy_T>> recursiveTimestep (
        	blocks, pdfFieldGpuID, sweepCollection, boundaryCollection, communication, pdfPackInfo );

        recursiveTimestep.addPostCollisionFunction([&](uint_t level) { if(level!=0) return; turbineOutput.write(); }, "VTK output");
        recursiveTimestep.addPostCollisionFunction( [&](uint_t){
                          if ((domainSetup.snapshotHandler_->storeSnapshot) && (timeloop.getCurrentTimeStep() % domainSetup.snapshotHandler_->snapshotFrequency == 0) && (timeloop.getCurrentTimeStep() > 0)) {
                             walberla::gpu::fieldCpy<PdfField_T, GPUField_T<real_t>>( blocks, pdfFieldCpuID, pdfFieldGpuID );
                             walberla::gpu::fieldCpy<VectorField_T, GPUField_T<real_t>>( blocks, forceFieldCpuID, forceFieldGpuID );
                             walberla::gpu::fieldCpy<VectorField_T, GPUField_T<real_t>>( blocks, meanVelocityFieldCpuID, meanVelocityFieldGpuID );
                             walberla::gpu::fieldCpy<SecondOrderTensorField_T, GPUField_T<real_t>>( blocks, sumOfSquaresFieldCpuID, sumOfSquaresFieldGpuID );
                             walberla::gpu::fieldCpy<ThirdOrderTensorField_T, GPUField_T<real_t>>( blocks, sumOfCubesFieldCpuID, sumOfCubesFieldGpuID );
                             gpuDeviceSynchronize();
                             domainSetup.snapshotHandler_->createSnapshot(timeloop.getCurrentTimeStep(), blocks, pdfFieldCpuID, forceFieldCpuID,
                                                                          uint_t(welfordOutputSweep.getCounter()), meanVelocityFieldCpuID, sumOfSquaresFieldCpuID, sumOfCubesFieldCpuID);
                          }
                          }, "Snapshot creation");
        	recursiveTimestep.addPostCollisionFunction([&](uint_t){
                           	if(welfordInterval && (uint_t(welfordOutputSweep.getCounter()) % welfordInterval == 0)){
                           		welfordOutputSweep.setCounter(real_t(1));
                           		for ( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt ) {
                           			auto dst = blockIt->getData<GPUField_T<real_t>>( meanVelocityFieldGpuID );
                           			const auto src = blockIt->getData<GPUField_T<real_t>>( velocityFieldGpuID );
                           			gpu::fieldCopy( dst, src );
                           			welfordOutputSosResetter(blockIt.get());
                           			welfordOutputSocResetter(blockIt.get());
                           		}
                           	} else {
                           		welfordOutputSweep.setCounter(real_t(welfordOutputSweep.getCounter() + 1));
                           	}
                           }, "Welford Output Sweep");
        	recursiveTimestep.addPostCollisionFunction([&](uint_t level) {
        std::vector<walberla::Block* > blocksOnLevel;
        blocks->getBlocks(blocksOnLevel, level);
        for (auto block: blocksOnLevel) {
        	welfordOutputLambda(block);
        }}, "Welford Output Sweep");

        	recursiveTimestep.addPostCollisionFunction([&](uint_t){
                           	if(welfordInterval && (uint_t(welfordEddyViscositySweep.getCounter()) % welfordInterval == 0)){
                           		welfordEddyViscositySweep.setCounter(real_t(1));
                           		for ( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt ) {
                           			auto dst = blockIt->getData<GPUField_T<real_t>>( meanEddyViscosityFieldGpuID );
                           			const auto src = blockIt->getData<GPUField_T<real_t>>( eddyViscosityFieldGpuID );
                           			gpu::fieldCopy( dst, src );
                           		}
                           	} else {
                           		welfordEddyViscositySweep.setCounter(real_t(welfordEddyViscositySweep.getCounter() + 1));
                           	}
                           }, "Welford Eddy Viscosity Sweep");
        	recursiveTimestep.addPostCollisionFunction([&](uint_t level) {
        std::vector<walberla::Block* > blocksOnLevel;
        blocks->getBlocks(blocksOnLevel, level);
        for (auto block: blocksOnLevel) {
        	welfordEddyViscosityLambda(block);
        }}, "Welford Eddy Viscosity Sweep");

        	recursiveTimestep.addPostCollisionFunction([&](uint_t){
                           	if(welfordInterval && (uint_t(welfordStrainRateSweep.getCounter()) % welfordInterval == 0)){
                           		welfordStrainRateSweep.setCounter(real_t(1));
                           		for ( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt ) {
                           			auto dst = blockIt->getData<GPUField_T<real_t>>( meanStrainRateFieldGpuID );
                           			const auto src = blockIt->getData<GPUField_T<real_t>>( strainRateFieldGpuID );
                           			gpu::fieldCopy( dst, src );
                           		}
                           	} else {
                           		welfordStrainRateSweep.setCounter(real_t(welfordStrainRateSweep.getCounter() + 1));
                           	}
                           }, "Welford Strain Rate Sweep");
        	recursiveTimestep.addPostCollisionFunction([&](uint_t level) {
        std::vector<walberla::Block* > blocksOnLevel;
        blocks->getBlocks(blocksOnLevel, level);
        for (auto block: blocksOnLevel) {
        	welfordStrainRateLambda(block);
        }}, "Welford Strain Rate Sweep");

        recursiveTimestep.addPostCollisionFunction([&](uint_t) { farm.callback(Component::Function::UPDATE_DISCRETISATION); }, "Rotation");
        recursiveTimestep.addPostCollisionFunction([&](uint_t level) {
        std::vector<walberla::Block* > blocksOnLevel;
        blocks->getBlocks(blocksOnLevel, level);
        for (auto block: blocksOnLevel) {
        	farm.evaluateDensityAndVelocity(block);
        }}, "Evaluate density and velocity");

        recursiveTimestep.addPostCollisionFunction( [&](uint_t){farm.syncNextNeighbour(mpi::SYNC_MACRO, 1, false);}, "Communicate macroscopic turbine data");
        recursiveTimestep.addPostCollisionFunction([&](uint_t) {farm.applyControl();}, "Apply Control");
        recursiveTimestep.addPostCollisionFunction([&](uint_t level) {
        std::vector<walberla::Block* > blocksOnLevel;
        blocks->getBlocks(blocksOnLevel, level);
        for (auto block: blocksOnLevel) {
        	flowDriver(block);
        }}, "Setting driving force");

        recursiveTimestep.addPostCollisionFunction([&](uint_t) { farm.callback( Component::Function::CALCULATE_FORCES ); }, "Calculate forces");
        recursiveTimestep.addPostCollisionFunction([&](uint_t){farm.syncNextNeighbour(mpi::SYNC_FORCE, maxKernelWidth+uint_t(real_t(diameter)/real_t(2.0)), false);}, "Communicate force data");
        recursiveTimestep.addPostCollisionFunction([&](uint_t level) {
        std::vector<walberla::Block* > blocksOnLevel;
        blocks->getBlocks(blocksOnLevel, level);
        for (auto block: blocksOnLevel) {
        	farm.spreadForces(block);
        }}, "Spread forces");

        recursiveTimestep.addPostCollisionFunction([&](uint_t) { farm.callback(Component::Function::RESET_COMMUNICATION_DATA); }, "Reset communication data");

        recursiveTimestep.addRefinementToTimeLoop( timeloop );

        // LBM stability check
        timeloop.addFuncAfterTimeStep( walberla::makeSharedFunctor( walberla::field::makeStabilityChecker< PdfField_T, FlagField_T >(globalConfig, blocks, pdfFieldCpuID, flagFieldID, FluidFlagUID )), "LBM stability check" );

        // log remaining time
        timeloop.addFuncAfterTimeStep( walberla::timing::RemainingTimeLogger( timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency ), "Remaining time logger" );

        WALBERLA_LOG_INFO_ON_ROOT("Running timeloop...")

        WALBERLA_MPI_WORLD_BARRIER()

        walberla::timing::TimingPool<TimingPolicy_T> timing;
        timing.registerTimer("Force Output");
        

        walberla::WcTimer timer;
        timer.start();

        for( uint_t i = 0; i < timesteps; ++i ) {

            timeloop.singleStep(timing);

            timing["Force Output"].start();
            // farm.callback(Component::Output::ORIENTATIONS);
            for(auto & block : *blocks) {
                farm.writeForceOutput(&block, forceOutputStart, forceOutputFrequency);
            }
            farm.writeTurbinePowerAndThrust();
            timing["Force Output"].end();
            

        }

        timer.end();

        double time = timer.max();
        walberla::mpi::reduceInplace(time, walberla::mpi::MAX);

        const auto timeloopTiming = timing.getReduced();
        WALBERLA_LOG_INFO_ON_ROOT("Timeloop timing:\n" << *timeloopTiming)

        walberla::lbm::PerformanceEvaluation<FlagField_T> performance(blocks, flagFieldID, FluidFlagUID);
        performance.logResultOnRoot(timesteps, time);

        farm.calculatePowerAndThrust();

        logging::WindTurbineApplicationLogger<domain::DomainSetup, domain::BoundarySetup, codegen::KernelInfo, TimingPolicy_T> logger {
            domainSetup, boundarySetup, windFarmConfig, &timing, performance.loggingString(timesteps, time)
        };

        return EXIT_SUCCESS;

    } // main

} // namespace turbine_core

int main(int argc, char** argv) {
    return turbine_core::main(argc, argv);
}