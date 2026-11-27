#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
//#include <likwid-marker.h>
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
#include "gpu/DeviceSelectMPI.h"
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

        size_t heapSize;
        gpuDeviceGetLimit(&heapSize, gpuLimitMallocHeapSize);
        gpuDeviceSetLimit(gpuLimitMallocHeapSize, heapSize * 20);

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
        
        const std::string turbineOutputDirectory{};

        WindFarm_T farm(turbineOutputDirectory, windFarmConfig);
        farm.createGeometry();
        farm.calculateTurbineAABBs();

        /// BLOCKFOREST

        WALBERLA_LOG_INFO_ON_ROOT("Creating blockforest...")

        auto boundariesConfig = globalConfig->getOneBlock( "Boundaries" );
        domain::BoundarySetup boundarySetup(boundariesConfig);

        domain::DomainSetup domainSetup ( globalConfig, boundarySetup.periodicity() );
        
        WALBERLA_ASSERT(!domainSetup.snapshotHandler_->loadSnapshot, "Restart is currently not supported for non-uniform grids.")

        refinement::StaticRefinementHandler<WindFarm_T> staticRefinementHandler{ globalConfig->getOneBlock("DomainSetup"), &farm, boundarySetup.environmentSetup() };
        auto blocks = domainSetup.createStructuredBlockForest(staticRefinementHandler, fieldGhostLayers);

	    WALBERLA_LOG_INFO_ON_ROOT("Initialize Windfarm...")

        farm.setBlockForest(blocks);
        farm.initialiseForceAndControlModel();

        auto domainConfig = globalConfig->getOneBlock( "DomainSetup" );
        const bool outputSetupForest = domainConfig.getParameter<bool>("outputSetupForest");
        const Vector3<uint_t> cellsPerBlock = domainConfig.getParameter<Vector3<uint_t>>("cellsPerBlock");

        //walberla::vtk::writeDomainDecomposition(blocks, "domain_decomposition", "vtk_out", "write_call", true, true, 0);
        if (outputSetupForest) {
            uint_t numBlocks = blocks->getNumberOfBlocks();
            walberla::mpi::reduceInplace(numBlocks, walberla::mpi::SUM);
            uint_t numCells = cellsPerBlock[0] * cellsPerBlock[1] * cellsPerBlock[2] * numBlocks;
            WALBERLA_LOG_INFO_ON_ROOT("Created Forest with " << numBlocks << " Blocks and " << numCells << " cells.")
            WALBERLA_LOG_INFO_ON_ROOT("Expected memory consumption is " << numCells * (27+7)*4 / (1000*1000*1000) << " GB")
            return 0;
        }

        /// FIELDS

        WALBERLA_LOG_INFO_ON_ROOT("Creating fields...")

        //layout - DO NOT CHANGE
        auto layout = codegen::KernelInfo::layout;
        const StorageSpecification_T storageSpec{};

        std::map<output::Fields::Types, BlockDataID> fieldMap {};
        uint_t initialWelfordCounter = 0;

        auto allocator = std::make_shared< walberla::gpu::HostFieldAllocator<real_t> >(); // use pinned memory allocator for faster CPU-GPU memory transfers
        BlockDataID densityFieldCpuID = walberla::field::addToStorage<ScalarField_T>(blocks, "density field CPU", real_t(1), layout, fieldGhostLayers, allocator);
        BlockDataID flagFieldID = walberla::field::addFlagFieldToStorage<FlagField_T>(blocks, "flag field", fieldGhostLayers);
        BlockDataID forceFieldCpuID = walberla::field::addToStorage<VectorField_T>(blocks, "force field CPU", real_t(0), layout, fieldGhostLayers, allocator);
        BlockDataID pdfFieldCpuID = walberla::lbm_generated::addPdfFieldToStorage(blocks, "pdf field CPU", storageSpec, fieldGhostLayers, layout, walberla::Set<walberla::SUID>::emptySet(), walberla::Set<walberla::SUID>::emptySet(), allocator);
        BlockDataID velocityFieldCpuID = walberla::field::addToStorage<VectorField_T>(blocks, "velocity field CPU", real_t(0), layout, fieldGhostLayers, allocator);


        fieldMap[output::Fields::DENSITY] = densityFieldCpuID;
        fieldMap[output::Fields::FLAG] = flagFieldID;
        fieldMap[output::Fields::FORCE] = forceFieldCpuID;
        fieldMap[output::Fields::PDF] = pdfFieldCpuID;
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

        // uTau from MOST
        const real_t uTau = kappa * initialVelocity[0] / std::log(real_t(1) / roughnessLengthRatio);

        /// DOMAIN INITIALISATION

        WALBERLA_LOG_INFO_ON_ROOT("Initialise domain...")

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


        WALBERLA_LOG_INFO_ON_ROOT("Add GPU fields...")

        BlockDataID densityFieldGpuID = walberla::gpu::addGPUFieldToStorage<ScalarField_T>( blocks, densityFieldCpuID, "density field GPU", true );
        BlockDataID forceFieldGpuID = walberla::gpu::addGPUFieldToStorage<VectorField_T>( blocks, forceFieldCpuID, "force field GPU", true );
        BlockDataID pdfFieldGpuID = walberla::lbm_generated::addGPUPdfFieldToStorage<PdfField_T, StorageSpecification_T >(blocks, pdfFieldCpuID, storageSpec, "pdf field GPU");
        BlockDataID velocityFieldGpuID = walberla::gpu::addGPUFieldToStorage<VectorField_T>( blocks, velocityFieldCpuID, "velocity field GPU", true );
        SweepCollection_T sweepCollection( blocks, densityFieldGpuID, forceFieldGpuID, 
                                           pdfFieldGpuID, velocityFieldGpuID, omega );

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
            WALBERLA_CHECK(samplingHeight >= uint_t(0), "Sampling height for wall function boundary conditions must be a positive integer.")
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
        

        
        
        

        /// COMMUNICATION

        WALBERLA_LOG_INFO_ON_ROOT("Set up communication...")

        // create communication for fields
        bool cudaEnabledMPI = parameters.getParameter<bool>( "cudaEnabledMPI", true );
        if(walberla::mpi::MPIManager::instance()->numProcesses() == 1)
            cudaEnabledMPI = false;

        auto communication = std::make_shared<walberla::gpu::communication::NonUniformGPUScheme<CommunicationStencil_T>>(blocks, cudaEnabledMPI);
        auto pdfPackInfo = walberla::lbm_generated::setupNonuniformGPUPdfCommunication<PdfGPUField_T>(blocks, pdfFieldGpuID);
        communication->addPackInfo( pdfPackInfo );
        auto densityPackInfo = std::make_shared<walberla::GeneratedNonUniformGPUFieldPackInfo<GPUField_T<real_t>, walberla::pystencils::waLBerlaWind_DensityPackInfo>>(densityFieldGpuID);
        communication->addPackInfo( densityPackInfo );
        auto velocityPackInfo = std::make_shared<walberla::GeneratedNonUniformGPUFieldPackInfo<GPUField_T<real_t>, walberla::pystencils::waLBerlaWind_VelocityPackInfo>>(velocityFieldGpuID);
        communication->addPackInfo( velocityPackInfo );

        /// TIMELOOP

        WALBERLA_LOG_INFO_ON_ROOT("Creating time loop...")
        using TimingPolicy_T = walberla::timing::DeviceSynchronizePolicy;
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

        walberla::lbm_generated::BasicRecursiveTimeStepGPU<PdfGPUField_T, SweepCollection_T, BoundaryCollection_T, walberla::timeloop::SweepTimeloop<TimingPolicy_T>> recursiveTimestep (
        	blocks, pdfFieldGpuID, sweepCollection, boundaryCollection, communication, pdfPackInfo );

        recursiveTimestep.addPostCollisionFunction([&](uint_t) { farm.callback(Component::Function::UPDATE_DISCRETISATION); }, "Updata Discretisation");
        recursiveTimestep.addPostCollisionFunction([&](uint_t level) {
            std::vector<walberla::Block* > blocksOnLevel;
            blocks->getBlocks(blocksOnLevel, level);
            for (auto block: blocksOnLevel) {
                farm.evaluateDensityAndVelocity(block);
        }}, "Evaluate Density/Velocity");

        recursiveTimestep.addPostCollisionFunction( [&](uint_t){farm.syncNextNeighbour(mpi::SYNC_MACRO, 1, false);}, "Sync Turbine Neighbours");
        recursiveTimestep.addPostCollisionFunction([&](uint_t level) {
            std::vector<walberla::Block* > blocksOnLevel;
            blocks->getBlocks(blocksOnLevel, level);
            for (auto block: blocksOnLevel) {
                flowDriver(block);
        }}, "Flow Driver");

        recursiveTimestep.addPostCollisionFunction([&](uint_t) { farm.callback( Component::Function::CALCULATE_FORCES ); }, "Calculate Forces");
        recursiveTimestep.addPostCollisionFunction([&](uint_t){farm.syncNextNeighbour(mpi::SYNC_FORCE, maxKernelWidth+uint_t(real_t(diameter)/real_t(2.0)), false);}, "Sync Forces");
        recursiveTimestep.addPostCollisionFunction([&](uint_t level) {
            std::vector<walberla::Block* > blocksOnLevel;
            blocks->getBlocks(blocksOnLevel, level);
            for (auto block: blocksOnLevel) {
                farm.spreadForces((block));
            }
        }, "Spread Forces");

        recursiveTimestep.addPostCollisionFunction([&](uint_t) { farm.callback(Component::Function::RESET_COMMUNICATION_DATA); }, "Reset Communication Data");

        recursiveTimestep.addRefinementToTimeLoop( timeloop );

        // LBM stability check
        timeloop.addFuncAfterTimeStep( walberla::makeSharedFunctor( walberla::field::makeStabilityChecker< PdfField_T, FlagField_T >(globalConfig, blocks, pdfFieldCpuID, flagFieldID, FluidFlagUID )), "LBM stability check" );

        // log remaining time
        timeloop.addFuncAfterTimeStep( walberla::timing::RemainingTimeLogger( timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency ), "Remaining time logger" );

        WALBERLA_MPI_WORLD_BARRIER()
        WALBERLA_LOG_INFO_ON_ROOT("Running timeloop...")


        walberla::timing::TimingPool<TimingPolicy_T> timing;

        walberla::WcTimer timer;
        //NVMON_MARKER_INIT;
        //NVMON_MARKER_REGISTER("timeloop");
        timer.start();
        for( uint_t i = 0; i < timesteps; ++i ) {
            //NVMON_MARKER_START("timeloop");
            timeloop.singleStep(timing);
            //NVMON_MARKER_STOP("timeloop");
        }
        timer.end();
        //NVMON_MARKER_CLOSE("timeloop");

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
