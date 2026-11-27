//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \file Logging.h
//! \author Helen SCHOTTENHAMML <helen.schottenhamml@fau.de>
//
//======================================================================================================================

#ifndef TURBINECORE_LOGGING_H
#define TURBINECORE_LOGGING_H

#pragma once

#include <ctime>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <core/config/Config.h>
#include <core/Hostname.h>
#include <core/timing/TimingPool.h>
#include <core/waLBerlaBuildInfo.h>

#include <core/math/Vector3.h>

#include "wind_turbine_core/utility/CenterHelper.h"

#include "wind_turbine_core/build_info/BuildInfo.h"

#include "conversion/Conversion.h"

namespace turbine_core {

    namespace logging {

        template< typename DomainSetup_T, typename BoundarySetup_T, typename KernelInfo_T, typename TimingPolicy_T >
        class WindTurbineApplicationLogger {

        public:

            using TimingPool_T = walberla::timing::TimingPool<TimingPolicy_T>;
            using TimingPoolPair = std::pair<std::string, TimingPool_T*>;
            using TimingPools = std::vector<TimingPoolPair>;

            WindTurbineApplicationLogger(
                    const DomainSetup_T & domainSetup,
                    const BoundarySetup_T & boundarySetup,
                    const walberla::Config::BlockHandle & windFarmConfig,
                    TimingPool_T * const timing,
                    const std::string & performanceString = "",
                    const bool gpuDirect = false,
                    const std::string & filename = "WindTurbine_Logging.log")
                    : WindTurbineApplicationLogger(domainSetup, boundarySetup, windFarmConfig,
                                                   TimingPools{TimingPoolPair{"Timing", timing}},
                                                   performanceString, gpuDirect, filename)
            {}


            WindTurbineApplicationLogger(
                    const DomainSetup_T & domainSetup,
                    const BoundarySetup_T & boundarySetup,
                    const walberla::Config::BlockHandle & windFarmConfig,
                    const TimingPools & timings = TimingPools{},
                    const std::string & performanceString = "",
                    const bool gpuDirect = false,
                    const std::string & filename = "WindTurbine_Logging.log")
                    : filename_(filename), delimiter(std::string(width_, '/') + "\n"), windFarmConfig_(windFarmConfig)
            {
                WALBERLA_ROOT_SECTION() {

                    // add manipulators
                    oss_ << std::boolalpha << std::setprecision(10);

                    // print header
                    logHeader();

                    // print build info
                    logBuildInfo();
                    oss_ << "\n\n" << delimiter << "\n";

                    // print walberla info
                    logWalberlaInfo();
                    oss_ << "\n\n" << delimiter << "\n";

                    // print cmake info
                    logCMakeInfo();
                    oss_ << "\n\n" << delimiter << "\n";

                    // print cmake info
                    logKernelInfo();
                    oss_ << "\n\n" << delimiter << "\n";

                    // domain setup
                    oss_ << domainSetup << "\n\n" << delimiter << "\n";

                    // boundary setup
                    oss_ << boundarySetup << "\n\n" << delimiter << "\n";

                    // conversion factors
                    logConversionFactors();
                    oss_ << "\n\n" << delimiter << "\n";

                    // wind farm
                    logWindFarm();
                    oss_ << "\n\n" << delimiter << "\n";

                    (void)gpuDirect;
#ifdef __CUDACC__
                    oss_ << "COMMUNICATION MODE:\n\n";
                    oss_ << "GPU direct: " << gpuDirect << "\n\n";
                    oss_ << delimiter << "\n";
#endif

                    if(!timings.empty()) {
                        oss_ << "TIMING POOL INFORMATION:\n\n";

                        // timings
                        for (auto &timing: timings) {
                            oss_ << timing.first << ":\n\n";
                            (timing.second)->print(oss_);
                            oss_ << "\n";
                        }

                        oss_ << delimiter << "\n";
                    }

                    if(!performanceString.empty()) {
                        // performance
                        oss_ << "PERFORMANCE EVALUATION:\n\n";
                        oss_ << performanceString << "\n\n";
                    }
                }
            }

            ~WindTurbineApplicationLogger() {

                WALBERLA_ROOT_SECTION() {

                    // print footer
                    logFooter();

                    // write to file
                    std::ofstream stream;

                    stream.open(filename_, std::ios::out | std::ios::trunc);

                    if (stream.is_open()) {

                        stream << oss_.str() << std::endl;
                        stream.close();

                    } else {

                        WALBERLA_LOG_INFO_ON_ROOT("Could not write logging file.")

                    }
                }

            }

        private:

            const std::string filename_;
            std::ostringstream oss_;

            const int width_ = 160;
            const std::string delimiter;

            const walberla::Config::BlockHandle windFarmConfig_;

            void logHeader() {

                std::time_t t;
                std::time(&t);

                char cTimeString[64];
                std::strftime(cTimeString, 64, "%A, %d. %B %Y, %H:%M:%S", std::localtime(&t));
                std::string timeString(cTimeString);

                oss_ << delimiter;
                oss_ << utility::centered("WIND TURBINE RUN - BEGIN LOGGING", width_) << "\n";
                oss_ << utility::centered(timeString, width_) << "\n"
                     << delimiter << "\n";

            }

            void logFooter() {
                oss_ << delimiter;
                oss_ << utility::centered("WIND TURBINE RUN - END LOGGING", width_) << "\n";
                oss_ << delimiter;
            }

            void logBuildInfo() {
                oss_ << "BUILD INFORMATION:\n";
                oss_ << "\n    - Build type:               " << WALBERLA_BUILD_TYPE
                     << "\n    - Host machine:             " << walberla::getHostName()
                     << "\n    - Build machine:            " << WALBERLA_BUILD_MACHINE
                     << "\n    - Compiler flags:           " << WALBERLA_COMPILER_FLAGS
                     << "\n    - Source directory:         " << TURBINE_CORE_SOURCE_DIR
                     << "\n    - Binary directory:         " << TURBINE_CORE_BUILD_DIR;
            }

            void logWalberlaInfo() {
                oss_ << "VERSION INFORMATION:\n";
                oss_ << "\n    - waLBerla version:         " << WALBERLA_VERSION_STRING
                     << "\n    - waLBerla git SHA1:        " << WALBERLA_GIT_SHA1
                     << "\n    - Turbine  git SHA1:        " << TURBINE_CORE_GIT_SHA1;
            }

            void logCMakeInfo() {
                oss_ << "CMAKE CONFIGURATION:\n";
                oss_ << "\n    - WALBERLA_DOUBLE_ACCURACY:            " << WALBERLA_DOUBLE_ACCURACY_STRING
                     << "\n    - WALBERLA_BUILD_WITH_MPI:             " << WALBERLA_BUILD_WITH_MPI_STRING
                     << "\n    - WALBERLA_BUILD_WITH_OPENMP:          " << WALBERLA_BUILD_WITH_OPENMP_STRING
                     << "\n    - WALBERLA_BUILD_WITH_CUDA:            " << WALBERLA_BUILD_WITH_CUDA_STRING
                     << "\n    - WALBERLA_OPTIMIZE_FOR_LOCALHOST:     " << WALBERLA_OPTIMIZE_FOR_LOCALHOST;
            }

            void logKernelInfo() {
                oss_ << "KERNEL CONFIGURATION:\n";
                oss_ << "\n    - Stencil:                             " << KernelInfo_T::stencil
                     << "\n    - Method:                              " << KernelInfo_T::method
                     << "\n    - Force model:                         " << KernelInfo_T::forceModel
                     << "\n    - Compressible:                        " << KernelInfo_T::compressible
                     << "\n    - ZeroCentering PDFs:                  " << KernelInfo_T::zeroCentered
                     << "\n    - Subgrid-scale model:                 " << KernelInfo_T::subgridScaleModel << "\n";

                oss_ << "\n    - Optimization Collision Rule:         " << KernelInfo_T::lbmOptimisationDict
                     << "\n    - CPU Vectorise Info        :          " << KernelInfo_T::cpuVectoriseInfo;

            }

            void logWindFarm() {

                walberla::Config::Blocks turbineBlocks{};
                windFarmConfig_.getBlocks(turbineBlocks);
                auto numTurbines = turbineBlocks.size();

                oss_ << "WIND FARM INFORMATION:\n";
                oss_ << "\n    - Number of Turbines:                  " << std::to_string(numTurbines);

                for (auto & tc : turbineBlocks) {

                    oss_ << "\n    - " << tc.getKey() << ":";

                    /// general parameter

                    const walberla::Vector3<real_t> bp = tc.getParameter<walberla::Vector3<real_t>>("basePoint");

                    oss_ << "\n        - base point:                 " << bp << "\n";

                    const real_t diameter_LU = tc.getParameter<real_t>("diameter_LU");

                    oss_ << "\n        - diameter (LU):              " << diameter_LU << "\n";

                    /// blade parameter

                    const uint_t nb = tc.getParameter<uint_t>("nBlades");
                    const uint_t npb = tc.getParameter<uint_t>("nPointsBlades");
                    const std::string bd = tc.getParameter<std::string>("bladeDescription");

                    oss_ << "\n        - number of blades:           " << nb
                         << "\n        - number of points per blade: " << npb
                         << "\n        - blade description:          " << bd;

                    const real_t bladePrecone = tc.getParameter<real_t>("bladePrecone");
                    const real_t bladePitch = tc.getParameter<real_t>("bladePitch");

                    oss_ << "\n        - blade pitch:                " << bladePitch
                         << "\n        - blade precone:              " << bladePrecone << "\n";

                    /// tower parameter

                    const uint_t npt = tc.getParameter<uint_t>("nPointsTower");
                    const bool useTowerEffect = tc.getParameter<bool>("towerEffect", false);
                    const std::string towerFile = tc.getParameter<std::string>("towerDescription");

                    if (useTowerEffect) {
                        oss_ << "\n        - tower effect:               " << "true"
                             << "\n        - tower description:           " << towerFile
                             << "\n        - number of points in tower:  " << npt << "\n";
                    } else {
                        oss_ << "\n        - tower effect:               " << "false" << "\n";
                    }

                    /// nacelle

                    const bool useNacelleEffect = tc.getParameter<bool>("nacelleEffect", false);
                    const real_t nacelleYaw = tc.getParameter<real_t>("nacelleYaw");

                    oss_ << "\n        - nacelle effect:             " << (useNacelleEffect ? "true" : "false")
                         << "\n        - nacelle yaw:                " << nacelleYaw << "\n";

                    /// hub

                    const real_t hubTilt = tc.getParameter<real_t>("hubTilt");
                    walberla::Vector3<real_t> hubRotation = tc.getParameter<walberla::Vector3<real_t>> ("hubRotationalVelocity");
                    const real_t hubRadius = tc.getParameter<real_t>("hubRadius_SI");
                    const real_t hubDeport = tc.getParameter<real_t>("hubDeport_SI");

                    oss_ << "\n        - hub rotational velocity:    " << hubRotation
                         << "\n        - hub tilt:                   " << hubTilt
                         << "\n        - hub deport (SI):            " << hubDeport
                         << "\n        - hub radius (SI):            " << hubRadius;

                }

            }

            void logConversionFactors() {

                oss_ << "SIMULATION CONVERSION FACTORS:\n";
                oss_ << "\n    - C_l:                      " << Conversion::C_l()
                     << "\n    - C_t:                      " << Conversion::C_t()
                     << "\n    - C_m:                      " << Conversion::C_m();

            }

        };

    } // namespace logging

} // namespace turbine_core

#endif // TURBINECORE_LOGGING_H
