

#pragma once

#ifndef TURBINECORE_AIRFOILCACHE_H
#define TURBINECORE_AIRFOILCACHE_H

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

#include <fstream>

#include <core/math/Constants.h>

#include "data_interpolator/DataInterpolator.h"
#include "aerodynamics/AirfoilPolar.h"

#include "aerodynamics/AirfoilPolar.h"
#include "conversion/Conversion.h"
#include "gpu/ErrorChecking.h"
#include "gpu/GPUWrapper.h"
#include "wind_turbine_core/ProjectDefines.h"
#include "wind_turbine_core/math/Vector3.h"

namespace turbine_core {

    namespace creator {

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT

        GLOBAL_PREFIX void newAirfoilPolarOnGpu(aerodynamics::AirfoilPolar * const d_airfoilPolar, const uint_t nPoints,
            real_t const * const aoa, Vector2<real_t> const * const polar) {
            *d_airfoilPolar = aerodynamics::AirfoilPolar(nPoints, aoa, polar);
        }

        GLOBAL_PREFIX void deleteAirfoilPolarOnGpu(aerodynamics::AirfoilPolar * const d_airfoilPolar) {
            d_airfoilPolar->~AirfoilPolar();
        }

#endif

        template<typename Topology_T, bool CUBIC>
        class CachedAirfoilPolars {
        public:

            CachedAirfoilPolars() = default;

            ~CachedAirfoilPolars() {
                for (auto & it : cache) {
                    freeDescription<Topology_T::isGPU>(it.second);
                }
            }

            struct Description {
                uint_t nPoints;
                real_t elementLength;
                Vector3<real_t> * positions;
                Vector3<real_t> * angles;
                real_t * widths;
                aerodynamics::AirfoilPolar * airfoilPolars;
            };

            std::unordered_map<std::string, Description> cache;

            HOST_PREFIX void parseDescriptionFile( const std::string & filename, const uint_t & nPoints){

                // early out if file has been cached previously
                if (cache.count(filename)) {
                    WALBERLA_CHECK(cache[filename].nPoints == nPoints, "Number of ALM points in cached turbine description does "
                                                                        "not match current turbine description: " + filename)
                    return;
                }

                std::vector<Vector3<real_t>> positions, angles;
                std::vector<real_t> width;
                std::vector<aerodynamics::AirfoilPolar> airfoilPolars;

                readDescriptionFile(filename, nPoints, positions, angles, width, airfoilPolars);

                const uint_t interpolationPoints_ = positions.size();

                std::vector<real_t> span;
                span.reserve(interpolationPoints_);
                span.push_back(real_t(0.0));

                for(uint_t i = 1; i < interpolationPoints_; ++i) {
                    span.push_back( (positions[i] - positions[i-1]).length() + span.back() );
                }

                Vector3DataInterpolator<CUBIC> positionInterpolator{};
                Vector3DataInterpolator<CUBIC> angleInterpolator{};
                ScalarDataInterpolator<CUBIC> widthInterpolator{};

                positionInterpolator.setPoints(interpolationPoints_, span.data(), positions.data());
                angleInterpolator.setPoints(interpolationPoints_, span.data(), angles.data());
                widthInterpolator.setPoints(interpolationPoints_, span.data(), width.data());

                const auto elementLength = (span[interpolationPoints_-1] - span[0]) / real_t(nPoints);

                auto cpuTurbineDescription = createDescription(nPoints, interpolationPoints_, elementLength,
                                                               span, positionInterpolator, angleInterpolator, widthInterpolator,
                                                               airfoilPolars);
                auto turbineDescription = createFinalDescription<Topology_T::isGPU>(cpuTurbineDescription);

                cache.emplace(filename, turbineDescription);
            }

        private:

            HOST_PREFIX void readDescriptionFile( const std::string & filename, const uint_t nPoints,
                std::vector<Vector3<real_t>> & positions,
                std::vector<Vector3<real_t>> & angles,
                std::vector<real_t> & width,
                std::vector<aerodynamics::AirfoilPolar> & airfoilPolars ) {

                std::string curLine;
                std::ifstream inputFile(filename.c_str());

                if (!inputFile.is_open()) {
                    std::cout << "Error opening file " << filename
                              << ", please check file existence." << std::endl;
                } else {
                    std::getline(inputFile, curLine); // skip header

                    // get first line
                    std::getline(inputFile, curLine);
                    std::stringstream firstIn(curLine);

                    real_t tmpX, tmpY, tmpZ, tmpXA, tmpYA, tmpZA, tmpW;
                    std::string tmpAirfoil;

                    firstIn >> tmpX >> tmpY >> tmpZ >> tmpXA >> tmpYA >> tmpZA >> tmpW >> tmpAirfoil;

                    positions.emplace_back(Vector3<real_t>(tmpX, tmpY, tmpZ) / Conversion::C_l());

                    angles.emplace_back(Vector3<real_t>(tmpXA, tmpYA, tmpZA) / real_t(180.0) * walberla::math::pi);

                    width.push_back(tmpW / Conversion::C_l());

                    auto ap = std::make_shared<aerodynamics::AirfoilPolar>(tmpAirfoil);
                    airfoilPolars.emplace_back(std::move(*ap));

                    std::string lastLine = curLine;
                    while (std::getline(inputFile, curLine)) {
                        bool is_empty = true;
                        for (char ch : curLine) is_empty = is_empty && isspace(ch);
                        if (!is_empty) {
                            lastLine = curLine;
                            if (nPoints) {
                                std::stringstream inStream(curLine);
                                inStream >> tmpX >> tmpY >> tmpZ >> tmpXA >> tmpYA >> tmpZA >> tmpW >> tmpAirfoil;

                                positions.emplace_back(Vector3<real_t>(tmpX, tmpY, tmpZ) / Conversion::C_l());
                                angles.emplace_back(Vector3<real_t>(tmpXA, tmpYA, tmpZA) / real_t(180.0) * walberla::math::pi);
                                width.push_back(tmpW / Conversion::C_l());

                                auto ap2 = std::make_shared<aerodynamics::AirfoilPolar>(tmpAirfoil);
                                airfoilPolars.emplace_back(std::move(*ap2));
                            }
                        }
                    } // end while

                    if(!nPoints) {
                        std::stringstream lastIn(lastLine);

                        lastIn >> tmpX >> tmpY >> tmpZ >> tmpXA >> tmpYA >> tmpZA >> tmpW >> tmpAirfoil;

                        positions.emplace_back(Vector3<real_t>(tmpX, tmpY, tmpZ) / Conversion::C_l());

                        auto tmpAngles = Vector3<real_t>(tmpXA, tmpYA, tmpZA) / real_t(180.0) * walberla::math::pi;
                        angles.push_back(tmpAngles);

                        width.push_back(tmpW / Conversion::C_l());

                        auto ap3 = std::make_shared<aerodynamics::AirfoilPolar>(tmpAirfoil);
                        airfoilPolars.emplace_back(std::move(*ap3));
                    }
                }

            }

            HOST_PREFIX Description createDescription(
                const uint_t nPoints, const uint_t interpolationPoints, const real_t elementLength,
                const std::vector<real_t> & span,
                const Vector3DataInterpolator<CUBIC> & positionInterpolator,
                const Vector3DataInterpolator<CUBIC> & angleInterpolator,
                const ScalarDataInterpolator<CUBIC> & widthInterpolator,
                const std::vector<aerodynamics::AirfoilPolar> & airfoilPolars) {

                Description turbineDescription{};
                turbineDescription.nPoints = nPoints;
                turbineDescription.elementLength = elementLength;
                turbineDescription.positions = (Vector3<real_t>*)malloc((nPoints+2)*sizeof(Vector3<real_t>));
                turbineDescription.angles = (Vector3<real_t>*)malloc((nPoints+2)*sizeof(Vector3<real_t>));

                /* -------------------------------- */
                turbineDescription.widths = (real_t*)malloc(nPoints*sizeof(real_t));
                turbineDescription.airfoilPolars = (aerodynamics::AirfoilPolar*)malloc(nPoints*sizeof(aerodynamics::AirfoilPolar));
                memset(turbineDescription.airfoilPolars, 0, nPoints*sizeof(aerodynamics::AirfoilPolar)); // needed so that pointers within the DataInterpolator are not initialised with garbage
                /* -------------------------------- */

                turbineDescription.positions[0] = positionInterpolator.y()[0];
                turbineDescription.positions[nPoints+1] = positionInterpolator.y()[interpolationPoints-1];
                turbineDescription.angles[0] = angleInterpolator.y()[0];
                turbineDescription.angles[nPoints+1] = angleInterpolator.y()[interpolationPoints-1];
                for(uint_t i = 1; i <= nPoints; ++i) {

                    const real_t tmpSpan = (real_t(i) - real_t(0.5)) * elementLength;
                    uint_t lid{0};
                    for(; lid < interpolationPoints-1; ++lid) {
                        if(tmpSpan < span[lid+1])
                            break;
                    }

                    real_t weight = (tmpSpan - span[lid]) / (span[lid+1] - span[lid]);
                    turbineDescription.airfoilPolars[i-1] = aerodynamics::AirfoilPolar(weight, airfoilPolars[lid], airfoilPolars[lid+1]);
                    turbineDescription.widths[i-1] = widthInterpolator.interpolate(tmpSpan);
                    turbineDescription.positions[i] = positionInterpolator.interpolate(tmpSpan);
                    turbineDescription.angles[i] = angleInterpolator.interpolate(tmpSpan);
                }

                return turbineDescription;
            }

            template<bool isGPU>
            HOST_PREFIX Description createFinalDescription(const Description & cpuDescription, char(*)[isGPU == false] = nullptr) {
                return cpuDescription;
            }

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
	    template<bool isGPU>
            HOST_PREFIX Description createFinalDescription(const Description & cpuDescription, char(*)[isGPU == true] = nullptr) {
                auto & nPoints = cpuDescription.nPoints;

                // copy description on GPU
                Description gpuDescription;
                gpuDescription.nPoints = cpuDescription.nPoints;
                gpuDescription.elementLength = cpuDescription.elementLength;

                // allocate data
                gpuDescription.positions = (Vector3<real_t>*)malloc((nPoints+2)*sizeof(Vector3<real_t>));
                gpuDescription.angles = (Vector3<real_t>*)malloc((nPoints+2)*sizeof(Vector3<real_t>));
                gpuMalloc(&gpuDescription.widths, nPoints*sizeof(real_t));
                gpuMalloc(&gpuDescription.airfoilPolars, nPoints*sizeof(aerodynamics::AirfoilPolar));
                gpuMemset(gpuDescription.airfoilPolars, 0, nPoints*sizeof(aerodynamics::AirfoilPolar));

                // copy data
                memcpy(gpuDescription.positions, cpuDescription.positions, (nPoints+2)*sizeof(Vector3<real_t>));
                memcpy(gpuDescription.angles, cpuDescription.angles, (nPoints+2)*sizeof(Vector3<real_t>));
		gpuMemcpy(&gpuDescription.widths[0], &cpuDescription.widths[0], nPoints*sizeof(real_t), gpuMemcpyHostToDevice);
                // gpuMemcpy(&gpuDescription.airfoilPolars[0], &cpuDescription.airfoilPolars[0], nPoints*sizeof(aerodynamics::AirfoilPolar), gpuMemcpyHostToDevice);

                for (uint_t i = 0; i < nPoints; ++i) {
                    createAirfoilPolarsOnGPU(&gpuDescription.airfoilPolars[i], &cpuDescription.airfoilPolars[i].interpolator);
                }

                free(cpuDescription.positions);
                free(cpuDescription.angles);
                free(cpuDescription.widths);
                free(cpuDescription.airfoilPolars);

                return gpuDescription;
            }
#endif
            template<bool isGPU>
            HOST_PREFIX void freeDescription(Description & description, char(*)[isGPU == false] = nullptr) {
                free(description.positions);
                free(description.angles);

                for (uint_t i = 0; i < description.nPoints; ++i) {
                    description.airfoilPolars[i].~AirfoilPolar();
                }

                free(description.widths);
                free(description.airfoilPolars);
            }

            template<bool isGPU>
            HOST_PREFIX void freeDescription(Description & description, char(*)[isGPU == true] = nullptr) {
                free(description.positions);
                free(description.angles);

                for (uint_t i = 0; i < description.nPoints; ++i) {
                    deleteAirfoilPolars(&description.airfoilPolars[i]);
                }

                gpuFree(description.widths);
                gpuFree(description.airfoilPolars);
            }

#if defined(__NVCC__) || defined(__HIP__)

            HOST_PREFIX void createAirfoilPolarsOnGPU( aerodynamics::AirfoilPolar * d_airfoilPolar, aerodynamics::AirfoilPolar::Interpolator_T * const interpolatorCpu ) {
                const uint_t nPoints = interpolatorCpu->length();

                // copy aoa and polar arrays to GPU
                real_t *d_aoa;
                Vector2<real_t> *d_polar;

                gpuMalloc(&d_aoa, nPoints*sizeof(real_t*));
                gpuMalloc(&d_polar, nPoints*sizeof(Vector2<real_t>*));

                gpuMemcpy(d_aoa, interpolatorCpu->x(), nPoints*sizeof(real_t), gpuMemcpyHostToDevice);
                gpuMemcpy(d_polar, interpolatorCpu->y(), nPoints*sizeof(Vector2<real_t>), gpuMemcpyHostToDevice);

                newAirfoilPolarOnGpu<<<1,1>>>(d_airfoilPolar, nPoints, d_aoa, d_polar);

                gpuFree(d_aoa);
                gpuFree(d_polar);

            }

            HOST_PREFIX void deleteAirfoilPolars( aerodynamics::AirfoilPolar * d_airfoilPolar ) {
                deleteAirfoilPolarOnGpu<<<1,1>>>(d_airfoilPolar);
            }

#else
            HOST_PREFIX void createAirfoilPolarsOnGPU( aerodynamics::AirfoilPolar *, aerodynamics::AirfoilPolar::Interpolator_T * const ) {
                WALBERLA_ABORT("Not implemented.")
            }

            HOST_PREFIX void deleteAirfoilPolars( aerodynamics::AirfoilPolar *) {
                WALBERLA_ABORT("Not implemented.")
            }
#endif

        };

    } // namespace creator

} // namespace turbine_core

#endif //TURBINECORE_AIRFOILCACHE_H
