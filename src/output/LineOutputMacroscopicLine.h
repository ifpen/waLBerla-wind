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
//! \file LineOutputMacroscopicLine.h
//! \author Helen SCHOTTENHAMML <helen.schottenhamml@fau.de>
//
//======================================================================================================================

#ifndef TURBINECORE_LINEOUTPUTMACROSCOPICLINE_H
#define TURBINECORE_LINEOUTPUTMACROSCOPICLINE_H

#pragma once

#include <core/config/Config.h>

#include "walberla_helper/field/Field.h"
#include "LineOutputLineBase.h"

namespace turbine_core {

    namespace output {

#ifdef __CUDACC__

        template<typename Stencil_T, bool zeroCentering, bool compressible>
        GLOBAL_PREFIX void interpolateMacroscopicDataOnGPU( const uint_t nPoints, Vector3<real_t> * points, blockforest::BlockInfo blockInfo,
                                                            field::Field<real_t> pdfField, field::Field<real_t> forceField, real_t * results ) {

            projectors::TrilinearMacroscopicFieldInterpolator <Stencil_T,field::Field<real_t>,field::Field<real_t>, zeroCentering, compressible>
                    interpolator(blockInfo, &pdfField, &forceField);

            for (uint_t p = 0; p < nPoints; ++p) {

                const auto point = points[p];

                real_t value[4]{};
                interpolator.get(point, value);

                results[4*p+0] = value[0];
                results[4*p+1] = value[1];
                results[4*p+2] = value[2];
                results[4*p+3] = value[3];
            }

        }

#endif

        template< typename PdfField_T, typename ForceField_T, typename Stencil_T, bool zeroCentering, bool compressible>
        class LineOutputMacroscopicLine : public LineOutputLineBase {

        public:

            HOST_PREFIX LineOutputMacroscopicLine ( const walberla::Config::BlockHandle & config,
                                                    walberla::StructuredBlockForest * const forest,
                                                    const typename LineOutputLineBase::FieldMap & fieldMap )
                    : LineOutputLineBase(config)
            {
                walberla::Config::BlockHandle fields = config.getOneBlock("Fields");
                parseFields(forest, fields, fieldMap);
                this->hasValidField_ = true;
            }

        private:

            HOST_PREFIX void parseFields( [[maybe_unused]] walberla::StructuredBlockForest * const forest, const walberla::Config::BlockHandle & fields, const typename LineOutputLineBase::FieldMap & fieldMap ) override {
                for( auto & field : fields ) {

                    auto fieldType = Fields::toType(field.first);
                    if( fieldType == Fields::DENSITY || fieldType == Fields::VELOCITY ) {

                        if( fieldMap.count(Fields::PDF) == 0 || fieldMap.count(Fields::FORCE) == 0 ) {
                            WALBERLA_ABORT("Could not find pdf or force field in field map.")
                        }

                        pdfFieldID_ = fieldMap.at(Fields::PDF);
                        forceFieldID_ = fieldMap.at(Fields::FORCE);

#ifdef __CUDACC__
                        walberla::IBlock * block{nullptr};

                        for( auto blockIt = forest->begin(); blockIt != forest->end(); ++blockIt ) {
                            if( blockIt.get() != nullptr ) {
                                block = blockIt.get();
                                break;
                            }
                        }

                        if( block->template isDataOfType<walberla::gpu::GPUField<real_t>>(pdfFieldID_) ) {
                            isGPUField_ = true;
                        }
#endif

                        return;
                    }
                }
            }

            HOST_PREFIX void interpolateBlockLocalLine( walberla::IBlock * block, walberla::StructuredBlockForest * const forest,
                                                        const typename LineOutputLineBase::Line & points, std::vector< std::vector<real_t> > & results ) override {

                blockforest::BlockInfo blockInfo(block, forest);

                field::Field<real_t> pdfField( block->template getData<PdfField_T>(pdfFieldID_) );
                field::Field<real_t> forceField( block->template getData<ForceField_T>(forceFieldID_) );

                const uint_t nPoints = points.size();

                results.resize(1);
                results[0].resize(4 * nPoints);

                if( !isGPUField_ ) {
                    projectors::TrilinearMacroscopicFieldInterpolator <Stencil_T,field::Field<real_t>,field::Field<real_t>,zeroCentering,compressible>
                            interpolator(blockInfo, &pdfField, &forceField);

                    for (uint_t i = 0; i < nPoints; ++i) {

                        real_t value[4]{};
                        interpolator.get(points[i], value);
                        results[0][i * 4] = value[0];
                        results[0][i * 4 + 1] = value[1];
                        results[0][i * 4 + 2] = value[2];
                        results[0][i * 4 + 3] = value[3];

                    }

                } else {

#ifdef __CUDACC__

                    real_t * d_results;
                    cudaMalloc( (void**)&d_results, 4 * nPoints * sizeof(real_t) );

                    Vector3<real_t> * d_points;
                    cudaMalloc( (void**)&d_points, nPoints * sizeof(Vector3<real_t>) );
                    cudaMemcpy( d_points, points.data(), nPoints * sizeof(Vector3<real_t>), cudaMemcpyHostToDevice );

                    gpuErrchk(cudaPeekAtLastError())
                    gpuErrchk(cudaDeviceSynchronize())

                    interpolateMacroscopicDataOnGPU<Stencil_T,zeroCentering,compressible><<<1,1>>>( nPoints, d_points, blockInfo, pdfField, forceField, d_results );

                    gpuErrchk(cudaPeekAtLastError())
                    gpuErrchk(cudaDeviceSynchronize())

                    cudaMemcpy( results[0].data(), d_results, 4 * nPoints * sizeof(real_t), cudaMemcpyDeviceToHost );

                    cudaFree( d_results );
                    cudaFree( d_points );

                    gpuErrchk(cudaPeekAtLastError())
                    gpuErrchk(cudaDeviceSynchronize())

#endif

                }

            }

            HOST_PREFIX void writeFields( std::ostringstream & oss, const uint_t pointIndex, std::vector<std::vector<real_t>> & results ) override {

                // write density
                oss << internal::writeFieldItem( results[0][pointIndex*4] );

                // write velocity
                const real_t xVal = results[0][pointIndex*4+1];
                const real_t yVal = results[0][pointIndex*4+2];
                const real_t zVal = results[0][pointIndex*4+3];

                auto mag = std::sqrt(xVal * xVal + yVal * yVal + zVal * zVal);

                oss << internal::writeFieldItem(xVal);
                oss << internal::writeFieldItem(yVal);
                oss << internal::writeFieldItem(zVal);
                oss << internal::writeFieldItem(mag);

                oss << "\n";

            } // function writeFields

            HOST_PREFIX void writeFieldSpecifier( std::ostringstream & oss, walberla::StructuredBlockForest * const ) const override {

                oss << internal::writeFieldIdentifier("Density");

                auto fieldBase = "Velocity";
                oss << internal::writeFieldIdentifier(fieldBase, "_X");
                oss << internal::writeFieldIdentifier(fieldBase, "_Y");
                oss << internal::writeFieldIdentifier(fieldBase, "_Z");
                oss << internal::writeFieldIdentifier(fieldBase, "_Mag");


            } // function writeFieldSpecifier

            BlockDataID pdfFieldID_{};
            BlockDataID forceFieldID_{};

            bool isGPUField_{false};

        }; // class LineOutputMacroscopicLine

    } // namespace output

} // namespace turbine_core

#endif //TURBINECORE_LINEOUTPUTMACROSCOPICLINE_H