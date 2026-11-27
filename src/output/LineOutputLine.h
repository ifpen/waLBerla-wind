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
//! \file LineOutputLine.h
//! \author Helen SCHOTTENHAMML <helen.schottenhamml@fau.de>
//
//======================================================================================================================

#ifndef TURBINECORE_LINEOUTPUTLINE_H
#define TURBINECORE_LINEOUTPUTLINE_H

#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include <core/config/Config.h>

#include "walberla_helper/field/Field.h"
#include "LineOutputLineBase.h"

namespace turbine_core {

    namespace output {

#ifdef __CUDACC__

        GLOBAL_PREFIX inline void interpolateFieldDataOnGPU( const uint_t nPoints, Vector3<real_t> * points, blockforest::BlockInfo blockInfo,
                                                      field::Field<real_t> field, real_t * results ) {

            projectors::TrilinearFieldInterpolator <field::Field<real_t>> interpolator(blockInfo, &field);

            uint_t fSize = field.fSize();

            if( fSize == 1 ) {
                for (uint_t p = 0; p < nPoints; ++p) {

                    const auto point = points[p];

                    real_t value{};
                    interpolator.get(point, &value);

                    results[p] = value;
                }
            } else if ( fSize == 3 ) {
                for (uint_t p = 0; p < nPoints; ++p) {

                    const auto point = points[p];

                    real_t value[3]{};
                    interpolator.get(point, value);

                    results[3*p+0] = value[0];
                    results[3*p+1] = value[1];
                    results[3*p+2] = value[2];
                }
            }

        }

#endif

        template< typename ScalarField_T, typename VectorField_T >
        class LineOutputLine : public LineOutputLineBase {

        public:

            HOST_PREFIX LineOutputLine ( const walberla::Config::BlockHandle & config,
                                         walberla::StructuredBlockForest * const forest,
                                         const typename LineOutputLineBase::FieldMap & fieldMap )
                    : LineOutputLineBase(config)
            {
                walberla::Config::BlockHandle fields = config.getOneBlock("Fields");
                parseFields(forest, fields, fieldMap);

                if(fields_.size())
                    this->hasValidField_ = true;
            }

        private:

            HOST_PREFIX void parseFields( walberla::StructuredBlockForest * const forest, const walberla::Config::BlockHandle & fields, const typename LineOutputLineBase::FieldMap & fieldMap ) override {

                for( auto & field : fields ) {
                    auto fieldType = Fields::toType(field.first);

                    walberla::IBlock * block{nullptr};

                    for( auto blockIt = forest->begin(); blockIt != forest->end(); ++blockIt ) {
                        if( blockIt.get() != nullptr ) {
                            block = blockIt.get();
                            break;
                        }
                    }

                    if(fieldMap.count(fieldType) != 0) {
                        auto fieldID = fieldMap.at(fieldType);

                        if (block->template isDataOfType<ScalarField_T>(fieldID)) {
                            fields_.emplace_back(fieldID, 1, false);
                        } else if (block->template isDataOfType<VectorField_T>(fieldID)) {
                            fields_.emplace_back(fieldID, 3, false);
                        }

#ifdef __CUDACC__
                        if( block->template isDataOfType<walberla::gpu::GPUField<real_t>>(fieldID) ) {
                            auto * fieldPtr = block->template getData<walberla::gpu::GPUField<real_t>>(fieldID);
                            const uint_t fSize = fieldPtr->fSize();
                            fields_.emplace_back(fieldID,fSize,true);
                        }
#endif
                    } else {
                        WALBERLA_LOG_WARNING_ON_ROOT("Tried to add " << Fields::toString(fieldType) << " field to line output. But there is no such field in the field map." )
                    }
                }
            }

            HOST_PREFIX void interpolateBlockLocalLine( walberla::IBlock * block, walberla::StructuredBlockForest * const forest,
                                                        const typename LineOutputLineBase::Line & points, std::vector< std::vector<real_t> > & results ) override {

                blockforest::BlockInfo blockInfo(block, forest);

                results.resize(fields_.size());

                const uint_t nPoints = points.size();

                for( uint_t f = 0; f < fields_.size(); ++f ) {

                    auto & fieldID = std::get<0>(fields_[f]);
                    const uint_t fSize = std::get<1>(fields_[f]);
                    const bool isGPUField = std::get<2>(fields_[f]);

                    results[f].resize(fSize * nPoints);

                    std::shared_ptr<field::Field<real_t>> fieldPtr;

                    if( block->isDataOfType<ScalarField_T>(fieldID) ) {
                        auto localField = std::make_shared<field::Field<real_t>>(block->getData<ScalarField_T>(fieldID));
                        fieldPtr.swap(localField);
                    } else if( block->isDataOfType<VectorField_T>(fieldID) ) {
                        auto localField = std::make_shared<field::Field<real_t>>(block->getData<VectorField_T>(fieldID));
                        fieldPtr.swap(localField);
                    }

                    if (isGPUField) {
#ifdef __CUDACC__
                        auto localField = std::make_shared<field::Field<real_t>>(block->getData<walberla::gpu::GPUField<real_t>>(fieldID));
                        fieldPtr.swap(localField);
#else
                        WALBERLA_ABORT("Field marked as GPU field but application built without CUDA enabled.")
#endif
                    }

                    if( !isGPUField ) {
                        projectors::TrilinearFieldInterpolator <field::Field<real_t>> interpolator(blockInfo, fieldPtr.get());

                        for (uint_t i = 0; i < nPoints; ++i) {

                            if (fSize == 1) {
                                real_t value{};
                                interpolator.get(points[i], &value);
                                results[f][i] = value;
                            } else if (fSize == 3) {
                                Vector3 <real_t> value{};
                                interpolator.get(points[i], value.data());
                                results[f][i * 3] = value[0];
                                results[f][i * 3 + 1] = value[1];
                                results[f][i * 3 + 2] = value[2];
                            }

                        }
                    } else {

#ifdef __CUDACC__

                        real_t * d_results;
                        cudaMalloc( (void**)&d_results, fSize * nPoints * sizeof(real_t) );

                        Vector3<real_t> * d_points;
                        cudaMalloc( (void**)&d_points, nPoints * sizeof(Vector3<real_t>) );
                        cudaMemcpy( d_points, points.data(), nPoints * sizeof(Vector3<real_t>), cudaMemcpyHostToDevice );

                        gpuErrchk(cudaPeekAtLastError())
                        gpuErrchk(cudaDeviceSynchronize())

                        interpolateFieldDataOnGPU<<<1,1>>>( nPoints, d_points, blockInfo, *(fieldPtr.get()), d_results );

                        gpuErrchk(cudaPeekAtLastError())
                        gpuErrchk(cudaDeviceSynchronize())

                        cudaMemcpy( results[f].data(), d_results, fSize * nPoints * sizeof(real_t), cudaMemcpyDeviceToHost );

                        cudaFree( d_results );
                        cudaFree( d_points );

                        gpuErrchk(cudaPeekAtLastError())
                        gpuErrchk(cudaDeviceSynchronize())

#endif

                    }
                } // loop over fields

            } // function interpolateBlockLocalLine

            HOST_PREFIX void writeFields( std::ostringstream & oss, const uint_t pointIndex, std::vector<std::vector<real_t>> & results ) override {

                for( uint_t f = 0; f < fields_.size(); ++f ) {
                    auto fSize = std::get<1>(fields_[f]);
                    if( fSize == 1 ) {
                        oss << internal::writeFieldItem( results[f][pointIndex] );
                    } else if( fSize == 3 ) {
                        const real_t xVal = results[f][pointIndex*3];
                        const real_t yVal = results[f][pointIndex*3+1];
                        const real_t zVal = results[f][pointIndex*3+2];

                        auto mag = std::sqrt(xVal * xVal + yVal * yVal + zVal * zVal);

                        oss << internal::writeFieldItem(xVal);
                        oss << internal::writeFieldItem(yVal);
                        oss << internal::writeFieldItem(zVal);
                        oss << internal::writeFieldItem(mag);
                    }
                    oss << "\n";
                }

            } // function writeFields

            HOST_PREFIX void writeFieldSpecifier( std::ostringstream & oss, walberla::StructuredBlockForest * const forest ) const override {

                for (auto & field : fields_) {

                    const auto fieldID = std::get<0>(field);
                    const auto fSize = std::get<1>(field);

                    if ( fSize == 1) {
                        oss << internal::writeFieldIdentifier(forest->getBlockDataIdentifier(fieldID));
                    } else if ( fSize == 3 ) {
                        auto fieldBase = forest->getBlockDataIdentifier(fieldID);
                        oss << internal::writeFieldIdentifier(fieldBase, "_X");
                        oss << internal::writeFieldIdentifier(fieldBase, "_Y");
                        oss << internal::writeFieldIdentifier(fieldBase, "_Z");
                        oss << internal::writeFieldIdentifier(fieldBase, "_Mag");
                    }

                }

            } // function writeFieldSpecifier

            std::vector<std::tuple<BlockDataID,uint_t,bool>> fields_;

        }; // class LineOutputLine

    } // namespace output

} // namespace turbine_core

#endif //TURBINECORE_LINEOUTPUTLINE_H