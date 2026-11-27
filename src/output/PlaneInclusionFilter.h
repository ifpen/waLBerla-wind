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
//! \file PlaneInclusionFilter.h
//! \author Helen SCHOTTENHAMML <helen.schottenhamml@fau.de>
//
//======================================================================================================================

#ifndef TURBINECORE_PLANEINCLUSIONFILTER_H
#define TURBINECORE_PLANEINCLUSIONFILTER_H

#pragma once

#include <cmath>

#include <core/cell/CellSet.h>
#include <core/math/Vector3.h>
#include <domain_decomposition/IBlock.h>
#include <domain_decomposition/StructuredBlockStorage.h>

#include "wind_turbine_core/WalberlaDataTypes.h"

namespace turbine_core {

    namespace output {

        class PlaneInclusionFilter {

            using Vector_T = walberla::Vector3<real_t>;

        public:

            PlaneInclusionFilter( const Vector_T & point, const Vector_T & normal,
                                  const real_t maxDistance = real_t(0.5) )
            : point_(point), normal_(normal.getNormalizedOrZero()), maxDistance_(maxDistance)
            {
                WALBERLA_ASSERT_FLOAT_UNEQUAL( normal_.length(), real_t(0), "Normal vector must not be null vector!" )
            }

            void operator()(walberla::CellSet & filteredCells, const walberla::IBlock & block,
                            const walberla::StructuredBlockStorage & storage, const uint_t ghostLayers) {

                const auto gl    = cell_idx_t( ghostLayers );
                const auto begin = cell_idx_t( -1 ) * gl;

                const auto xEnd = cell_idx_t( storage.getNumberOfXCells(block) ) + gl;
                const auto yEnd = cell_idx_t( storage.getNumberOfYCells(block) ) + gl;
                const auto zEnd = cell_idx_t( storage.getNumberOfZCells(block) ) + gl;

                for( cell_idx_t z = begin; z < zEnd; ++z ) {
                    for (cell_idx_t y = begin; y < yEnd; ++y) {
                        for (cell_idx_t x = begin; x < xEnd; ++x) {

                            // get global cell
                            walberla::Cell cell{ x,y,z };
                            storage.transformBlockLocalToGlobalCell(cell, block);

                            // calculate distance of cell center to plane
                            auto cellCenter = storage.getCellCenter(cell, storage.getLevel(block));
                            const auto v = point_ - cellCenter;

                            const real_t distance = std::abs(v * normal_);

                            // if cell center is close enough to plane, add to filteredCells
                            if (distance <= maxDistance_) {
                                filteredCells.insert(x, y, z);
                            }
                        }
                    }
                }

            }

        private:

            const Vector_T point_;
            const Vector_T normal_;

            const real_t maxDistance_;

        }; // class PlaneInclusionFilter

    } // namespace output

} // namespace turbine_core

#endif // TURBINECORE_PLANEINCLUSIONFILTER_H 
