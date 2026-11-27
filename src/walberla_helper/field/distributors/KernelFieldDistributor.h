#pragma once

#ifndef TURBINECORE_KERNELFIELDDISTRIBUTOR_H
#define TURBINECORE_KERNELFIELDDISTRIBUTOR_H

#include <cassert>
#include <cmath>
#include <algorithm>

#include "wind_turbine_core/ProjectDefines.h"
#include "wind_turbine_core/WalberlaDataTypes.h"

#include "wind_turbine_core/math/Vector3.h"

#include "walberla_helper/blockforest/BlockInfo.h"

namespace turbine_core {

    namespace projectors {

        using std::max;

        template< typename Field_T, typename Kernel_T >
        class KernelFieldDistributor
        {
        public:

            using Kernel = Kernel_T;

            HOST_DEVICE_PREFIX KernelFieldDistributor( const blockforest::BlockInfo & blockInfo,
                                                       Field_T * baseField )
                    : blockInfo_(&blockInfo), baseField_(baseField)
            {
                assert(baseField->nrOfGhostLayers() >= math::max(math::max(static_cast<cell_idx_t>(Kernel_T::xWidth), static_cast<cell_idx_t>(Kernel_T::yWidth)), static_cast<cell_idx_t>(Kernel_T::zWidth))
                    && "field for kernel distribution needs sufficiently many ghost layer");
            }

            template< typename ForwardIterator_T >
            HOST_DEVICE_PREFIX inline void distribute( const Vector3<real_t> & position, ForwardIterator_T distributeValueBegin ) {
                distribute( position[0], position[1], position[2], distributeValueBegin );
            }

            template< typename ForwardIterator_T >
            HOST_DEVICE_PREFIX inline void distribute( const real_t x, const real_t y, const real_t z, ForwardIterator_T distributeValueBegin ) {

                const Vector3<real_t> kernelWidth (Kernel_T::xWidth, Kernel_T::yWidth, Kernel_T::zWidth);
                const auto aabb = blockInfo_->getAABB();
                const auto extendedAABB = aabb.getTranslated(-aabb.min()).getExtended(real_t(baseField_->nrOfGhostLayers()));

                auto centerCell = blockInfo_->getBlockLocalCell( x, y, z );

                assert(extendedAABB.getExtended(kernelWidth).contains(centerCell) && "Distribution position is not contained inside the extended AABB block of this distributor!\n");

                const real_t dx = blockInfo_->dx();
                const real_t dy = blockInfo_->dy();
                const real_t dz = blockInfo_->dz();

                const Vector3<uint_t> neighborhoodSize {uint_t(Kernel_T::xWidth), uint_t(Kernel_T::yWidth), uint_t(Kernel_T::zWidth)};

                math::GenericAABB<cell_idx_t> cellNeighborhood( centerCell[0] - cell_idx_t(neighborhoodSize[0]), centerCell[1] - cell_idx_t(neighborhoodSize[1]), centerCell[2] - cell_idx_t(neighborhoodSize[2]),
                                                                centerCell[0] + cell_idx_t(neighborhoodSize[0]), centerCell[1] + cell_idx_t(neighborhoodSize[1]), centerCell[2] + cell_idx_t(neighborhoodSize[2]) );

                uint_t counter{0};
                real_t sumOfWeights {0};
                real_t sumOfWeightsUnavailable {0};
                real_t weights[(uint_t(1) + uint_t(2) * static_cast<uint_t>(Kernel_T::xWidth)) *
                               (uint_t(1) + uint_t(2) * static_cast<uint_t>(Kernel_T::yWidth)) *
                               (uint_t(1) + uint_t(2) * static_cast<uint_t>(Kernel_T::zWidth))]{};


                // get distribution weights and count available cells in surrounding cells
                for( cell_idx_t zC = cellNeighborhood.zMin(); zC <= cellNeighborhood.zMax(); ++zC ) {
                    for (cell_idx_t yC = cellNeighborhood.yMin(); yC <= cellNeighborhood.yMax(); ++yC) {
                        for (cell_idx_t xC = cellNeighborhood.xMin(); xC <= cellNeighborhood.xMax(); ++xC) {

                            Vector3<cell_idx_t> curCell{xC,yC,zC};
                            Vector3<real_t> curCellCenter = blockInfo_->getBlockLocalCellCenter(curCell);

                            if (extendedAABB.contains(curCell)) {
                                weights[counter] = Kernel_T::getWeight(x, y, z, curCellCenter[0], curCellCenter[1],
                                                                       curCellCenter[2],
                                                                       dx, dy, dz);
                                sumOfWeights += weights[counter];
                            } else {
                                weights[counter] = real_t(0);
                                sumOfWeightsUnavailable += Kernel_T::getWeight(x, y, z, curCellCenter[0], curCellCenter[1], curCellCenter[2],
                                                                              dx, dy, dz);
                            }
                            ++counter;
                        }
                    }
                }

                // check if at least one cell was available, to prevent division by 0
                if (sumOfWeights <= real_t(0))
                    return;

                const real_t totalSum = sumOfWeights + sumOfWeightsUnavailable;

                // scale domain weights if some non-domain cells are in neighborhood and/or if the sum is not one
                const real_t scalingFactor = real_t(1) / totalSum; //(real_t(1) + sumOfWeightsUnavailable / sumOfWeights) / totalSum;

                // distribute the values to the neighboring domain cells with the corresponding (scaled) weighting
                counter = uint_t(0);
                for( cell_idx_t zC = cellNeighborhood.zMin(); zC <= cellNeighborhood.zMax(); ++zC ) {
                    for (cell_idx_t yC = cellNeighborhood.yMin(); yC <= cellNeighborhood.yMax(); ++yC) {
                        for (cell_idx_t xC = cellNeighborhood.xMin(); xC <= cellNeighborhood.xMax(); ++xC) {
                            if (weights[counter] > real_t(0)) {
                                addWeightedCellValue(distributeValueBegin, Vector3<cell_idx_t>(xC, yC, zC),
                                                     scalingFactor * weights[counter]);
                            }
                            ++counter;
                        }
                    }
                }
            }

            auto * getField() const {
                return baseField_;
            }

        private:

            template< typename ForwardIterator_T >
            HOST_DEVICE_PREFIX void addWeightedCellValue( ForwardIterator_T distributeValueBegin, const Vector3<cell_idx_t> & curCell, const real_t & weighting )
            {
                for( uint_t f = 0; f < baseField_->fSize(); ++f )
                {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
                                        atomicAdd(&(baseField_->get(curCell, f)), weighting * (*distributeValueBegin));
                                        __threadfence();
#else
                    baseField_->get( curCell, f) += weighting * (*distributeValueBegin);
#endif

                    ++distributeValueBegin;
                }
            }

            const blockforest::BlockInfo * blockInfo_;

            Field_T * const baseField_;

        };

    }

}

#endif //TURBINECORE_KERNELFIELDDISTRIBUTOR_H
