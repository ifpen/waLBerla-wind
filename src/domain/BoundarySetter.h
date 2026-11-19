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
//! \file BoundarySetter.h
//! \author Helen SCHOTTENHAMML <helen.schottenhamml@fau.de>
//
//======================================================================================================================

#ifndef TURBINECORE_BOUNDARYSETTER_H
#define TURBINECORE_BOUNDARYSETTER_H

#pragma once

#include <blockforest/StructuredBlockForest.h>
#include <blockforest/PhantomBlockForest.h>

#include "EnvironmentSetup.h"

namespace turbine_core {

    namespace domain {

        template< typename BoundaryHandling_T >
        class BoundarySetter {

        public:

            HOST_PREFIX BoundarySetter(const std::shared_ptr<walberla::StructuredBlockForest> & storage,
                                       const EnvironmentSetup::Type & setup, const InflowSetup::Type & inflowType, const OutflowSetup::Type & outflowType,
                                       const WallSetup::Type & wallType,
                                       const BlockDataID & boundaryHandlingId, const uint_t nGhostLayers,
                                       const walberla::FlagUID & FluidFlagUID,
                                       const walberla::FlagUID & NoSlipFlagUID,
                                       const walberla::FlagUID & WFBFlagUID,
                                       const walberla::FlagUID & SymmetryFlagUID,
                                       const walberla::FlagUID & UniformInflowFlagUID,
                                       const walberla::FlagUID & LogLawInflowFlagUID,
                                       const walberla::FlagUID & OutflowFlagUID )
                    : storage_(storage), setup_(setup), inflowType_(inflowType), outflowType_(outflowType), wallType_(wallType),
                      boundaryHandlingId_(boundaryHandlingId),
                      ghostLayers_(nGhostLayers) {
                setBoundaries(FluidFlagUID, NoSlipFlagUID, WFBFlagUID, SymmetryFlagUID, UniformInflowFlagUID, LogLawInflowFlagUID, OutflowFlagUID);
            }

            HOST_PREFIX void setBoundaries(const walberla::FlagUID & FluidFlagUID,
                                           const walberla::FlagUID & NoSlipFlagUID,
                                           const walberla::FlagUID & WFBFlagUID,
                                           const walberla::FlagUID & SymmetryFlagUID,
                                           const walberla::FlagUID & UniformInflowFlagUID,
                                           const walberla::FlagUID & LogLawInflowFlagUID,
                                           const walberla::FlagUID & OutflowFlagUID);

        private:

            const std::shared_ptr<walberla::StructuredBlockForest> & storage_;
            const EnvironmentSetup::Type setup_;
            const InflowSetup::Type inflowType_;
            const OutflowSetup::Type outflowType_;
            const WallSetup::Type wallType_;

            const BlockDataID boundaryHandlingId_;

            const uint_t ghostLayers_;

        };

        template< typename BoundaryHandling_T >
        HOST_PREFIX void BoundarySetter<BoundaryHandling_T>::setBoundaries(const walberla::FlagUID & FluidFlagUID,
                                                                           const walberla::FlagUID & NoSlipFlagUID,
                                                                           const walberla::FlagUID & WFBFlagUID,
                                                                           const walberla::FlagUID & SymmetryFlagUID,
                                                                           const walberla::FlagUID & UniformInflowFlagUID,
                                                                           const walberla::FlagUID & LogLawInflowFlagUID,
                                                                           const walberla::FlagUID & OutflowFlagUID) {

            for (auto block = storage_->begin(); block != storage_->end(); ++block) {

                BoundaryHandling_T * handling = block->getData<BoundaryHandling_T>(boundaryHandlingId_);

                // FLUID

                const uint_t level = storage_->getLevel(*block);
                walberla::CellInterval domainBB = storage_->getDomainCellBB(level);
                storage_->transformGlobalToBlockLocalCellInterval(domainBB, *block);

                auto ghost = cell_idx_t(ghostLayers_);
                domainBB.expand(ghost);

                // remaining domain
                handling->forceDomain(domainBB);

                /// get all the boundaries

                // WEST - Inflow
                walberla::CellInterval west(domainBB.xMin(), domainBB.yMin(), domainBB.zMin(),
                                            domainBB.xMin() + ghost - cell_idx_t(1), domainBB.yMax(), domainBB.zMax());

                // EAST - Outflow
                walberla::CellInterval east(domainBB.xMax() - ghost + cell_idx_t(1), domainBB.yMin(), domainBB.zMin(),
                                            domainBB.xMax(), domainBB.yMax(), domainBB.zMax());

                // BOTTOM - NoSlip
                walberla::CellInterval bottom(domainBB.xMin(), domainBB.yMin(), domainBB.zMin(),
                                              domainBB.xMax(), domainBB.yMax(),
                                              domainBB.zMin() + ghost - cell_idx_t(1));

                // TOP - NoSlip or FreeSlip
                walberla::CellInterval top(domainBB.xMin(), domainBB.yMin(), domainBB.zMax() - ghost + cell_idx_t(1),
                                           domainBB.xMax(), domainBB.yMax(), domainBB.zMax());

                // FRONT - NoSlip or Periodic
                walberla::CellInterval front(domainBB.xMin(), domainBB.yMin(), domainBB.zMin(),
                                             domainBB.xMax(), domainBB.yMin() + ghost - cell_idx_t(1), domainBB.zMax());

                // BACK - NoSlip or Periodic
                walberla::CellInterval back(domainBB.xMin(), domainBB.yMax() - ghost + cell_idx_t(1), domainBB.zMin(),
                                            domainBB.xMax(), domainBB.yMax(), domainBB.zMax());

                if (inflowType_ == InflowSetup::InflowUniform) {
                    handling->forceBoundary(UniformInflowFlagUID, west);
                } else if (inflowType_ == InflowSetup::InflowLogLaw) {
                    handling->forceBoundary(LogLawInflowFlagUID, west);
                }
                if (outflowType_ == OutflowSetup::Outflow) {
                    handling->forceBoundary(OutflowFlagUID, east);
                }

                switch (setup_) {
                    case EnvironmentSetup::Periodic :
                        break;
                    case EnvironmentSetup::Tunnel :
                        if (wallType_ == WallSetup::WFB)
                            WALBERLA_LOG_WARNING_ON_ROOT("WFB cannot be used for Tunnel setup. Defaulting to no-slip.")
                        handling->forceBoundary(NoSlipFlagUID, bottom);
                        handling->forceBoundary(NoSlipFlagUID, top);
                        handling->forceBoundary(NoSlipFlagUID, front);
                        handling->forceBoundary(NoSlipFlagUID, back);
                        break;
                    case EnvironmentSetup::Open:
                        // front: periodic
                        // back: periodic
                        if (wallType_ == WallSetup::NoSlip)
                            handling->forceBoundary(NoSlipFlagUID, bottom);
                        if (wallType_ == WallSetup::WFB)
                            handling->forceBoundary(WFBFlagUID, bottom);
                        handling->forceBoundary(SymmetryFlagUID, top);
                        break;
                    case EnvironmentSetup::FreeSlip:
                        // front: periodic
                        // back: periodic
                        handling->forceBoundary(SymmetryFlagUID, bottom);
                        handling->forceBoundary(SymmetryFlagUID, top);
                        break;
                    default: WALBERLA_ABORT("Unknown Boundary Setup in setFlags.")
                }

            }
        }

    } // domain

} // turbine_core


#endif //TURBINECORE_BOUNDARYSETTER_H
