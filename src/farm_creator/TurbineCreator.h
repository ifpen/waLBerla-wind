
#pragma once

#ifndef TURBINECORE_TURBINECREATOR_H
#define TURBINECORE_TURBINECREATOR_H

#include <vector>

#include "wind_turbine_core/ProjectDefines.h"

#include "component/Component.h"

namespace turbine_core {

    namespace creator {

        template <typename Topology_T>
        class TurbineCreator {

        public:

            HOST_PREFIX TurbineCreator() {}

            HOST_PREFIX void createGeometry( std::vector<std::shared_ptr<Topology_T>> & turbines) const {
                doCreateGeometry(turbines);
            }

            HOST_PREFIX void initialiseForceAndControlModel( std::vector<std::shared_ptr<Topology_T>> & turbines, const std::vector<AABB> & turbineAABBs,
                                                             const std::shared_ptr<domain::TurbineDomain> & domain,
                                                             std::vector<bool> &isControlled,
                                                             std::vector <uint_t> &nPointsComponents,
                                                             std::vector <uint_t> &nComponents,
                                                             std::vector <uint_t> &nPointsComponentsForSpread) const {
                doInitialiseForceAndControlModel(turbines, turbineAABBs, domain, isControlled, nPointsComponents,
                                                 nComponents, nPointsComponentsForSpread);
            }

            HOST_PREFIX virtual ~TurbineCreator() {}

        private:

            HOST_PREFIX virtual void doCreateGeometry( std::vector<std::shared_ptr<Topology_T>> & turbines)  const = 0;
            HOST_PREFIX virtual void doInitialiseForceAndControlModel( std::vector<std::shared_ptr<Topology_T>> & turbines,
                                                                       const std::vector<AABB> & turbineAABBs,
                                                                       const std::shared_ptr<domain::TurbineDomain> & domain,
                                                                       std::vector<bool> &isControlled,
                                                                       std::vector <uint_t> &nPointsComponents,
                                                                       std::vector <uint_t> &nComponents,
                                                                       std::vector <uint_t> &nPointsComponentsForSpread) const = 0;

        };

    } // namespace creator

} // namespace turbine_core

#endif //TURBINECORE_TURBINECREATOR_H
