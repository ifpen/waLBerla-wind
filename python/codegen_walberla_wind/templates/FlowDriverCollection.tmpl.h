//======================================================================================================================
//
//  This file is part of waLBerla-wind. waLBerla-wind is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla-wind is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla-wind (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \\file {{class_name}}.h
//! \\author lbmpy
//======================================================================================================================

#pragma once

#include <memory>

#include <blockforest/StructuredBlockForest.h>
#include <domain_decomposition/IBlock.h>
#include <core/config/Config.h>
#include <timeloop/ITimeloop.h>

#include "conversion/Conversion.h"
#include "data_interpolator/DataInterpolator.h"
#include "domain/DomainSetup.h"
#include "external_forces/DriverSetup.h"

#include "{{collection_name}}.h"

#ifndef TURBINECORE_{{class_name.upper()}}_H
#define TURBINECORE_{{class_name.upper()}}_H

{% set dynamic = 'DynamicPressureGradient' in driver_classes.keys() -%}

namespace walberla {
    namespace {{namespace}} {

        class {{class_name}} {

        public:

            {{class_name}}( std::shared_ptr<StructuredBlockForest> & blocks, {% if dynamic -%}timeloop::ITimeloop * timeloop,{%- endif %}
                                  std::shared_ptr<Config> & globalConfig,
                                  const turbine_core::domain::DomainSetup & domainSetup,
                                  BlockDataID forceFieldID, BlockDataID velocityFieldID,
                                  const cell_idx_t ghostLayers{% if runtime_gpu_indexing -%}, const int gpuBlockSize0, const int gpuBlockSize1, const int gpuBlockSize2{%- endif %} )
            : sbf_{blocks}, ghostLayers_(ghostLayers){% if dynamic -%}, timeloop_(timeloop){%- endif %}
            {
                const auto parameters = globalConfig->getOneBlock("Parameters");
                driverType_ = turbine_core::external_forces::DriverSetup::toType(parameters.getParameter<std::string>("driverType", "none"));

                const auto defaultValue = std::numeric_limits<{{data_type}}>::quiet_NaN();
                const auto initialisation = globalConfig->getOneBlock("Initialisation");

                // Pressure-driven (static & dynamic)
                const {{data_type}} kappa = parameters.getParameter<{{data_type}}>("kappa", {{data_type}}(0.42));
                const Vector3<{{data_type}}> initialVelocity = initialisation.getParameter<Vector3<{{data_type}}>>("initialVelocity", Vector3<{{data_type}}>(defaultValue));
                const {{data_type}} roughnessLengthRatio = parameters.getParameter<{{data_type}}>("roughnessLengthRatio", defaultValue);
                const auto uTau = kappa * initialVelocity[0] / std::log({{data_type}}(1) / roughnessLengthRatio);
                auto pressureGradient = uTau * uTau / {{data_type}}(domainSetup.domainSize_[2]);

                {{data_type}} windDirection = parameters.getParameter<{{data_type}}>("windDirectionDegrees", {{data_type}}(0));
                windDirection *= turbine_core::math::pi / real_t(180);
                // the kernels only need cos/sin of the wind direction -> evaluate them once here instead of per cell
                const {{data_type}} cosWindDirection = std::cos(windDirection);
                const {{data_type}} sinWindDirection = std::sin(windDirection);

                {% if dynamic -%}
                windDirection_ = windDirection;
                {{data_type}} windDirectionRate{0};
                if (driverType_ == turbine_core::external_forces::DriverSetup::DynamicPressureGradient) {
                    const std::string windDirChangeRateTable = parameters.getParameter<std::string>("windDirectionChangeRateTable", "");
                    initialiseWindDirectionInterpolator(windDirChangeRateTable);
                }
                {%- endif %}

                // Coriolis-driven
                {{data_type}} coriolisFrequency = parameters.getParameter<{{data_type}}>("coriolisFrequency", defaultValue);
                coriolisFrequency *= turbine_core::Conversion::C_t();
                Vector3<{{data_type}}> geostrophicWind = parameters.getParameter<Vector3<{{data_type}}>>("geostrophicWind", Vector3<{{data_type}}>(defaultValue));
                geostrophicWind *= (turbine_core::Conversion::C_t() / turbine_core::Conversion::C_l());
                [[maybe_unused]] {{data_type}} geostrophicWind_x = geostrophicWind[0];
                [[maybe_unused]] {{data_type}} geostrophicWind_y = geostrophicWind[1];
                [[maybe_unused]] {{data_type}} geostrophicWind_z = geostrophicWind[2];

                {% if nonuniform -%}
                const real_t forceScalingFactor{1.0};
                {%- endif %}

                {{collection_object_name}} = std::make_shared<{{collection_name}}>(blocks, {{parameter_list}});

            }

            void run (IBlock * block) {
                {% if nonuniform -%}
                const auto levelDiff = {{data_type}}(sbf_->getDepth() - sbf_->getLevel(*block));
                {{collection_object_name}}->setForcescalingfactor(std::pow({{data_type}}(2), levelDiff));
                {%- endif %}
                switch (driverType_) {
                    {% for driver_enum, driver_class in driver_classes.items() -%}
                    case turbine_core::external_forces::DriverSetup::{{driver_enum}}: {
                    {%- if driver_enum == "DynamicPressureGradient" %}
                        auto windDirChangeRate = windDirChangeRateInterpolator_({{data_type}}(timeloop_->getCurrentTimeStep()));
                        {{collection_object_name}}->setWinddirectionrate(windDirChangeRate);
                        windDirection_ += windDirChangeRate;
                        {{collection_object_name}}->setCoswinddirection(std::cos(windDirection_));
                        {{collection_object_name}}->setSinwinddirection(std::sin(windDirection_));
                    {%- endif %}
                        {{collection_object_name}}->{{driver_class}}(block, ghostLayers_); break; }
                    {% endfor %}
                    default: WALBERLA_ABORT("Unknow flow driver type in flow driver collection.")
                }
            }

            void runOnCellInterval(IBlock * block, const CellInterval & globalCellInterval) {
                {% if nonuniform -%}
                const auto levelDiff = {{data_type}}(sbf_->getDepth() - sbf_->getLevel(*block));
                {{collection_object_name}}->setForcescalingfactor(std::pow({{data_type}}(2), levelDiff));
                {%- endif %}
                switch (driverType_) {
                    {% for driver_enum, driver_class in driver_classes.items() -%}
                    case turbine_core::external_forces::DriverSetup::{{driver_enum}}: {
                    {%- if driver_enum == "DynamicPressureGradient" %}
                        auto windDirChangeRate = windDirChangeRateInterpolator_({{data_type}}(timeloop_->getCurrentTimeStep()));
                        {{collection_object_name}}->setWinddirectionrate(windDirChangeRate);
                        windDirection_ += windDirChangeRate;
                        {{collection_object_name}}->setCoswinddirection(std::cos(windDirection_));
                        {{collection_object_name}}->setSinwinddirection(std::sin(windDirection_));
                    {%- endif %}
                        {{collection_object_name}}->{{driver_class}}CellInterval(block, globalCellInterval); break; }
                    {% endfor %}
                    default: WALBERLA_ABORT("Unknow flow driver type in flow driver collection.")
                }
            }

            void operator()( IBlock * block ) {
                run(block);
            }

            std::function<void (IBlock*)> getSweep() {
                return [this](IBlock* block) { this->run(block); };
            }

            std::function<void (IBlock*)> getSweepOnCellInterval(const shared_ptr<StructuredBlockStorage> & blocks, const CellInterval & globalCellInterval)
            {
                return [this, blocks, globalCellInterval]
                        (IBlock * b)
                { this->runOnCellInterval(b, globalCellInterval); };
            }

        private:

            turbine_core::external_forces::DriverSetup::Type driverType_{};
            std::shared_ptr<StructuredBlockForest> sbf_{};
            const cell_idx_t ghostLayers_;

            {% if dynamic %}
            timeloop::ITimeloop * const timeloop_{};
            {{data_type}} windDirection_{};
            turbine_core::data_interpolation::DataInterpolator<real_t, real_t, false> windDirChangeRateInterpolator_{};
            {% endif %}

            std::shared_ptr<{{collection_name}}> {{collection_object_name}}{};

            {% if dynamic -%}
            void initialiseWindDirectionInterpolator(const std::string & tableName);
            {%- endif %}

        };

        {% if dynamic -%}
        void {{class_name}}::initialiseWindDirectionInterpolator(const std::string & tableName) {
            if(!tableName.empty()) {
                std::ifstream file(tableName);
                std::string line;

                std::vector<real_t> times;
                std::vector<real_t> windDirChangeRate;

                if (!file.is_open()) {
                    WALBERLA_ABORT("Error opening file " << tableName << ", please check file existence.");
                } else {
                    std::getline(file, line); // skip header

                    while (std::getline(file, line)) {
                        bool is_empty = true;
                        for (char ch : line) {
                            is_empty = is_empty && isspace(ch);
                        }
                        if(is_empty)
                            continue;

                        std::stringstream in(line);
                        real_t time, changeRate;
                        in >> time >> changeRate;
                        times.push_back(time  / turbine_core::Conversion::C_t());                    // s as an input
                        windDirChangeRate.emplace_back(changeRate * turbine_core::Conversion::C_t());  // rad/s as an input
                    }

                    file.close();
                }

                windDirChangeRateInterpolator_.setPoints(times.size(), times.data(), windDirChangeRate.data());
            } else {
                WALBERLA_ABORT("Using dynamic pressure gradient as flow driver. Please provide a wind direction change table.")
            };

        }
        {%- endif %}

    } // namespace {{namespace}}
} // namespace walberla


#endif //TURBINECORE_{{class_name.upper()}}_H
