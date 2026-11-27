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
//! \file TurbineBeforeFunctors.h
//! \author Helen SCHOTTENHAMML <helen.schottenhamml@fau.de>
//
//======================================================================================================================

#ifndef TURBINECORE_TURBINEBEFOREFUNCTORS_H
#define TURBINECORE_TURBINEBEFOREFUNCTORS_H

#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <timeloop/Timeloop.h>

#include "wind_turbine_core/WalberlaDataTypes.h"

namespace turbine_core {

    namespace output {

        class GenericTurbineBeforeFunctor {

            using BeforeFunction_T = std::function< void () >;

        public:

            GenericTurbineBeforeFunctor( const BeforeFunction_T & beforeFunction, walberla::timeloop::ITimeloop * const timeloop )
                    : beforeFunction_(beforeFunction), timeloop_(timeloop)
            {}

            void operator()() {

                for( const auto & interval : outputInterval_ ) {

                    const auto outputInterval = std::get<0>(interval);

                    // never output -> never call function
                    if( outputInterval == 0 )
                        continue;
                    // account for potential starting time step
                    uint_t shiftedTimeStep = timeloop_->getCurrentTimeStep() - std::get<2>(interval); // shift by startingTimeStep

                    auto n = uint_t(std::floor(real_t(shiftedTimeStep) / real_t(outputInterval)));
                    uint_t off = shiftedTimeStep - n * outputInterval;

                    // not in sampling interval -> no action necessary
                    if( off > std::get<1>(interval) )
                        continue;

                    beforeFunction_();
                    // only call function once per timestep
                    return;

                }

            }

            void addInterval( const uint_t interval ) {
                outputInterval_.emplace_back(std::tuple<uint_t,uint_t,uint_t>(interval, uint_t(0), uint_t(0)));
            }

            void addInterval( const std::pair<uint_t, uint_t> & interval ) {
                outputInterval_.emplace_back(std::tuple<uint_t,uint_t,uint_t>(interval.first, interval.second, uint_t(0)));
            }

            void addInterval( const std::vector<std::pair<uint_t, uint_t>> & intervalVector ) {
                for( const auto & interval : intervalVector ) {
                    outputInterval_.emplace_back(std::tuple<uint_t,uint_t,uint_t>(interval.first, interval.second, uint_t(0)));
                }
            }

            void addInterval( const std::tuple<uint_t, uint_t, uint_t> & interval ) {
                outputInterval_.emplace_back(interval);
            }

            void addInterval( const std::vector<std::tuple<uint_t, uint_t, uint_t>> & intervalVector ) {
                for( const auto & interval : intervalVector ) {
                    outputInterval_.emplace_back(interval);
                }
            }

        private:

            // stores output interval of different output; second entry is the sampling interval; third is the starting time step
            std::vector<std::tuple<uint_t, uint_t, uint_t>> outputInterval_{};

            const BeforeFunction_T beforeFunction_{};
            walberla::timeloop::ITimeloop * const timeloop_{};

        }; // struct PdfSynchronisation

    } // namespace output

} // namespace turbine_core

#endif // TURBINECORE_TURBINEBEFOREFUNCTORS_H 
