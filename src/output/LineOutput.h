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
//! \file LineOutput.h
//! \author Helen SCHOTTENHAMML <helen.schottenhamml@fau.de>
//
//======================================================================================================================

#ifndef TURBINECORE_LINEOUTPUT_H
#define TURBINECORE_LINEOUTPUT_H

#pragma once

#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

#include <stencil/D3Q27.h>

#include "output/TurbineBeforeFunctors.h"

#include "LineOutputLine.h"
#include "LineOutputMacroscopicLine.h"

namespace turbine_core {

    namespace output {

        template< typename ScalarField_T, typename VectorField_T,
                  typename PdfField_T = field::Field<real_t>, typename ForceField_T = field::Field<real_t>,
                  typename Stencil_T = walberla::stencil::D3Q27, bool zeroCentering = false, bool compressible = false >
        class LineOutput {

            using FieldMap = LineOutputLineBase::FieldMap;
            using BeforeFunction_T = std::function< void () >;

        public:

            LineOutput( const walberla::Config::BlockHandle & config,
                        walberla::StructuredBlockForest * const forest,
                        walberla::timeloop::ITimeloop * const timeloop,
                        const FieldMap & fieldMap )
            : forest_(forest), timeloop_(timeloop)
            {
                WALBERLA_ROOT_SECTION() {
                    // clear directory
                    std::filesystem::path path(basefolder_);
                    if( std::filesystem::exists(path) ) {
                        std::filesystem::remove_all(path);
                    }

                    if( !std::filesystem::exists(path) ) {
                        std::filesystem::create_directories(path);
                    }
                }

                if( config.getNumBlocks("LineOutput") ) {

                    std::vector<walberla::Config::BlockHandle> blocks;
                    config.getBlocks("LineOutput", blocks);

                    lineOutput_.reserve(blocks.size());

                    bool usePdfs{false};
                    if( fieldMap.count(Fields::DENSITY) == 0 && fieldMap.count(Fields::VELOCITY) == 0 )
                        usePdfs = true;

                    for( auto & handle : blocks ) {

                        auto fields = handle.getOneBlock("Fields");
                        if( usePdfs && (fields.isDefined("density") || fields.isDefined("velocity") ||
                                        fields.isDefined("Density") || fields.isDefined("Velocity")) ) {
                            lineOutput_.emplace_back(std::make_unique<LineOutputMacroscopicLine<PdfField_T,ForceField_T, Stencil_T, zeroCentering, compressible>>(handle, forest_, fieldMap));
                        }

                        lineOutput_.emplace_back(std::make_unique<LineOutputLine<ScalarField_T,VectorField_T>>(handle, forest_, fieldMap));
                    }
                }

                for( uint_t i = 0; i < lineOutput_.size(); ++i ) {
                    lineOutput_[i]->writeLineHeaders(forest_, filename(i));
                }

            }

            LineOutput( const walberla::Config::BlockHandle & config,
                        const std::shared_ptr<walberla::StructuredBlockForest> & forest,
                        walberla::timeloop::ITimeloop * const timeloop,
                        const FieldMap & fieldMap )
            : LineOutput(config, forest.get(), timeloop, fieldMap)
            {}

            inline void operator()() { write(); }

            void write() {
                const auto timestep = timeloop_->getCurrentTimeStep();
                for( uint_t i = 0; i < lineOutput_.size(); ++i ) {
                    auto & lineOut = lineOutput_[i];
                    if(timestep >= lineOut->startingTimeStep() && (timestep % lineOut->outputFrequency()) == 0)
                        lineOut->writeLine(forest_, filename(i), timestep);
                }
            }

            void forceWrite() {
                for( uint_t i = 0; i < lineOutput_.size(); ++i ) {
                    lineOutput_[i]->writeLine(forest_, filename(i), timeloop_->getCurrentTimeStep());
                }
            }

            void addToBeforeFunctor( GenericTurbineBeforeFunctor & beforeFunctor ) {
                for( uint_t i = 0; i < lineOutput_.size(); ++i ) {
                    beforeFunctor.addInterval(lineOutput_[i]->outputFrequency());
                }
            }

            std::vector<std::tuple<uint_t, uint_t, uint_t>> getOutputFrequencies() {
                std::vector<std::tuple<uint_t, uint_t, uint_t>> freqs(lineOutput_.size());
                for( uint_t i = 0; i < lineOutput_.size(); ++i ) {
                    freqs[i] = {lineOutput_[i]->outputFrequency(), 0, lineOutput_[i]->startingTimeStep()};
                }
                return freqs;
            }

        private:

            std::string filename( const uint_t lineOutputNumber ) {
                std::ostringstream filename;
                filename << "LineOutput" << std::setfill('0') << std::setw(2) << lineOutputNumber
                         << "_Line";

                return (basefolder_ / filename.str()).c_str();
            }

            std::vector<std::unique_ptr<LineOutputLineBase>> lineOutput_;

            walberla::StructuredBlockForest * const forest_;
            walberla::timeloop::ITimeloop * const timeloop_;

            const std::filesystem::path basefolder_{"LineOutput"};

        };

    }

} // turbine_core 

#endif //TURBINECORE_LINEOUTPUT_H