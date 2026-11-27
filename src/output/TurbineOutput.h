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
//! \file TurbineOutput.h
//! \author Helen SCHOTTENHAMML <helen.schottenhamml@fau.de>
//
//======================================================================================================================

#ifndef TURBINECORE_TURBINEOUTPUT_H
#define TURBINECORE_TURBINEOUTPUT_H

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include <blockforest/StructuredBlockForest.h>
#include <core/config/Config.h>
#include <field/FlagUID.h>
#include <field/vtk/FlagFieldCellFilter.h>
#include <field/vtk/VTKWriter.h>
#include <vtk/VTKOutput.h>
#include <vtk/ChainedFilter.h>

#include "wind_turbine_core/Enums.h"
#include "wind_turbine_core/WalberlaDataTypes.h"

#include "output/LineOutput.h"
#include "output/PlaneInclusionFilter.h"

#include "output/TurbineBeforeFunctors.h"

namespace turbine_core {

    namespace output {

        /**
         * Wrapper for all simulation output.
         * Reads specific configuration from 'Output' block in config file.
         */
        template< typename FlagField_T, typename ScalarField_T, typename VectorField_T,
                typename SecondOrderTensorField_T, typename ThirdOrderTensorField_T,
                typename PdfField_T, typename Stencil_T, bool zeroCentering, bool compressible >
        class TurbineOutput {

            using FieldMap_T = std::map<Fields::Types, BlockDataID>;
            using Vector_T = walberla::Vector3<real_t>;

            using VTKOutputPtr_T = std::shared_ptr<walberla::vtk::VTKOutput>;
            using LineOutput_T = LineOutput<ScalarField_T, VectorField_T, PdfField_T, VectorField_T, Stencil_T, zeroCentering, compressible>;
            using LineOutputPtr_T = std::shared_ptr<LineOutput_T>;

            using BeforeFunction_T = std::function< void () >;

        public:

            TurbineOutput( const std::shared_ptr<walberla::StructuredBlockForest> & forest,
                           walberla::timeloop::ITimeloop * const timeloop,
                           const walberla::Config::BlockHandle & config, const walberla::FlagUID & fluidFlagUID,
                           const FieldMap_T & fieldMap,
                           const bool binary = true, const bool continuousNumbering = false )
                    : forest_(forest), timeloop_(timeloop), binary_(binary), continuousNumbering_(continuousNumbering),
                      fluidFlagUID_(fluidFlagUID), fields_(fieldMap)
            {

                WALBERLA_ASSERT( !fields_.empty(), "Trying to set up output for empty field map." )

                const std::string baseFolder{"simulation_step"};

                /// setup VTK writer

                std::vector<walberla::Config::BlockHandle> vtkBlocks;
                config.getBlocks("VTK", vtkBlocks);
                for (const auto & block : vtkBlocks) {
                    initVTKOutput( block );
                }

                /// setup LINE OUTPUT writer
                if( config.getNumBlocks( "LineOutput" ) ) {
                    initLineOutput( config );
                }

                if(vtkBlocks.empty() && config.getNumBlocks( "LineOutput" )==0 ){
                    emptyOutput_=true;
                }

            }

            void addBeforeFunction( const BeforeFunction_T & f ) {
                beforeFunctions_.push_back(f);
            }

            void addMacroscopicFieldCalculator( const BeforeFunction_T & macroscopicFieldCalculator ) {

                GenericTurbineBeforeFunctor beforeFunctor( macroscopicFieldCalculator, timeloop_ );

                for( const auto vtkFieldInterval : fieldVTKOutputInterval_ ) {
                    beforeFunctor.addInterval(vtkFieldInterval);
                }

                if( lineOutput_ ) {
                    const auto freqs = lineOutput_->getOutputFrequencies();
                    for(auto freq : freqs) {
                        beforeFunctor.addInterval(freq);
                    }
                }

                if( lineOutput_ )
                    lineOutput_->addToBeforeFunctor( beforeFunctor );

                beforeFunctions_.emplace_back(beforeFunctor);

            }

            inline void operator()() { write(); }

            void write() {

                // no output block was specified
                if(emptyOutput_){
                    return;
                }

                // all specified output blocks have a output frequency of 0
                if (!writeVTKOutput_ && !writeLineOutput_) {
                    return;
                }

                const auto t = timeloop_->getCurrentTimeStep();
                // skipping this timestep
                if(t < skipTimestep_)
                    return;

                bool callBeforeFunctions = false;
                for (auto it = fieldVTKOutputInterval_.begin(); it != fieldVTKOutputInterval_.end(); ++it) {
                    if (t % *it == 0) callBeforeFunctions = true;
                }
                for (auto it = lineOutputOutputInterval_.begin(); it != lineOutputOutputInterval_.end(); ++it) {
                    if (t % *it == 0) callBeforeFunctions = true;
                }

                if (callBeforeFunctions) {
                    for( const auto & f : beforeFunctions_ ) {
                        f();
                    }
                }

                if( flagFieldVTKOutput_ ) {
                    flagFieldVTKOutput_->write();
                }

                if( domainDecompositionOutput_ ) {
                    domainDecompositionOutput_->write();
                }

                for( auto & fieldWriter : fieldVTKOutput_ ) {
                    fieldWriter->write();
                }

                if( lineOutput_ ) {
                    lineOutput_->write();
                }
            }

            void forceWrite( const uint_t counter ) {

                for( const auto & f : beforeFunctions_ ) {
                    f();
                }

                if( flagFieldVTKOutput_ ) {
                    flagFieldVTKOutput_->forceWrite(counter);
                }

                if( domainDecompositionOutput_ ) {
                    domainDecompositionOutput_->forceWrite(counter);
                }

                for( auto & fieldWriter : fieldVTKOutput_ ) {
                    fieldWriter->forceWrite(counter);
                }

                if( lineOutput_ ) {
                    lineOutput_->forceWrite(counter);
                }
            };


            auto skipTimestep() const {
                return skipTimestep_;
            }

            auto initialExecutionCount() const {
                return initialExecutionCount_;
            }

        private:

            void initVTKOutput( const walberla::Config::BlockHandle & config ) {

                const std::string baseFolder = config.getParameter<std::string>( "baseFolder", "vtk_out" );
                const std::string executionFolder = config.getParameter<std::string>( "executionFolder", "simulation_step" );

                const uint_t writeFrequency = config.getParameter<uint_t>( "writeFrequency" );
                const uint_t ghostLayers = config.getParameter<uint_t>( "ghostLayers", uint_t(0) );
                initialExecutionCount_ = config.getParameter<uint_t>( "initialExecutionCount", uint_t(0) );
                skipTimestep_ = config.getParameter<uint_t>( "skipTimestep", uint_t(0) );

                if (writeFrequency != 0)
                    writeVTKOutput_ = true;

                useMPIIO_ = config.getParameter<bool>("useMPIIO", false);
                oneFilePerProcess_ = config.getParameter<bool>("oneFilePerProcess", false);
                amrFileFormat_ = config.getParameter<bool>("amrFileFormat", false);
                forcePVTU_ = config.getParameter<bool>("forcePVTU", false);
                littleEndian_ = config.getParameter<bool>("littleEndian", true);

                /// setup FLAG FIELD writer
                if( config.isDefined( "writeFlagField" ) ) {

                    WALBERLA_ASSERT_NULLPTR(flagFieldVTKOutput_, "VTK Output for flag field can be specified in only ONE block.")

                    const uint_t flag_GhostLayers = config.getParameter<uint_t>("writeFlagField", ghostLayers);

                    std::string writerName{"flag_field_" + std::to_string(writeFrequency)};
                    flagFieldVTKOutput_ = walberla::vtk::createVTKOutput_BlockData( *forest_, writerName, writeFrequency, flag_GhostLayers, forcePVTU_,
                                                                                    baseFolder, executionFolder,
                                                                                    continuousNumbering_, binary_,
                                                                                    littleEndian_, useMPIIO_, initialExecutionCount_ + skipTimestep_, amrFileFormat_, oneFilePerProcess_);

                    auto flagWriter = std::make_shared<walberla::field::VTKWriter<FlagField_T>>(fields_.at(Fields::FLAG), "Flag Field");
                    flagFieldVTKOutput_->addCellDataWriter(flagWriter);

                }

                /// setup DOMAIN DECOMPOSITION writer
                if( config.isDefined( "writeDomainDecomposition" ) ) {

                    WALBERLA_ASSERT_NULLPTR(domainDecompositionOutput_, "VTK Output for domain decomposition can be specified in only ONE block.")

                    std::string writerName{"domain_decomposition_" + std::to_string(writeFrequency)};
                    domainDecompositionOutput_ = walberla::vtk::createVTKOutput_DomainDecomposition( forest_, writerName, writeFrequency,
                                                                                                     baseFolder, executionFolder,
                                                                                                     continuousNumbering_, binary_,
                                                                                                     littleEndian_, useMPIIO_, initialExecutionCount_ + skipTimestep_ );

                }

                /// setup FIELDS writer

                if( config.getNumBlocks( "fields" ) ) {

                    // inclusion filters
                    std::vector< walberla::vtk::VTKOutput::CellFilter > cellFilter;
                    parseInclusionFilters(cellFilter, config);

                    // FIELD writers
                    std::string writerName {"field_writer_" + std::to_string(writeFrequency) };
                    fieldVTKOutput_.emplace_back(walberla::vtk::createVTKOutput_BlockData(*forest_, writerName, writeFrequency, ghostLayers, forcePVTU_,
                                                                                          baseFolder, executionFolder,continuousNumbering_,binary_,
                                                                                          littleEndian_, useMPIIO_, initialExecutionCount_ + skipTimestep_, amrFileFormat_, oneFilePerProcess_));
                    auto & fieldWriter = fieldVTKOutput_.back();
                    fieldWriter->setInitialWriteCallsToSkip(skipTimestep_);

                    fieldVTKOutputInterval_.insert(writeFrequency);

                    for(const auto & cf : cellFilter) {
                        fieldWriter->addCellInclusionFilter(cf);
                    }

                    auto fieldBlock = config.getOneBlock("fields");

                    walberla::IBlock * block{nullptr};
                    for( auto blockIt = forest_->begin(); blockIt != forest_->end(); ++blockIt ) {
                        if( blockIt.get() != nullptr ) {
                            block = blockIt.get();
                            break;
                        }
                    }

                    for (auto fieldIt = fieldBlock.begin(); fieldIt != fieldBlock.end(); ++fieldIt) {

                        const auto fieldType = Fields::toType(fieldIt->first);

                        BlockDataID fieldID{};

                        if( fields_.count(fieldType) != 0 ) {
                            fieldID = fields_.at(fieldType);
                        } else {
                            WALBERLA_ABORT("Field " << fieldIt->first << " not found in field map.")
                        }

                        if (block->isDataOfType<ScalarField_T>(fieldID)) {
                            auto writer = std::make_shared<walberla::field::VTKWriter<ScalarField_T>>(fieldID, fieldIt->first);
                            fieldWriter->addCellDataWriter(writer);
                        } else if (block->isDataOfType<VectorField_T>(fieldID)) {
                            auto writer = std::make_shared<walberla::field::VTKWriter<VectorField_T>>(fieldID, fieldIt->first);
                            fieldWriter->addCellDataWriter(writer);
                        } else if (block->isDataOfType<SecondOrderTensorField_T>(fieldID)) {
                            auto writer = std::make_shared<walberla::field::VTKWriter<SecondOrderTensorField_T>>(fieldID, fieldIt->first);
                            fieldWriter->addCellDataWriter(writer);
                        } else if (block->isDataOfType<ThirdOrderTensorField_T>(fieldID)) {
                            auto writer = std::make_shared<walberla::field::VTKWriter<ThirdOrderTensorField_T>>(fieldID, fieldIt->first);
                            fieldWriter->addCellDataWriter(writer);
                        } else if (block->isDataOfType<PdfField_T>(fieldID)) {
                            auto writer = std::make_shared<walberla::field::VTKWriter<PdfField_T>>(fieldID,
                                    fieldIt->first);
                            fieldWriter->addCellDataWriter(writer);
                        } else {
                            WALBERLA_ABORT("Unknown field type for field " << fieldIt->first << ".")
                        }

                    }

                }

            }

            void initLineOutput( const walberla::Config::BlockHandle & config ) {

                walberla::Config::Blocks lineOutputBlocks{};
                config.getBlocks("LineOutput", lineOutputBlocks);
                if( !lineOutputBlocks.empty() )  {
                    lineOutput_ = std::make_shared<LineOutput_T>( config, forest_, timeloop_, fields_ );
                }
                for (auto & block : lineOutputBlocks) {
                    const uint_t frequency = block.getParameter<uint_t>("outputFrequency");
                    if (frequency != 0)
                        writeLineOutput_ = true;
                    lineOutputOutputInterval_.insert( frequency );
                }

            }

            void parseInclusionFilters( std::vector<walberla::vtk::VTKOutput::CellFilter> & cellFilter, const walberla::Config::BlockHandle & vtkConfig ) {

                if( vtkConfig.getNumBlocks( "inclusion_filters" ) ) {

                    auto inclusionBlock = vtkConfig.getOneBlock( "inclusion_filters" );

                    if( inclusionBlock.getNumBlocks("combine") ) {

                        std::vector<walberla::vtk::ChainedFilter> chainedFilter{};
                        parseCombineFilter(chainedFilter, inclusionBlock);
                        for( const auto & combine : chainedFilter ) {
                            cellFilter.emplace_back(combine);
                        }

                    }

                    // DOMAIN filter
                    if( inclusionBlock.isDefined( "DomainFilter" ) ) {
                        walberla::field::FlagFieldCellFilter <FlagField_T> fluidFilter(fields_.at(Fields::FLAG));
                        fluidFilter.addFlag(fluidFlagUID_);
                        cellFilter.push_back(fluidFilter);
                    }

                    // AABB filter
                    if( inclusionBlock.getNumBlocks("AABB") ) {

                        std::vector<walberla::vtk::AABBCellFilter> aabbFilter{};
                        parseAABBInclusionFilter(aabbFilter, inclusionBlock);
                        for( const auto & aabb : aabbFilter ) {
                            cellFilter.emplace_back(aabb);
                        }

                    }

                    // PLANE filter
                    if( inclusionBlock.getNumBlocks("Plane") ) {

                        std::vector<PlaneInclusionFilter> planeFilter{};
                        parsePlaneInclusionFilter(planeFilter, inclusionBlock);
                        for( const auto & plane : planeFilter ) {
                            cellFilter.emplace_back(plane);
                        }

                    }

                }

            }

            void parseCombineFilter( std::vector<walberla::vtk::ChainedFilter> & chainedFilter, const walberla::Config::BlockHandle & inclusionBlock ) {

                std::vector<walberla::Config::BlockHandle> combineFilterBlocks;
                inclusionBlock.getBlocks("combine", combineFilterBlocks);

                for( const auto & combineFilter : combineFilterBlocks ) {

                    walberla::vtk::ChainedFilter chained{};

                    if( combineFilter.isDefined("DomainFilter") ) {
                        walberla::field::FlagFieldCellFilter <FlagField_T> fluidFilter(fields_.at(Fields::FLAG));
                        fluidFilter.addFlag(fluidFlagUID_);
                        chained.addFilter(fluidFilter);
                    }

                    if( combineFilter.getNumBlocks("AABB") ) {

                        std::vector<walberla::vtk::AABBCellFilter> aabbFilter{};
                        parseAABBInclusionFilter(aabbFilter, combineFilter);
                        for( const auto & aabb : aabbFilter ) {
                            chained.addFilter(aabb);
                        }

                    }

                    if( combineFilter.getNumBlocks("Plane") ) {

                        std::vector<PlaneInclusionFilter> planeFilter{};
                        parsePlaneInclusionFilter(planeFilter, combineFilter);
                        for( const auto & plane : planeFilter ) {
                            chained.addFilter(plane);
                        }

                    }

                    chainedFilter.emplace_back(chained);

                }

            }

            void parseAABBInclusionFilter( std::vector<walberla::vtk::AABBCellFilter> & aabbFilter, const walberla::Config::BlockHandle & inclusionBlock ) {

                std::vector<walberla::Config::BlockHandle> AABBFilterBlocks;
                inclusionBlock.getBlocks("AABB", AABBFilterBlocks);

                for (const auto & filter : AABBFilterBlocks) {

                    const Vector_T min = filter.getParameter<Vector_T>("min");
                    const Vector_T max = filter.getParameter<Vector_T>("max");

                    aabbFilter.emplace_back(walberla::vtk::AABBCellFilter({min,max}));

                }

            }

            void parsePlaneInclusionFilter ( std::vector<PlaneInclusionFilter> & planeFilter, const walberla::Config::BlockHandle & inclusionBlock ) {

                std::vector<walberla::Config::BlockHandle> planeFilterBlocks;
                inclusionBlock.getBlocks("Plane", planeFilterBlocks);

                for( const auto & filter : planeFilterBlocks ) {

                    const Vector_T point  = filter.getParameter<Vector_T>("point");
                    const Vector_T normal = filter.getParameter<Vector_T>("normal");

                    const real_t maxDistance = filter.getParameter<real_t>("maxDistance", real_t(0.5));

                    planeFilter.emplace_back(point, normal, maxDistance);

                }

            }

            const std::shared_ptr<walberla::StructuredBlockForest> forest_;
            walberla::timeloop::ITimeloop * const timeloop_;

            uint_t initialExecutionCount_{0};
            uint_t skipTimestep_{0};

            bool binary_{true};
            bool continuousNumbering_{false};
            bool useMPIIO_{false};
            bool oneFilePerProcess_{false};
            bool amrFileFormat_{false};
            bool forcePVTU_{false};
            bool littleEndian_{true};

            bool emptyOutput_{false};

            walberla::FlagUID fluidFlagUID_{};

            // fields
            FieldMap_T fields_;

            // possible VTK output
            VTKOutputPtr_T flagFieldVTKOutput_{};
            VTKOutputPtr_T domainDecompositionOutput_{};

            bool writeVTKOutput_{false};
            std::vector<VTKOutputPtr_T> fieldVTKOutput_{};
            std::set<uint_t> fieldVTKOutputInterval_{};

            // line output
            bool writeLineOutput_{false};
            LineOutputPtr_T lineOutput_{};
            std::set<uint_t> lineOutputOutputInterval_{};

            std::vector<BeforeFunction_T> beforeFunctions_{};

        }; // class TurbineOutput

    }

} // turbine_core

#endif // TURBINECORE_TURBINEOUTPUT_H
