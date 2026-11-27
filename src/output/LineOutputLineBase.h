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
//! \file LineOutputLineBase.h
//! \author Helen SCHOTTENHAMML <helen.schottenhamml@fau.de>
//
//======================================================================================================================

#ifndef TURBINECORE_LINEOUTPUTLINEBASE_H
#define TURBINECORE_LINEOUTPUTLINEBASE_H

#pragma once

#ifdef __CUDACC__
#include <gpu/GPUField.h>
#endif

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <core/config/Config.h>
#include <core/mpi/MPIWrapper.h>
#include <core/mpi/Reduce.h>

#include "walberla_helper/field/interpolators/TrilinearFieldInterpolator.h"
#include "wind_turbine_core/Enums.h"

namespace turbine_core {

    namespace output {

        namespace internal {

            constexpr uint_t positionWidth{15};
            constexpr uint_t fieldWidth{20};

            constexpr char comment_char[]{"///"};

            inline void writePositionItem(std::ostringstream & oss, const real_t item) {
                const int precision = int(oss.precision());
                oss << std::right << std::setw(positionWidth) << std::setfill(' ') << std::setprecision(4)
                    << std::fixed << item;
                oss << std::setprecision(precision);
            }

            inline std::string writeFieldItem(const real_t item) {
                std::ostringstream oss;
                oss << std::right << std::setw(fieldWidth) << std::setfill(' ') << std::scientific << item;
                return oss.str();
            }

            inline std::string writeFieldIdentifier(const std::string & name, const std::string & post = "") {
                std::ostringstream oss;
                oss << std::right << std::setw(fieldWidth) << std::setfill(' ') << (name + post);
                return oss.str();
            }

        } // internal


        class LineOutputLineBase {

        public:

            using FieldMap = std::map<Fields::Types, BlockDataID>;
            using Line = std::vector<Vector3<real_t>>;
            using BlockLocalLine = std::map<walberla::IBlock*, Line>;

            HOST_PREFIX virtual ~LineOutputLineBase() {}

            HOST_PREFIX LineOutputLineBase( const walberla::Config::BlockHandle & config  )
                    : outputFrequency_(config.getParameter<uint_t>("outputFrequency", uint_t(1))),
                      startingTimeStep_(config.getParameter<uint_t>("startingTimeStep", uint_t(0)))

            {
                std::vector<walberla::Config::BlockHandle> lines;
                config.getBlocks("Line", lines, 1);

                for (auto & line : lines) {

                    Vector3<real_t> start = line.getParameter<Vector3<real_t>>("start");
                    Vector3<real_t> end = line.getParameter<Vector3<real_t>>("end");

                    WALBERLA_CHECK_FLOAT_UNEQUAL((end-start).length(), real_t(0.0), "Line length must be greater than 0!")

                    uint_t resolution = line.getParameter<uint_t>("resolution", uint_t((end - start).length()));

                    addLine(start, end, resolution);

                }

            }

            HOST_PREFIX void writeLine( walberla::StructuredBlockForest * const forest,
                                        const std::string & basename,
                                        const uint_t timestep ) {

                if(!hasValidField_)
                    return;

                for( uint_t i = 0; i < lines_.size(); ++i ) {

//                    writeLineHeader(i, forest, filename);
                    const auto lineStr = std::to_string(i);
                    const auto lineNum = std::string(2 - std::min(2ul, lineStr.length()), '0') + lineStr;
                    const auto filename = basename + lineNum + ".txt";

                    WALBERLA_MPI_BARRIER()

                    auto start = lines_[i].front();
                    auto end = lines_[i].back();
                    const real_t len = (end - start).length();

                    // sort all points per containing block
                    std::vector<std::pair<walberla::IBlock*, std::vector<Vector3<real_t>>>> blockLine{lines_[i].size()};

                    auto * prevBlock = forest->getBlock(start[0], start[1], start[2]);
                    blockLine[0] = std::make_pair(prevBlock, std::vector<Vector3<real_t>>{});
                    uint_t blockLocalCounter{0};
                    for (auto & point : lines_[i]) {
                        auto * block = forest->getBlock(point[0], point[1], point[2]);
                        if(block != prevBlock) {
                            ++blockLocalCounter;
                            blockLine[blockLocalCounter] = std::make_pair(block, std::vector<Vector3<real_t>>{});
                            prevBlock = block;
                        }
                        blockLine[blockLocalCounter].second.push_back(point);
                    }

                    ++blockLocalCounter;
                    blockLine.resize(blockLocalCounter);

                    const auto nBlockLines = blockLocalCounter;
                    uint_t maxNBlockLines{nBlockLines};

                    WALBERLA_MPI_SECTION() {
                        walberla::mpi::allReduceInplace(maxNBlockLines, walberla::mpi::MAX);
                    }

                    for( uint_t l = 0; l < maxNBlockLines; ++l ) {

                        // traverse block local data one by one
                        WALBERLA_MPI_BARRIER()

                        if(l >= nBlockLines) {
                            continue;
                        }

                        auto & blockLocalLine = blockLine[l];
                        auto * block = blockLocalLine.first;

                        if( block == nullptr )
                            continue;

                        auto & points = blockLocalLine.second;

                        std::vector<std::vector<real_t> > results{};
                        interpolateBlockLocalLine(block, forest, points, results);

                        // write data per point
                        std::ostringstream oss;

                        for (uint_t p = 0; p < points.size(); ++p) {

                            oss << std::right << std::setw(7) << std::setfill(' ') << timestep;
                            real_t pos = (start - points[p]).length();

                            // ensure alignment with field specifier due to comment
                            oss << std::string(sizeof(internal::comment_char) / sizeof(internal::comment_char[0]),
                                               ' ');

                            internal::writePositionItem(oss, pos);
                            internal::writePositionItem(oss, pos / len);

                            writeFields(oss, p, results);

                        } // loop over all points

                        // open file in append mode
                        std::ofstream file;
                        file.open(filename, std::ios::out | std::ios::app);
                        if (file.is_open()) {
                            file << oss.str();
                            file.close();
                        } else {
                            WALBERLA_LOG_WARNING_ON_ROOT("Could not write into file " << filename)
                        }

                    } // loop over block-local lines

                } // loop over lines

            }

            HOST_PREFIX void writeLineHeaders( walberla::StructuredBlockForest * const forest,
                                               const std::string & basename ) const {

                for( uint_t i = 0; i < lines_.size(); ++i ) {

                    const auto lineStr = std::to_string(i);
                    const auto lineNum = std::string(2 - std::min(2ul, lineStr.length()), '0') + lineStr;
                    const auto filename = basename + lineNum + ".txt";

                    writeLineHeader(i, forest, filename);

                }

            }

            HOST_PREFIX auto outputFrequency() const { return outputFrequency_; }
            HOST_PREFIX auto startingTimeStep() const { return startingTimeStep_; }

        protected:

            std::vector<Line> lines_;
            bool hasValidField_{false};

        private:

            HOST_PREFIX void addLine( const Vector3<real_t> & start, const Vector3<real_t> & end, const uint_t nPoints ) {

                std::vector<Vector3<real_t>> points;
                points.reserve(nPoints);

                points.push_back(start);

                const Vector3<real_t> dx = (end - start) / real_t(nPoints - 1);

                for (uint_t i = 1; i < nPoints; ++i) {
                    points.push_back(start + dx * real_t(i));
                }

                lines_.emplace_back(points);
            }

            HOST_PREFIX void writeLineHeader( const uint_t lineNumber, walberla::StructuredBlockForest * const forest,
                                              const std::string & filename ) const {

                if(!hasValidField_)
                    return;

                WALBERLA_ROOT_SECTION() {

                    std::ostringstream oss;

                    auto & line = lines_[lineNumber];

                    auto & start = line.front();
                    auto & end = line.back();

                    oss << "\n\n" << internal::comment_char << " LINE " << ": ";
                    oss << std::defaultfloat << "START " << start << "; END " << end << "; POINTS "
                        << lines_[lineNumber].size() << " " << std::setfill('/') << "\n\n";

                    oss << internal::comment_char << std::right << std::setw(internal::positionWidth) << std::setfill(' ')
                        << "writing step";
                    oss << std::right << std::setw(internal::positionWidth) << std::setfill(' ') << "abs. Position";
                    oss << std::right << std::setw(internal::positionWidth) << std::setfill(' ') << "rel. Position";

                    writeFieldSpecifier( oss, forest );

                    oss << "\n";

                    std::ofstream file;
                    file.open(filename, std::ios::out | std::ios::app);
                    if (file.is_open()) {
                        file << oss.str();
                        file.close();
                    } else {
                        WALBERLA_ABORT("In LineOutputBase : Could not open file " << filename << "...!")
                    }
                }
            }

            // private abstract functions

            HOST_PREFIX virtual void parseFields( walberla::StructuredBlockForest * const forest, const walberla::Config::BlockHandle & fields, const FieldMap & fieldMap ) = 0;
            HOST_PREFIX virtual void interpolateBlockLocalLine( walberla::IBlock * block, walberla::StructuredBlockForest * const forest,
                                                                const Line & points, std::vector< std::vector<real_t> > & results ) = 0;
            HOST_PREFIX virtual void writeFields( std::ostringstream & oss, const uint_t pointIndex, std::vector<std::vector<real_t>> & results ) = 0;
            HOST_PREFIX virtual void writeFieldSpecifier( std::ostringstream & oss, walberla::StructuredBlockForest * const forest ) const = 0;

            const uint_t outputFrequency_{};
            const uint_t startingTimeStep_{};

        }; // class LineOutputLineBase

    } // namespace output

} // namespace turbine_core

#endif //TURBINECORE_LINEOUTPUTLINEBASE_H