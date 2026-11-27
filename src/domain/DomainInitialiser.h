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
//! \file DomainInitialiser.h
//! \author Helen SCHOTTENHAMML <helen.schottenhamml@fau.de>
//
//======================================================================================================================

#ifndef TURBINECORE_DOMAININITIALISER_H
#define TURBINECORE_DOMAININITIALISER_H

#pragma once

#include <blockforest/StructuredBlockForest.h>
#include <core/config/Config.h>
#include <field/iterators/IteratorMacros.h>

#include "EnvironmentSetup.h"
#include "wind_turbine_core/ProjectDefines.h"
#include "wind_turbine_core/WalberlaDataTypes.h"
#include "wind_turbine_core/math/Vector3.h"

namespace turbine_core {

    namespace domain {

        class DomainInitialisation {

        public:

            enum Type {
                UNIFORM,
                ASMUTH,
                LOG_LAW
            };

            virtual ~DomainInitialisation() {}

            static Type toType(const std::string & identifier) {

                auto id = string::toLowercase(identifier);
                id = string::removeSpaces(id);

                if (id == "uniform") {
                    return Type::UNIFORM;
                } else if (id == "asmuth") {
                    return Type::ASMUTH;
                } else if (id == "loglaw") {
                    return Type::LOG_LAW;
                } else {
                    WALBERLA_ABORT("Invalid initialisation setup name (" << identifier << ").")
                }

            }

            static std::string toString(const DomainInitialisation::Type & type) {

                switch (type) {
                    case Type::UNIFORM :
                        return "Uniform";
                    case Type::ASMUTH :
                        return "Asmuth";
                    case Type::LOG_LAW :
                        return "LogLaw";
                }

            }

            virtual Type getType() const = 0;

            DomainInitialisation(const std::shared_ptr<walberla::blockforest::StructuredBlockForest>& blocks)
            : blocks_(blocks)
            {}

            //TODO rework -> a lof of copy paste !

            // use with functor and velocity field
            template<typename VelocityField_T, typename SetterFunctor_T>
            void setViaVelocityField(walberla::BlockDataID velocityFieldId,
                                      SetterFunctor_T & setter){

                for( auto blockIt = blocks_->begin(); blockIt != blocks_->end(); ++blockIt ) {

                    auto velField = blockIt->getData<VelocityField_T>( velocityFieldId );
                    WALBERLA_ASSERT_NOT_NULLPTR(velField)

                    for ( auto cellIt = velField->beginWithGhostLayer(); cellIt != velField->end(); ++cellIt ) {

                        walberla::Cell globalCell{};
                        const walberla::Cell localCell = cellIt.cell();
                        blocks_->transformBlockLocalToGlobalCell(globalCell, *blockIt, localCell);
                        walberla::Vector3<real_t> cellCenter;
                        blocks_->getCellCenter(cellCenter, globalCell, blocks_->getLevel(*blockIt));

                        auto initVel = get(cellCenter);

                        velField->get(localCell, 0) = initVel[0];
                        velField->get(localCell, 1) = initVel[1];
                        velField->get(localCell, 2) = initVel[2];
                    }

                    setter(blockIt.get());

                }

            }

            // manipulate pdf field directly
            template<typename PdfField_T>
            void setPDFField(const BlockDataID & pdfFieldId) {

                for( auto blockIt = blocks_->begin(); blockIt != blocks_->end(); ++blockIt ) {

                    auto pdfField = blockIt->getData<PdfField_T>( pdfFieldId );
                    WALBERLA_ASSERT_NOT_NULLPTR(pdfField)

                    for ( auto cellIt = pdfField->beginWithGhostLayer(); cellIt != pdfField->end(); ++cellIt ) {

                        walberla::Cell globalCell{};
                        const walberla::Cell localCell = cellIt.cell();
                        blocks_->transformBlockLocalToGlobalCell(globalCell, *blockIt, localCell);
                        walberla::Vector3<real_t> cellCenter;
                        blocks_->getCellCenter(cellCenter, globalCell);

                        auto initVel = get(cellCenter);

                        pdfField->setDensityAndVelocity(localCell, initVel, real_t(1.0));
                    }

                }
            }

        private:

            virtual walberla::Vector3<real_t> get(real_t x, real_t y, real_t z) const = 0;

            walberla::Vector3<real_t> get(const walberla::Vector3<real_t> &position) const {
                return get(position[0], position[1], position[2]);
            }

            std::shared_ptr<walberla::blockforest::StructuredBlockForest> blocks_;

        };

        class UniformInitialisation : public DomainInitialisation {

        public:
            static constexpr Type TYPE = Type::UNIFORM;

            explicit UniformInitialisation(const std::shared_ptr<walberla::blockforest::StructuredBlockForest>& blocks, const real_t initialVelocity)
            : DomainInitialisation(blocks), initialVelocity_(initialVelocity)
            {}

            explicit UniformInitialisation(const std::shared_ptr<walberla::blockforest::StructuredBlockForest>& blocks, const Vector3<real_t> & initialVelocity)
            : DomainInitialisation(blocks), initialVelocity_(initialVelocity.data())
            {}

            [[nodiscard]] Type getType() const override {
                return TYPE;
            }

            walberla::Vector3<real_t> get(const real_t, const real_t, const real_t) const override {
                return initialVelocity_;
            }

        private:
            const walberla::Vector3<real_t> initialVelocity_;
        };

        class AsmuthInitialisation : public DomainInitialisation {

        public:

            explicit AsmuthInitialisation(const std::shared_ptr<walberla::blockforest::StructuredBlockForest>& blocks,
                                          const real_t flowDirection, const real_t frictionVelocity, const real_t roughnessLength, const real_t kappa,
                                          const real_t B, const real_t viscosity, const Vector3<uint_t>& domainSize,
                                          const bool usePerturbations)
                    : DomainInitialisation(blocks),
                      flowDirection_(flowDirection), frictionVelocity_(frictionVelocity), roughnessLength_(roughnessLength), kappa_(kappa), B_(B), viscosity_(viscosity),
                      usePerturbations_(usePerturbations), domainSize_(real_t(domainSize[0]), real_t(domainSize[1]), real_t(domainSize[2]))
                    {}

            Type getType() const override {
                if(usePerturbations_)
                    return Type::ASMUTH;
                else
                    return Type::LOG_LAW;
            }

            walberla::Vector3<real_t> get(const real_t x, const real_t y, const real_t z) const override {

                const uint_t flowAxis = 0;
                const uint_t wallAxis = 2;
                const uint_t remAxis = 1;

                auto const & delta = domainSize_[wallAxis];

                const auto rel_x = x / domainSize_[flowAxis];
                const auto rel_y = y / domainSize_[remAxis];

                const real_t pos = std::max(z, real_t(0.05));
                const auto rel_z = pos / delta;

                auto initialVel = frictionVelocity_ / kappa_ * (std::log( pos / roughnessLength_ ));

                walberla::Vector3<real_t> vel;
                vel[flowAxis] = initialVel * std::cos(flowDirection_);

                if(usePerturbations_) {
                    vel[remAxis] = initialVel * std::sin(flowDirection_) + real_t(2.0) * frictionVelocity_ / kappa_ * std::sin(walberla::math::pi * real_t(16.0) * rel_x) * std::sin(walberla::math::pi * real_t(8.0) * rel_z) /
                                   (std::pow(rel_z, real_t(2.0)) + real_t(1.0));

                    vel[wallAxis] = real_t(8.0) * frictionVelocity_ / kappa_ *
                                    (std::sin(walberla::math::pi * real_t(8.0) * rel_y) * std::sin(walberla::math::pi * real_t(8.0) * rel_z) + std::sin(walberla::math::pi * real_t(8.0) * rel_x)) /
                                    (std::pow(real_t(0.5) * delta - pos, real_t(2.0)) + real_t(1.0));
                } else {
                    vel[remAxis] = real_t(0.0);

                    vel[wallAxis] = real_t(0.0);
                }

                return vel;
            }

        private:
            const real_t flowDirection_;
            const real_t frictionVelocity_;
            const real_t roughnessLength_;
            const real_t kappa_;
            const real_t B_;
            const real_t viscosity_;

            const bool usePerturbations_;

            const Vector3<real_t> domainSize_;
        };

    } // domain

} // turbine_core


#endif //TURBINECORE_DOMAININITIALISER_H
