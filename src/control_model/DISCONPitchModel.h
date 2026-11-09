
#pragma once

#ifndef TURBINECORE_DISCON_PITCH_MODEL_H
#define TURBINECORE_DISCON_PITCH_MODEL_H

#include "ControlModel.h"
#include "conversion/Conversion.h"

namespace turbine_core {

    namespace control_model {

        class DISCONPitchModel final: public ControlModel {

            friend class ControlModel;

        public:

            static constexpr Type TYPE = Type::DISCON_PITCH_CONTROL;

            HOST_DEVICE_PREFIX DISCONPitchModel() {}

            HOST_PREFIX DISCONPitchModel(
                    const real_t & pitchK,
                    const real_t & pitchControlKP,
                    const real_t & pitchControlKI,
                    const real_t & pitchMin,
                    const real_t & pitchMax,
                    const real_t & ratedGeneratorSpeed,
                    const real_t & nGear,
                    const real_t & maxPitchRate
                ) {
                    pitchK_ = pitchK;
                    pitchControlKP_ = pitchControlKP;
                    pitchControlKI_ = pitchControlKI;

                    pitchMin_ = pitchMin;
                    pitchMax_ = pitchMax;

                    ratedGeneratorSpeed_ = ratedGeneratorSpeed;
                    nGear_ = nGear;  // Gearbox ratio
                    maxPitchRate_ = maxPitchRate;
                    speedError_ = real_t(0);
                    intSpeedError_ = real_t(0);

                    const auto GK = real_t(1)/ (real_t(1) + previousPitch_/pitchK_);
                    intSpeedError_ = previousPitch_ / (GK * pitchControlKI_);
            }

            HOST_DEVICE_PREFIX DISCONPitchModel(
                    const real_t & previousPitch,
                    const real_t & pitchK,
                    const real_t & pitchControlKP,
                    const real_t & pitchControlKI,
                    const real_t & pitchMin,
                    const real_t & pitchMax,
                    const real_t & ratedGeneratorSpeed,
                    const real_t & nGear,
                    const real_t & maxPitchRate
                )
                    :
            previousPitch_(previousPitch),
            pitchK_(pitchK),
            pitchControlKP_(pitchControlKP),
            pitchControlKI_(pitchControlKI),
            pitchMin_(pitchMin),
            pitchMax_(pitchMax),
            maxPitchRate_(maxPitchRate),
            ratedGeneratorSpeed_(ratedGeneratorSpeed),
            nGear_(nGear),
            speedError_(real_t(0)),
            intSpeedError_(real_t(0))
            {
                auto GK = real_t(1)/ (real_t(1) + previousPitch_/pitchK_);
                intSpeedError_ = previousPitch_ / (GK * pitchControlKI_);
            }

            HOST_DEVICE_PREFIX DISCONPitchModel( const DISCONPitchModel & disconModel)
                : previousPitch_(disconModel.previousPitch_),
                  pitchK_(disconModel.pitchK_),
                  pitchControlKP_(disconModel.pitchControlKP_),
                  pitchControlKI_(disconModel.pitchControlKI_),
                  pitchMin_(disconModel.pitchMin_),
                  pitchMax_(disconModel.pitchMax_),
                  maxPitchRate_(disconModel.maxPitchRate_),
                  ratedGeneratorSpeed_(disconModel.ratedGeneratorSpeed_),
                  nGear_(disconModel.nGear_),
                  speedError_(disconModel.speedError_),
                  intSpeedError_(disconModel.intSpeedError_) {

                auto GK = real_t(1)/ (real_t(1) + previousPitch_/pitchK_);
                intSpeedError_ = previousPitch_ / (GK * pitchControlKI_);
            }

            HOST_DEVICE_PREFIX DISCONPitchModel & operator=( DISCONPitchModel && model ) = delete;

            HOST_DEVICE_PREFIX ~DISCONPitchModel() override = default;

            HOST_DEVICE_PREFIX auto & previousPitch() {
                return previousPitch_;
            }

            HOST_DEVICE_PREFIX auto & pitchK() {
                return pitchK_;
            }

            HOST_DEVICE_PREFIX auto & pitchControlKP() {
                return pitchControlKP_;
            }

            HOST_DEVICE_PREFIX auto & pitchControlKI() {
                return pitchControlKI_;
            }

            HOST_DEVICE_PREFIX auto & pitchMin() {
                return pitchMin_;
            }

            HOST_DEVICE_PREFIX auto & pitchMax() {
                return pitchMax_;
            }

            HOST_DEVICE_PREFIX auto & maxPitchRate() {
                return maxPitchRate_;
            }

            HOST_DEVICE_PREFIX auto & ratedGeneratorSpeed() {
                return ratedGeneratorSpeed_;
            }

            HOST_DEVICE_PREFIX auto & nGear() {
                return nGear_;
            }

            HOST_DEVICE_PREFIX auto & speedError() {
                return speedError_;
            }

            HOST_DEVICE_PREFIX auto & intSpeedError() {
                return intSpeedError_;
            }

        private:

            HOST_DEVICE_PREFIX DISCONPitchModel * do_clone() const override {
                return new DISCONPitchModel(*this);
            }

            HOST_DEVICE_PREFIX void do_print() const override {
                printf("DISCONPitchModel Control Model\n");
            }

            HOST_DEVICE_PREFIX void do_applyControl(Discretisation * const discretisation, const uint_t /*timeStep*/,
                                                    const real_t /*rotorMeanWind*/, const real_t /*rotorTorque*/, const real_t rotorOmega,
                                                    const Vector3<real_t> & /*rotorWindVaneWind*/) override {

                const auto rotorOmegaHSF = real_t(fabs(rotorOmega)) * nGear_; // High speed shaft
                const auto GK = real_t(1)/ (real_t(1) + previousPitch_/pitchK_);

                speedError_ = rotorOmegaHSF - ratedGeneratorSpeed_;
                intSpeedError_ += speedError_; // Unit time step in LBM units

                if(intSpeedError_ > pitchMax_ / (GK * pitchControlKI_)) {
                    intSpeedError_ = pitchMax_ / (GK * pitchControlKI_);
                } else if (intSpeedError_ < pitchMin_ / (GK * pitchControlKI_)) {
                    intSpeedError_ = pitchMin_ / (GK * pitchControlKI_);
                }

                auto pitchP = GK * pitchControlKP_ * speedError_;
                auto pitchI = GK * pitchControlKI_ * intSpeedError_;
                auto pitchCommanded = pitchP + pitchI;
                // pitchCommanded = std::max(std::min(pitchCommanded, pitchMax_), pitchMin_);
                if(pitchCommanded > pitchMax_) {
                    pitchCommanded = pitchMax_;
                } else if (pitchCommanded < pitchMin_) {
                    pitchCommanded = pitchMin_;
                }

                // Implement pitch rate limiter
                auto pitchRate = (pitchCommanded - previousPitch_); // Unit time step in LBM units
                if (fabs(pitchRate) > maxPitchRate_) {
                    pitchRate = pitchRate / real_t(fabs(pitchRate)) * maxPitchRate_;
                    pitchCommanded = previousPitch_ + pitchRate; // Unit time step in LBM units
                }

                Quaternion<real_t> quatX(Vector3<real_t>(1,0,0), real_t(0));
                Quaternion<real_t> quatY(Vector3<real_t>(0,1,0), real_t(0));
                Quaternion<real_t> quatZ(Vector3<real_t>(0,0,1), previousPitch_-pitchCommanded);
                auto product = quatX * quatY * quatZ;
                discretisation->updateRelativeOrientation(product);
                previousPitch_ = pitchCommanded;
            }

            HOST_DEVICE_PREFIX void do_checkNeeds(bool & needsMeanVelocity, bool & needsTorque, bool & needsWindVane) override {
                needsMeanVelocity = false;
                needsTorque = true;
                needsWindVane = false;
            }

            HOST_DEVICE_PREFIX walberla::mpi::MPISize do_getItemSize() override {
                return 11 * sizeof(real_t);
            };

            HOST_DEVICE_PREFIX void do_fillBuffer(BufferType_T * buffer) override {
                //
                // fill buffer
                memcpy(buffer, &previousPitch_, sizeof(real_t));
                buffer = buffer + sizeof(previousPitch_);
                memcpy(buffer, &pitchK_, sizeof(real_t));
                buffer = buffer + sizeof(pitchK_);
                memcpy(buffer, &pitchControlKP_, sizeof(real_t));
                buffer = buffer + sizeof(pitchControlKP_);
                memcpy(buffer, &pitchControlKI_, sizeof(real_t));
                buffer = buffer + sizeof(pitchControlKI_);
                memcpy(buffer, &pitchMin_, sizeof(real_t));
                buffer = buffer + sizeof(pitchMin_);
                memcpy(buffer, &pitchMax_, sizeof(real_t));
                buffer = buffer + sizeof(pitchMax_);
                memcpy(buffer, &ratedGeneratorSpeed_, sizeof(real_t));
                buffer = buffer + sizeof(ratedGeneratorSpeed_);
                memcpy(buffer, &nGear_, sizeof(real_t));
                buffer = buffer + sizeof(nGear_);
                memcpy(buffer, &maxPitchRate_, sizeof(real_t));
                buffer = buffer + sizeof(maxPitchRate_);
                memcpy(buffer, &speedError_, sizeof(real_t));
                buffer = buffer + sizeof(speedError_);
                memcpy(buffer, &intSpeedError_, sizeof(real_t));
                buffer = buffer + sizeof(intSpeedError_);
            };

            HOST_DEVICE_PREFIX void do_unfillBuffer(BufferType_T * buffer) override {
                //
                // unfill buffer
                memcpy(&previousPitch_, buffer, sizeof(real_t));
                buffer = buffer + sizeof(real_t);
                memcpy(&pitchK_, buffer, sizeof(real_t));
                buffer = buffer + sizeof(real_t);
                memcpy(&pitchControlKP_, buffer, sizeof(real_t));
                buffer = buffer + sizeof(real_t);
                memcpy(&pitchControlKI_, buffer, sizeof(real_t));
                buffer = buffer + sizeof(real_t);
                memcpy(&pitchMin_, buffer, sizeof(real_t));
                buffer = buffer + sizeof(real_t);
                memcpy(&pitchMax_, buffer, sizeof(real_t));
                buffer = buffer + sizeof(real_t);
                memcpy(&ratedGeneratorSpeed_, buffer, sizeof(real_t));
                buffer = buffer + sizeof(real_t);
                memcpy(&nGear_, buffer, sizeof(real_t));
                buffer = buffer + sizeof(real_t);
                memcpy(&maxPitchRate_, buffer, sizeof(real_t));
                buffer = buffer + sizeof(real_t);
                memcpy(&speedError_, buffer, sizeof(real_t));
                buffer = buffer + sizeof(real_t);
                memcpy(&intSpeedError_, buffer, sizeof(real_t));
                buffer = buffer + sizeof(real_t);
            }

            real_t previousPitch_{real_t(.0)};
            real_t pitchK_{};
            real_t pitchControlKP_{};
            real_t pitchControlKI_{};
            real_t pitchMin_{};
            real_t pitchMax_{};
            real_t maxPitchRate_{};

            real_t ratedGeneratorSpeed_{};
            real_t nGear_{};
            real_t speedError_{real_t(0)};
            real_t intSpeedError_{real_t(0)};
            bool pastFirstTimeStep_{false};
        };

    } // namespace control_model

} // namespace turbine_core

#endif //TURBINECORE_DISCON_PITCH_MODEL_H
