#pragma once
#include <memory>
#include <sophus/se3.hpp>

class SensorBase {
public:
    SensorBase() {}
    SensorBase(const Sophus::SE3d& Tbs_) : Tbs(Tbs_) {}
    virtual ~SensorBase() {}

    Sophus::SE3d Tbs;
};

using SensorBasePtr = std::shared_ptr<SensorBase>;
using SensorBaseConstPtr = std::shared_ptr<const SensorBase>;

class ImuSensor : public SensorBase
{
public:
    ImuSensor() {}
    ImuSensor(double gn, double an, double gbn, double abn)
        :gyro_noise(gn), acc_noise(an), gyro_bias_noise(gbn), acc_bias_noise(abn) {}
    virtual ~ImuSensor() {}

    // Process noise
    double gyro_noise;
    double acc_noise;
    double gyro_bias_noise;
    double acc_bias_noise;
};

using ImuSensorPtr = std::shared_ptr<ImuSensor>;
using ImuSensorConstPtr = std::shared_ptr<const ImuSensor>;
