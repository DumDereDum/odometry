#ifndef ODOMETRY_HPP
#define ODOMETRY_HPP

#include <opencv2/core/ocl.hpp>
#include <opencv2/core/affine.hpp>

#include "../src/odometry_frame.hpp"
#include "../src/odometry_settings.hpp"

using namespace cv;

enum class OdometryType
{
    ICP  = 0,
    RGB  = 1,
    RGBD = 2 
};

class OdometryImpl
{
private:
    virtual bool compute_corresps() const = 0;
    virtual bool compute_Rt() const = 0;
public:
    OdometryImpl() {};
    ~OdometryImpl() {};

    virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const = 0;
};

class Odometry
{
private:
    Ptr<OdometryImpl>odometry;
public:
    Odometry(OdometryType otype, OdometrySettings settings);
    ~Odometry();

    bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt);
};


#endif //ODOMETRY_HPP