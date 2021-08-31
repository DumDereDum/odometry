#ifndef ODOMETRY_HPP
#define ODOMETRY_HPP

#include <opencv2/core/ocl.hpp>
#include <opencv2/core/affine.hpp>

#include "odometry_func/odometry_impl.hpp"
#include "odometry_func/odometry_icp.hpp"
#include "odometry_func/odometry_rgb.hpp"
#include "odometry_func/odometry_rgbd.hpp"
#include "odometry_func/odometry_frame.hpp"
#include "odometry_func/odometry_settings.hpp"

using namespace cv;
/*
enum class OdometryType
{
    ICP  = 0,
    RGB  = 1,
    RGBD = 2 
};
*/
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