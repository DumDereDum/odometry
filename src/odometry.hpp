#ifndef ODOMETRY_HPP
#define ODOMETRY_HPP

#include <opencv2/core/ocl.hpp>

#include "odometry_func/odometry_impl.hpp"
#include "odometry_func/odometry_icp.hpp"
#include "odometry_func/odometry_rgb.hpp"
#include "odometry_func/odometry_rgbd.hpp"

using namespace cv;

enum class OdometryType
{
    ICP  = 0,
    RGB  = 1,
    RGBD = 2 
};

class Odometry
{
private:
    OdometryImpl *odometry;
public:
    Odometry(OdometryType otype);
    ~Odometry();

    bool compute();
};

#endif //ODOMETRY_HPP