#include <iostream>
#include "odometry.hpp"


int main()
{
    Odometry od_icp  = Odometry(OdometryType::ICP);
    Odometry od_rgb  = Odometry(OdometryType::RGB);
    Odometry od_rgbd = Odometry(OdometryType::RGBD);

    od_icp.compute();
    od_rgb.compute();
    od_rgbd.compute();

    return 0;
}