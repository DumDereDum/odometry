#include <iostream>

#include <opencv2/highgui.hpp>

#include "odometry.hpp"
#include "odometry_func/odometry_frame.hpp"


int main()
{
    Odometry od_icp  = Odometry(OdometryType::ICP);
    Odometry od_rgb  = Odometry(OdometryType::RGB);
    Odometry od_rgbd = Odometry(OdometryType::RGBD);

    od_icp.compute();
    od_rgb.compute();
    od_rgbd.compute();

    //Mat img(480, 640, CV_8UC3, Scalar(10, 100, 150));

    //imshow("test", img);

    return 0;
}