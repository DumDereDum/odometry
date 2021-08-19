#include <iostream>

#include <opencv2/highgui.hpp>

#include "odometry.hpp"
#include "odometry_func/odometry_frame.hpp"
#include "odometry_func/odometry_frame_impl.hpp"

using namespace cv;

int main()
{
    //Odometry od_icp  = Odometry(OdometryType::ICP);
    //Odometry od_rgb  = Odometry(OdometryType::RGB);
    //Odometry od_rgbd = Odometry(OdometryType::RGBD);

    //od_icp.compute();
    //od_rgb.compute();
    //od_rgbd.compute();

    //Mat img(480, 640, CV_8UC3, Scalar(10, 100, 150));

    //imshow("test", img);

    OdometryFrameImpl * of_mat = new OdometryFrameImplTMat<Mat>(Mat());
    of_mat->setImage(Mat());

    OdometryFrameImpl * of_umat = new OdometryFrameImplTMat<UMat>(UMat());
    of_umat->setImage(Mat());



    return 0;
}