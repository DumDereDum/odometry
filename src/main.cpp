#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>

#include "odometry.hpp"
#include "odometry_func/odometry_frame.hpp"
#include "odometry_func/odometry_frame_impl.hpp"

using namespace cv;

Ptr<OdometryFrameImpl> createOdometryFrame(InputArray image)
{
    bool allEmpty = image.empty();
    bool useOcl = image.isUMat();
    //if (useOcl && !allEmpty)
    if (useOcl)
        return makePtr<OdometryFrameImplTMat<UMat>>(image);
    else
        return makePtr<OdometryFrameImplTMat<Mat>>(image);
}

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
    //waitKey(10000);
    
    //OdometryFrameImpl * of_mat = new OdometryFrameImplTMat<Mat>(Mat());
    //of_mat->setImage(Mat());
    //OdometryFrameImpl * of_umat = new OdometryFrameImplTMat<UMat>(UMat());
    //of_umat->setImage(Mat());
    
    //Ptr<OdometryFrameImpl> of_mat = makePtr<OdometryFrameImplTMat<Mat>>(Mat());;
    //of_mat->setImage(Mat());
    //Ptr<OdometryFrameImpl> of_umat = makePtr<OdometryFrameImplTMat<UMat>>(UMat());;
    //of_umat->setImage(UMat());

    
    Ptr<OdometryFrameImpl> of_mat = createOdometryFrame(Mat());
    of_mat->setImage(Mat());
    Ptr<OdometryFrameImpl> of_umat = createOdometryFrame(UMat());
    of_umat->setImage(UMat());



    return 0;
}