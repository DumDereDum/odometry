#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>

#include "odometry.hpp"
#include "odometry_func/odometry_frame.hpp"
//#include "odometry_func/odometry_frame_impl.hpp"

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
    //waitKey(10000);
    
    //OdometryFrameImpl * of_mat = new OdometryFrameImplTMat<Mat>(Mat());
    //of_mat->setImage(Mat());
    //OdometryFrameImpl * of_umat = new OdometryFrameImplTMat<UMat>(UMat());
    //of_umat->setImage(Mat());
    
    //Ptr<OdometryFrameImpl> of_mat = makePtr<OdometryFrameImplTMat<Mat>>(Mat());;
    //of_mat->setImage(Mat());
    //Ptr<OdometryFrameImpl> of_umat = makePtr<OdometryFrameImplTMat<UMat>>(UMat());;
    //of_umat->setImage(UMat());

    //Ptr<OdometryFrameImpl> of_mat = createOdometryFrame(Mat());
    //of_mat->setImage(Mat());
    //Ptr<OdometryFrameImpl> of_umat = createOdometryFrame(UMat());
    //of_umat->setImage(UMat());

    //OdometryFrame odf;
    //odf.create(Mat());
    //odf.create(UMat());

    Odometry od_icp = Odometry(OdometryType::ICP);
    OdometryFrame odf;
    odf.create(Mat());
    od_icp.compute(odf);
    std::cout << std::endl;
    
    Odometry od_rgb = Odometry(OdometryType::RGB);
    od_rgb.compute(odf);
    std::cout << std::endl;
    
    Odometry od_rgbd = Odometry(OdometryType::RGBD);
    od_rgbd.compute(odf);
    std::cout << std::endl;

    return 0;
}