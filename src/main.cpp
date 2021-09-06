#include <iostream>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/utility.hpp>

#include "odometry.hpp"
#include "odometry_func/odometry_frame.hpp"
#include "odometry_func/odometry_settings.hpp"
#include "tmp.hpp"

using namespace cv;

int main(int argc, char** argv)
{
    Size frameSize = Size(640, 480);
    float fx, fy, cx, cy;
    fx = fy = 525.f;
    cx = frameSize.width / 2 - 0.5f;
    cy = frameSize.height / 2 - 0.5f;
    Matx33f intr = Matx33f(fx, 0, cx,
        0, fy, cy,
        0, 0, 1);
    //float depthFactor = 5000;
    double depthFactor = 0.1;
    Ptr<Scene> scene = Scene::create(true, frameSize, intr, depthFactor);

    std::vector<Affine3f> poses = scene->getPoses();
    AccessFlag af = ACCESS_RW;
    Mat depth = scene->depth(poses[1]);
    Mat rgb = scene->rgb(poses[1]);
    //UMat depth = scene->depth(poses[1]).getUMat(af);
    //UMat rgb = scene->rgb(poses[1]).getUMat(af);

    //imshow("depth", depth);
    //imshow("rgb", rgb);
    //waitKey(1000);

    OdometryFrame odf = OdometryFrame(Mat());
    //OdometryFrame odf = OdometryFrame(UMat());
    odf.setImage(rgb);
    odf.setDepth(depth);
    odf.findMask(depth);

    Mat odf_rgb, odf_depth, odf_mask;
    //UMat odf_rgb, odf_depth;
    odf.getImage(odf_rgb);
    odf.getDepth(odf_depth);
    odf.getMask(odf_mask);

    //imshow("odf_depth", odf_depth);
    //imshow("odf_rgb", odf_rgb);
    //imshow("odf_mask", odf_mask);
    //waitKey(60000);
    
    odf.prepareRGBFrame();


    /*
    Ptr<OdometryFrame> frame_prev = Ptr<OdometryFrame>(new OdometryFrame()),
                       frame_curr = Ptr<OdometryFrame>(new OdometryFrame());
    Ptr<Odometry> odometry = Odometry::create(string(argv[3]) + "Odometry");
    */

    ///*
    OdometrySettings ods;
    ods.setCameraMatrix(Mat());

    Odometry od_icp = Odometry(OdometryType::ICP, ods);
    //OdometryFrame odf();
    Affine3f Rt;
    od_icp.compute(odf, odf, Rt.matrix);
    std::cout << std::endl;
    
    Odometry od_rgb = Odometry(OdometryType::RGB, ods);
    od_rgb.compute(odf, odf, Rt.matrix);
    std::cout << std::endl;
    
    Odometry od_rgbd = Odometry(OdometryType::RGBD, ods);
    od_rgbd.compute(odf, odf, Rt.matrix);
    std::cout << std::endl;
    //*/

    return 0;
}

/*
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

}
*/