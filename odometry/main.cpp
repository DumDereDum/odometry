#include <iostream>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/utility.hpp>

#include "include/odometry.hpp"
#include "include/tmp.hpp"

using namespace cv;

void displayOdometryFrame(Mat depth, Mat rgb, OdometryFrame odf)
{
    Mat odf_rgb, odf_depth, odf_gray, odf_mask;
    odf.getImage(odf_rgb);
    odf.getDepth(odf_depth);
    odf.getGrayImage(odf_gray);
    odf.getMask(odf_mask);

    imshow("depth", depth);
    imshow("rgb", rgb);
    imshow("odf_depth", odf_depth);
    imshow("odf_rgb", odf_rgb);
    imshow("odf_gray", odf_gray);
    imshow("odf_mask", odf_mask);
    waitKey(10000);
}

void displayOdometryPyrs(String name, OdometryFrame odf, OdometryFramePyramidType oftype)
{
    size_t nLevels = odf.getPyramidLevels(oftype);
    for (size_t l = 0; l < nLevels; l++)
    {
        Mat img;
        odf.getPyramidAt(img, oftype, l);
        imshow(name, img);
        waitKey(1000);
    }
}

void displaySobel(OdometryFrame odf)
{
    size_t nLevels = odf.getPyramidLevels(OdometryFramePyramidType::PYR_DIX);
    for (size_t l = 0; l < nLevels; l++)
    {
        Mat dx, dy;
        Mat abs_grad_x, abs_grad_y;
        Mat grad;
        odf.getPyramidAt(dx, OdometryFramePyramidType::PYR_DIX, l);
        odf.getPyramidAt(dy, OdometryFramePyramidType::PYR_DIY, l);
        convertScaleAbs(dx, abs_grad_x);
        convertScaleAbs(dy, abs_grad_y);
        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
        imshow("Sobel", grad);
        waitKey(1000);
    }
}

Size frameSize = Size(640, 480);
float fx = 525.f;
float fy = 525.f;
float cx = frameSize.width / 2 - 0.5f;
float cy = frameSize.height / 2 - 0.5f;
Matx33f intr = Matx33f(fx, 0, cx,
    0, fy, cy,
    0, 0, 1);
//float depthFactor = 5000;
float depthFactor = 0.1;

bool display = true;
bool icp     = false;
bool rgb     = true;
bool rgbd    = false;
bool settings= false;

int main(int argc, char** argv)
{
    Ptr<Scene> scene = Scene::create(true, frameSize, intr, depthFactor);
    std::vector<Affine3f> poses = scene->getPoses();
    AccessFlag af = ACCESS_RW;

    /*
    Ptr<OdometryFrame> frame_prev = Ptr<OdometryFrame>(new OdometryFrame()),
                       frame_curr = Ptr<OdometryFrame>(new OdometryFrame());
    Ptr<Odometry> odometry = Odometry::create(string(argv[3]) + "Odometry");
    */

    if (settings)
    {
        OdometrySettings ods;
        const std::vector<int> ic = { 7, 7, 7, 7 };
        ods.setIterCounts(ic);
        Mat icr;
        ods.getIterCounts(icr);
        std::cout << icr << std::endl;
    }

    if (icp)
    {
        Mat depth = scene->depth(poses[1]);
        Mat rgb = scene->rgb(poses[1]);
        
        OdometrySettings ods;
        ods.setCameraMatrix(Mat());
        Odometry od_icp = Odometry(OdometryType::ICP, ods);
        //OdometryFrame odf = OdometryFrame(Mat());
        OdometryFrame odf = od_icp.createOdometryFrame();

        odf.setImage(rgb);
        odf.setDepth(depth);
        odf.findMask(depth);
        
        if (display)
            displayOdometryFrame(depth, rgb, odf);

        /* ICP */
        //OdometrySettings ods;
        //ods.setCameraMatrix(Mat());
        //Odometry od_icp = Odometry(OdometryType::ICP, ods);
        
        Affine3f Rt;
        od_icp.prepareFrames(odf, odf);
        od_icp.compute(odf, odf, Rt.matrix);

    }

    if (rgb)
    {
        Mat depth = scene->depth(poses[1]);
        Mat rgb = scene->rgb(poses[1]);
        
        OdometrySettings ods;
        ods.setCameraMatrix(Mat());
        Odometry od_rgb = Odometry(OdometryType::RGB, ods);
        //OdometryFrame odf = OdometryFrame(Mat());
        OdometryFrame odf = od_rgb.createOdometryFrame();
        
        odf.setImage(rgb);
        odf.setDepth(depth);
        odf.findMask(depth);
        Mat odf_rgb, odf_depth, odf_gray, odf_mask;
        
        if (display)
        {
            displayOdometryFrame(depth, rgb, odf);
        }
        /* RGB */
        //OdometrySettings ods;
        //ods.setCameraMatrix(Mat());
        //Odometry od_rgb = Odometry(OdometryType::RGB, ods);
        
        Affine3f Rt;
        od_rgb.prepareFrames(odf, odf);
        if (display)
        {
            displayOdometryPyrs("PYR_IMAGE", odf, OdometryFramePyramidType::PYR_IMAGE);
            displayOdometryPyrs("PYR_DEPTH", odf, OdometryFramePyramidType::PYR_DEPTH);
            displayOdometryPyrs("PYR_MASK", odf, OdometryFramePyramidType::PYR_MASK);
            displaySobel(odf);
        }
        od_rgb.compute(odf, odf, Rt.matrix);

    }

    if (rgbd)
    {
        Mat depth = scene->depth(poses[1]);
        Mat rgb = scene->rgb(poses[1]);
        
        OdometrySettings ods;
        ods.setCameraMatrix(Mat());
        Odometry od_rgbd = Odometry(OdometryType::RGBD, ods);
        //OdometryFrame odf = OdometryFrame(Mat());
        OdometryFrame odf = od_rgbd.createOdometryFrame();
        
        odf.setImage(rgb);
        odf.setDepth(depth);
        odf.findMask(depth);
        Mat odf_rgb, odf_depth, odf_gray, odf_mask;

        if (display)
            displayOdometryFrame(depth, rgb, odf);

        /* RGBD */
        //OdometrySettings ods;
        //ods.setCameraMatrix(Mat());
        //Odometry od_rgbd = Odometry(OdometryType::RGBD, ods);
        
        Affine3f Rt;
        od_rgbd.prepareFrames(odf, odf);
        od_rgbd.compute(odf, odf, Rt.matrix);

    }

    return 0;
}
