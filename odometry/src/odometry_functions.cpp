#include "odometry_functions.hpp"
//#include "utils.hpp"
#include "../include/depth_to_3d.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

bool prepareRGBFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, OdometrySettings settings)
{
    std::cout << "prepareRGBFrame()" << std::endl;
    prepareRGBFrameBase(srcFrame, settings);
    prepareRGBFrameSrc(srcFrame, settings);
    prepareRGBFrameDst(dstFrame, settings);

	return true;
}

bool prepareRGBFrameBase(OdometryFrame& frame, OdometrySettings settings)
{
    std::cout << "prepareRGBFrameBase()" << std::endl;
    // Can be transformed into template argument in the future
    // when this algorithm supports OCL UMats too
 
    typedef Mat TMat;

    TMat image;
    frame.getGrayImage(image);
    if (image.empty())
    {
        if (frame.getPyramidLevels(OdometryFramePyramidType::PYR_IMAGE) > 0)
        {
            TMat pyr0;
            frame.getPyramidAt(pyr0, OdometryFramePyramidType::PYR_IMAGE, 0);
            frame.setImage(pyr0);
        }
        else
            CV_Error(Error::StsBadSize, "Image or pyramidImage have to be set.");
    }
    checkImage(image);


    TMat depth;
    frame.getDepth(depth);
    if (depth.empty())
    {
        if (frame.getPyramidLevels(OdometryFramePyramidType::PYR_DEPTH) > 0)
        {
            TMat pyr0;
            frame.getPyramidAt(pyr0, OdometryFramePyramidType::PYR_DEPTH, 0);
            frame.setDepth(pyr0);
        }
        else if (frame.getPyramidLevels(OdometryFramePyramidType::PYR_CLOUD) > 0)
        {
            TMat cloud;
            frame.getPyramidAt(cloud, OdometryFramePyramidType::PYR_CLOUD, 0);
            std::vector<TMat> xyz;
            split(cloud, xyz);
            frame.setDepth(xyz[2]);
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }
    checkDepth(depth, image.size());

    TMat mask;
    frame.getMask(mask);
    if (mask.empty() && frame.getPyramidLevels(OdometryFramePyramidType::PYR_MASK) > 0)
    {
        TMat pyr0;
        frame.getPyramidAt(pyr0, OdometryFramePyramidType::PYR_MASK, 0);
        frame.setMask(pyr0);
    }
    checkMask(mask, image.size());

    std::vector<int> iterCounts;
    Mat miterCounts;
    settings.getIterCounts(miterCounts);
    for (int i = 0; i < miterCounts.size().height; i++)
        iterCounts.push_back(miterCounts.at<int>(i));

    std::vector<TMat> ipyramids;
    preparePyramidImage(image, ipyramids, iterCounts.size());
    setPyramids(frame, OdometryFramePyramidType::PYR_IMAGE, ipyramids);

    std::vector<TMat> dpyramids;
    preparePyramidImage(depth, dpyramids, iterCounts.size());
    setPyramids(frame, OdometryFramePyramidType::PYR_DEPTH, dpyramids);

    std::vector<TMat> mpyramids;
    std::vector<TMat> npyramids;
    preparePyramidMask<TMat>(mask, dpyramids, settings.getMinDepth(), settings.getMaxDepth(),
        npyramids, mpyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_MASK, mpyramids);

	return true;
}

bool prepareRGBFrameSrc(OdometryFrame& frame, OdometrySettings settings)
{
    std::cout << "prepareRGBFrameSrc()" << std::endl;
    typedef Mat TMat;
    std::vector<TMat> dpyramids = getPyramids(frame, OdometryFramePyramidType::PYR_DEPTH);
    std::vector<TMat> mpyramids = getPyramids(frame, OdometryFramePyramidType::PYR_MASK);
    std::vector<TMat> cpyramids;
    Matx33f cameraMatrix;
    settings.getCameraMatrix(cameraMatrix);
    
    preparePyramidCloud<TMat>(dpyramids, cameraMatrix, cpyramids, mpyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_CLOUD, cpyramids);
	
    return true;
}

bool prepareRGBFrameDst(OdometryFrame& frame, OdometrySettings settings)
{
    std::cout << "prepareRGBFrameDst()" << std::endl;
    typedef Mat TMat;
    std::vector<TMat> ipyramids = getPyramids(frame, OdometryFramePyramidType::PYR_IMAGE);
    std::vector<TMat> dxpyramids, dypyramids;
    preparePyramidSobel<TMat>(ipyramids, 1, 0, dxpyramids, settings.getSobelSize());
    preparePyramidSobel<TMat>(ipyramids, 1, 0, dypyramids, settings.getSobelSize());
    setPyramids(frame, OdometryFramePyramidType::PYR_DIX, dxpyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_DIY, dypyramids);
    
    return true;
}

static
void preparePyramidImage(InputArray image, InputOutputArrayOfArrays pyramidImage, size_t levelCount)
{
    if (!pyramidImage.empty())
    {
        size_t nLevels = pyramidImage.size(-1).width;
        if (nLevels < levelCount)
            CV_Error(Error::StsBadSize, "Levels count of pyramidImage has to be equal or less than size of iterCounts.");

        CV_Assert(pyramidImage.size(0) == image.size());
        for (size_t i = 0; i < nLevels; i++)
            CV_Assert(pyramidImage.type((int)i) == image.type());
    }
    else
        buildPyramid(image, pyramidImage, (int)levelCount - 1);
}

void setPyramids(OdometryFrame& odf, OdometryFramePyramidType oftype, InputArrayOfArrays pyramidImage)
{
    size_t nLevels = pyramidImage.size(-1).width;
    std::vector<Mat> pyramids;
    pyramidImage.getMatVector(pyramids);
    odf.setPyramidLevels(nLevels);
    for (size_t l = 0; l < nLevels; l++)
    {
        odf.setPyramidAt(pyramids[l], oftype, l);
    }
}

std::vector<Mat> getPyramids(OdometryFrame& odf, OdometryFramePyramidType oftype)
{
    size_t nLevels = odf.getPyramidLevels(oftype);
    std::vector<Mat> pyramids;
    for (size_t l = 0; l < nLevels; l++)
    {
        Mat img;
        odf.getPyramidAt(img, oftype, l);
        pyramids.push_back(img);
    }
    return pyramids;
}

template<typename TMat>
void preparePyramidMask(InputArray mask, InputArrayOfArrays pyramidDepth, float minDepth, float maxDepth,
    InputArrayOfArrays pyramidNormal,
    InputOutputArrayOfArrays pyramidMask)
{
    minDepth = std::max(0.f, minDepth);

    int nLevels = pyramidDepth.size(-1).width;
    if (!pyramidMask.empty())
    {
        if (pyramidMask.size(-1).width != nLevels)
            CV_Error(Error::StsBadSize, "Levels count of pyramidMask has to be equal to size of pyramidDepth.");

        for (int i = 0; i < pyramidMask.size(-1).width; i++)
        {
            CV_Assert(pyramidMask.size(i) == pyramidDepth.size(i));
            CV_Assert(pyramidMask.type(i) == CV_8UC1);
        }
    }
    else
    {
        TMat validMask;
        if (mask.empty())
            validMask = TMat(pyramidDepth.size(0), CV_8UC1, Scalar(255));
        else
            validMask = getTMat<TMat>(mask, -1).clone();

        buildPyramid(validMask, pyramidMask, nLevels - 1);

        for (int i = 0; i < pyramidMask.size(-1).width; i++)
        {
            TMat levelDepth = getTMat<TMat>(pyramidDepth, i).clone();
            patchNaNs(levelDepth, 0);

            TMat& levelMask = getTMatRef<TMat>(pyramidMask, i);
            TMat gtmin, ltmax, tmpMask;
            cv::compare(levelDepth, Scalar(minDepth), gtmin, CMP_GT);
            cv::compare(levelDepth, Scalar(maxDepth), ltmax, CMP_LT);
            cv::bitwise_and(gtmin, ltmax, tmpMask);
            cv::bitwise_and(levelMask, tmpMask, levelMask);

            if (!pyramidNormal.empty())
            {
                CV_Assert(pyramidNormal.type(i) == CV_32FC3);
                CV_Assert(pyramidNormal.size(i) == pyramidDepth.size(i));
                TMat levelNormal = getTMat<TMat>(pyramidNormal, i).clone();

                TMat validNormalMask;
                // NaN check
                cv::compare(levelNormal, levelNormal, validNormalMask, CMP_EQ);
                CV_Assert(validNormalMask.type() == CV_8UC3);

                std::vector<TMat> channelMasks;
                split(validNormalMask, channelMasks);
                TMat tmpChMask;
                cv::bitwise_and(channelMasks[0], channelMasks[1], tmpChMask);
                cv::bitwise_and(channelMasks[2], tmpChMask, validNormalMask);
                cv::bitwise_and(levelMask, validNormalMask, levelMask);
            }
        }
    }
}

template<typename TMat>
static
void preparePyramidCloud(InputArrayOfArrays pyramidDepth, const Matx33f& cameraMatrix, InputOutputArrayOfArrays pyramidCloud, InputArrayOfArrays pyramidMask)
{
    size_t depthSize = pyramidDepth.size(-1).width;
    size_t cloudSize = pyramidCloud.size(-1).width;
    if (!pyramidCloud.empty())
    {
        if (cloudSize != depthSize)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidCloud.");

        for (size_t i = 0; i < depthSize; i++)
        {
            CV_Assert(pyramidCloud.size((int)i) == pyramidDepth.size((int)i));
            CV_Assert(pyramidCloud.type((int)i) == CV_32FC3);
        }
    }
    else
    {
        std::vector<Matx33f> pyramidCameraMatrix;
        buildPyramidCameraMatrix(cameraMatrix, (int)depthSize, pyramidCameraMatrix);

        pyramidCloud.create((int)depthSize, 1, CV_32FC3, -1);
        for (size_t i = 0; i < depthSize; i++)
        {
            TMat cloud;
            depthTo3d(getTMat<TMat>(pyramidDepth, (int)i), pyramidCameraMatrix[i], cloud, getTMat<TMat>(pyramidMask, (int)i));
            getTMatRef<TMat>(pyramidCloud, (int)i) = cloud;

        }
    }
}

static
void buildPyramidCameraMatrix(const Matx33f& cameraMatrix, int levels, std::vector<Matx33f>& pyramidCameraMatrix)
{
    pyramidCameraMatrix.resize(levels);

    for (int i = 0; i < levels; i++)
    {
        Matx33f levelCameraMatrix = (i == 0) ? cameraMatrix : 0.5f * pyramidCameraMatrix[i - 1];
        levelCameraMatrix(2, 2) = 1.0;
        pyramidCameraMatrix[i] = levelCameraMatrix;
    }
}


template<typename TMat>
void preparePyramidSobel(InputArrayOfArrays pyramidImage, int dx, int dy, InputOutputArrayOfArrays pyramidSobel, int sobelSize)
{
    size_t imgLevels = pyramidImage.size(-1).width;
    size_t sobelLvls = pyramidSobel.size(-1).width;
    if (!pyramidSobel.empty())
    {
        if (sobelLvls != imgLevels)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidSobel.");

        for (size_t i = 0; i < sobelLvls; i++)
        {
            CV_Assert(pyramidSobel.size((int)i) == pyramidImage.size((int)i));
            CV_Assert(pyramidSobel.type((int)i) == CV_16SC1);
        }
    }
    else
    {
        pyramidSobel.create((int)imgLevels, 1, CV_16SC1, -1);
        for (size_t i = 0; i < imgLevels; i++)
        {
            Sobel(getTMat<TMat>(pyramidImage, (int)i), getTMatRef<TMat>(pyramidSobel, (int)i), CV_16S, dx, dy, sobelSize);
        }
    }
}


static
bool RGBDICPOdometryImpl(OutputArray _Rt, const Mat& initRt,
                         const OdometryFrame srcFrame,
                         const OdometryFrame dstFrame,
                         const Matx33f& cameraMatrix,
                         float maxDepthDiff, const std::vector<int>& iterCounts,
                         double maxTranslation, double maxRotation,
                         int method, int transfromType)
{
    int transformDim = -1;
    CalcRgbdEquationCoeffsPtr rgbdEquationFuncPtr = 0;
    CalcICPEquationCoeffsPtr icpEquationFuncPtr = 0;
    switch(transfromType)
    {
    case OdometryTransformType::RIGID_BODY_MOTION:
        transformDim = 6;
        rgbdEquationFuncPtr = calcRgbdEquationCoeffs;
        icpEquationFuncPtr = calcICPEquationCoeffs;
        break;
    case OdometryTransformType::ROTATION:
        transformDim = 3;
        rgbdEquationFuncPtr = calcRgbdEquationCoeffsRotation;
        icpEquationFuncPtr = calcICPEquationCoeffsRotation;
        break;
    case OdometryTransformType::TRANSLATION:
        transformDim = 3;
        rgbdEquationFuncPtr = calcRgbdEquationCoeffsTranslation;
        icpEquationFuncPtr = calcICPEquationCoeffsTranslation;
        break;
    default:
        CV_Error(Error::StsBadArg, "Incorrect transformation type");
    }
/*
    const int minOverdetermScale = 20;
    const int minCorrespsCount = minOverdetermScale * transformDim;

    std::vector<Matx33f> pyramidCameraMatrix;
    buildPyramidCameraMatrix(cameraMatrix, (int)iterCounts.size(), pyramidCameraMatrix);

    Mat resultRt = initRt.empty() ? Mat::eye(4,4,CV_64FC1) : initRt.clone();
    Mat currRt, ksi;

    bool isOk = false;
    for(int level = (int)iterCounts.size() - 1; level >= 0; level--)
    {
        const Matx33f& levelCameraMatrix = pyramidCameraMatrix[level];
        const Matx33f& levelCameraMatrix_inv = levelCameraMatrix.inv(DECOMP_SVD);
        const Mat srcLevelDepth, dstLevelDepth;
        srcFrame.getPyramidAt(srcLevelDepth, OdometryFramePyramidType::PYR_DEPTH, level);
        dstFrame.getPyramidAt(dstLevelDepth, OdometryFramePyramidType::PYR_DEPTH, level);

        const double fx = levelCameraMatrix(0, 0);
        const double fy = levelCameraMatrix(1, 1);
        const double determinantThreshold = 1e-6;

        Mat AtA_rgbd, AtB_rgbd, AtA_icp, AtB_icp;
        Mat corresps_rgbd, corresps_icp;

        // Run transformation search on current level iteratively.
        for(int iter = 0; iter < iterCounts[level]; iter ++)
        {
            Mat resultRt_inv = resultRt.inv(DECOMP_SVD);

            const Mat pyramidMask;
            srcFrame->getPyramidAt(pyramidMask, OdometryFramePyramidType::PYR_MASK, level);

            if(method & RGBD_ODOMETRY)
            {
                const Mat pyramidTexturedMask;
                dstFrame->getPyramidAt(pyramidTexturedMask, OdometryFramePyramidType::PYR_TEXMASK, level);
                computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
                                srcLevelDepth, pyramidMask, dstLevelDepth, pyramidTexturedMask,
                                maxDepthDiff, corresps_rgbd);
            }

            if(method & ICP_ODOMETRY)
            {
                const Mat pyramidNormalsMask;
                dstFrame->getPyramidAt(pyramidNormalsMask, OdometryFramePyramidType::PYR_NORMMASK, level);
                computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
                                srcLevelDepth, pyramidMask, dstLevelDepth, pyramidNormalsMask,
                                maxDepthDiff, corresps_icp);
            }

            if(corresps_rgbd.rows < minCorrespsCount && corresps_icp.rows < minCorrespsCount)
                break;

            const Mat srcPyrCloud;
            srcFrame->getPyramidAt(srcPyrCloud, OdometryFramePyramidType::PYR_CLOUD, level);


            Mat AtA(transformDim, transformDim, CV_64FC1, Scalar(0)), AtB(transformDim, 1, CV_64FC1, Scalar(0));
            if(corresps_rgbd.rows >= minCorrespsCount)
            {
                const Mat srcPyrImage, dstPyrImage, dstPyrIdx, dstPyrIdy;
                srcFrame->getPyramidAt(srcPyrImage, OdometryFramePyramidType::PYR_IMAGE, level);
                dstFrame->getPyramidAt(dstPyrImage, OdometryFramePyramidType::PYR_IMAGE, level);
                dstFrame->getPyramidAt(dstPyrIdx, OdometryFramePyramidType::PYR_DIX, level);
                dstFrame->getPyramidAt(dstPyrIdy, OdometryFramePyramidType::PYR_DIY, level);
                calcRgbdLsmMatrices(srcPyrImage, srcPyrCloud, resultRt, dstPyrImage, dstPyrIdx, dstPyrIdy,
                                    corresps_rgbd, fx, fy, sobelScale,
                                    AtA_rgbd, AtB_rgbd, rgbdEquationFuncPtr, transformDim);

                AtA += AtA_rgbd;
                AtB += AtB_rgbd;
            }
            if(corresps_icp.rows >= minCorrespsCount)
            {
                const Mat dstPyrCloud, dstPyrNormals;
                dstFrame->getPyramidAt(dstPyrCloud, OdometryFramePyramidType::PYR_CLOUD, level);
                dstFrame->getPyramidAt(dstPyrNormals, OdometryFramePyramidType::PYR_NORM, level);
                calcICPLsmMatrices(srcPyrCloud, resultRt, dstPyrCloud, dstPyrNormals,
                                   corresps_icp, AtA_icp, AtB_icp, icpEquationFuncPtr, transformDim);
                AtA += AtA_icp;
                AtB += AtB_icp;
            }

            bool solutionExist = solveSystem(AtA, AtB, determinantThreshold, ksi);
            if(!solutionExist)
                break;

            if(transfromType == Odometry::ROTATION)
            {
                Mat tmp(6, 1, CV_64FC1, Scalar(0));
                ksi.copyTo(tmp.rowRange(0,3));
                ksi = tmp;
            }
            else if(transfromType == Odometry::TRANSLATION)
            {
                Mat tmp(6, 1, CV_64FC1, Scalar(0));
                ksi.copyTo(tmp.rowRange(3,6));
                ksi = tmp;
            }

            computeProjectiveMatrix(ksi, currRt);
            resultRt = currRt * resultRt;
            isOk = true;
        }
    }
    _Rt.create(resultRt.size(), resultRt.type());
    Mat Rt = _Rt.getMat();
    resultRt.copyTo(Rt);

    if(isOk)
    {
        Mat deltaRt;
        if(initRt.empty())
            deltaRt = resultRt;
        else
            deltaRt = resultRt * initRt.inv(DECOMP_SVD);

        isOk = testDeltaTransformation(deltaRt, maxTranslation, maxRotation);
    }

    return isOk;
*/
    return true;
}