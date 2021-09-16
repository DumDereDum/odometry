#include "odometry_functions.hpp"
//#include "utils.hpp"
#include "../include/depth_to_3d.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <opencv2/core/dualquaternion.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

bool prepareRGBFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame, OdometrySettings settings)
{
    std::cout << "prepareRGBFrame()" << std::endl;

    prepareRGBFrameBase(srcFrame, settings);
    prepareRGBFrameBase(dstFrame, settings);
    
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
    std::vector<TMat> mpyramids = getPyramids(frame, OdometryFramePyramidType::PYR_MASK);
    std::vector<TMat> dxpyramids, dypyramids, tmpyramids;

    Mat _minGradientMagnitudes;
    std::vector<float> minGradientMagnitudes;
    settings.getMinGradientMagnitudes(_minGradientMagnitudes);
    for (int i = 0; i < _minGradientMagnitudes.size().height; i++)
        minGradientMagnitudes.push_back(_minGradientMagnitudes.at<float>(i));

    preparePyramidSobel<TMat>(ipyramids, 1, 0, dxpyramids, settings.getSobelSize());
    preparePyramidSobel<TMat>(ipyramids, 0, 1, dypyramids, settings.getSobelSize());
    preparePyramidTexturedMask(dxpyramids, dypyramids, minGradientMagnitudes,
        mpyramids, settings.getMaxPointsPart(), tmpyramids, settings.getSobelScale());
    
    setPyramids(frame, OdometryFramePyramidType::PYR_DIX, dxpyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_DIY, dypyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_TEXMASK, tmpyramids);
    
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
            //depthTo3d(getTMat<TMat>(pyramidDepth, (int)i), pyramidCameraMatrix[i], cloud, getTMat<TMat>(pyramidMask, (int)i));
            depthTo3d(getTMat<TMat>(pyramidDepth, (int)i), pyramidCameraMatrix[i], cloud, Mat());
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
void preparePyramidTexturedMask(InputArrayOfArrays pyramid_dI_dx, InputArrayOfArrays pyramid_dI_dy,
    InputArray minGradMagnitudes, InputArrayOfArrays pyramidMask, double maxPointsPart,
    InputOutputArrayOfArrays pyramidTexturedMask, double sobelScale)
{
    size_t didxLevels = pyramid_dI_dx.size(-1).width;
    size_t texLevels = pyramidTexturedMask.size(-1).width;
    if (!pyramidTexturedMask.empty())
    {
        if (texLevels != didxLevels)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidTexturedMask.");

        for (size_t i = 0; i < texLevels; i++)
        {
            CV_Assert(pyramidTexturedMask.size((int)i) == pyramid_dI_dx.size((int)i));
            CV_Assert(pyramidTexturedMask.type((int)i) == CV_8UC1);
        }
    }
    else
    {
        CV_Assert(minGradMagnitudes.type() == CV_32F);
        Mat_<float> mgMags = minGradMagnitudes.getMat();

        const float sobelScale2_inv = 1. / (float)(sobelScale * sobelScale);
        pyramidTexturedMask.create((int)didxLevels, 1, CV_8UC1, -1);
        for (size_t i = 0; i < didxLevels; i++)
        {
            const float minScaledGradMagnitude2 = mgMags((int)i) * mgMags((int)i) * sobelScale2_inv;
            const Mat& dIdx = pyramid_dI_dx.getMat((int)i);
            const Mat& dIdy = pyramid_dI_dy.getMat((int)i);

            Mat texturedMask(dIdx.size(), CV_8UC1, Scalar(0));

            for (int y = 0; y < dIdx.rows; y++)
            {
                const short* dIdx_row = dIdx.ptr<short>(y);
                const short* dIdy_row = dIdy.ptr<short>(y);
                uchar* texturedMask_row = texturedMask.ptr<uchar>(y);
                for (int x = 0; x < dIdx.cols; x++)
                {
                    float magnitude2 = static_cast<float>(dIdx_row[x] * dIdx_row[x] + dIdy_row[x] * dIdy_row[x]);
                    if (magnitude2 >= minScaledGradMagnitude2)
                        texturedMask_row[x] = 255;
                }
            }
            Mat texMask = texturedMask & pyramidMask.getMat((int)i);
            
            randomSubsetOfMask(texMask, (float)maxPointsPart);
            pyramidTexturedMask.getMatRef((int)i) = texMask;
        }
    }
}

static
void randomSubsetOfMask(InputOutputArray _mask, float part)
{
    const int minPointsCount = 1000; // minimum point count (we can process them fast)
    const int nonzeros = countNonZero(_mask);
    const int needCount = std::max(minPointsCount, int(_mask.total() * part));
    if (needCount < nonzeros)
    {
        RNG rng;
        Mat mask = _mask.getMat();
        Mat subset(mask.size(), CV_8UC1, Scalar(0));

        int subsetSize = 0;
        while (subsetSize < needCount)
        {
            int y = rng(mask.rows);
            int x = rng(mask.cols);
            if (mask.at<uchar>(y, x))
            {
                subset.at<uchar>(y, x) = 255;
                mask.at<uchar>(y, x) = 0;
                subsetSize++;
            }
        }
        _mask.assign(subset);
    }
}

bool RGBDICPOdometryImpl(OutputArray _Rt, const Mat& initRt,
                         const OdometryFrame srcFrame,
                         const OdometryFrame dstFrame,
                         const Matx33f& cameraMatrix,
                         float maxDepthDiff, const std::vector<int>& iterCounts,
                         double maxTranslation, double maxRotation, double sobelScale,
                         OdometryType method, OdometryTransformType transfromType)
{
    std::cout << "RGBDICPOdometryImpl()" << std::endl;
    
    //Mat img0, img1;
    //srcFrame.getImage(img0);
    //dstFrame.getImage(img1);
    //imshow("img0", img0);
    //imshow("img1", img1);
    //waitKey(10000);
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
            srcFrame.getPyramidAt(pyramidMask, OdometryFramePyramidType::PYR_MASK, level);

            if(method == OdometryType::RGB)
            {
                const Mat pyramidTexturedMask;
                dstFrame.getPyramidAt(pyramidTexturedMask, OdometryFramePyramidType::PYR_TEXMASK, level);
                computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
                                srcLevelDepth, pyramidMask, dstLevelDepth, pyramidTexturedMask,
                                maxDepthDiff, corresps_rgbd);
            }

            if(method == OdometryType::ICP)
            {
                const Mat pyramidNormalsMask;
                dstFrame.getPyramidAt(pyramidNormalsMask, OdometryFramePyramidType::PYR_NORMMASK, level);
                computeCorresps(levelCameraMatrix, levelCameraMatrix_inv, resultRt_inv,
                                srcLevelDepth, pyramidMask, dstLevelDepth, pyramidNormalsMask,
                                maxDepthDiff, corresps_icp);
            }


            if(corresps_rgbd.rows < minCorrespsCount && corresps_icp.rows < minCorrespsCount)
                break;
            const Mat srcPyrCloud;
            srcFrame.getPyramidAt(srcPyrCloud, OdometryFramePyramidType::PYR_CLOUD, level);


            Mat AtA(transformDim, transformDim, CV_64FC1, Scalar(0)), AtB(transformDim, 1, CV_64FC1, Scalar(0));
            if(corresps_rgbd.rows >= minCorrespsCount)
            {
                const Mat srcPyrImage, dstPyrImage, dstPyrIdx, dstPyrIdy;
                srcFrame.getPyramidAt(srcPyrImage, OdometryFramePyramidType::PYR_IMAGE, level);
                dstFrame.getPyramidAt(dstPyrImage, OdometryFramePyramidType::PYR_IMAGE, level);
                dstFrame.getPyramidAt(dstPyrIdx, OdometryFramePyramidType::PYR_DIX, level);
                dstFrame.getPyramidAt(dstPyrIdy, OdometryFramePyramidType::PYR_DIY, level);
                calcRgbdLsmMatrices(srcPyrImage, srcPyrCloud, resultRt, dstPyrImage, dstPyrIdx, dstPyrIdy,
                                    corresps_rgbd, fx, fy, sobelScale,
                                    AtA_rgbd, AtB_rgbd, rgbdEquationFuncPtr, transformDim);
                AtA += AtA_rgbd;
                AtB += AtB_rgbd;
            }
            if(corresps_icp.rows >= minCorrespsCount)
            {
                const Mat dstPyrCloud, dstPyrNormals;
                dstFrame.getPyramidAt(dstPyrCloud, OdometryFramePyramidType::PYR_CLOUD, level);
                dstFrame.getPyramidAt(dstPyrNormals, OdometryFramePyramidType::PYR_NORM, level);
                calcICPLsmMatrices(srcPyrCloud, resultRt, dstPyrCloud, dstPyrNormals,
                                   corresps_icp, AtA_icp, AtB_icp, icpEquationFuncPtr, transformDim);
                AtA += AtA_icp;
                AtB += AtB_icp;
            }
            //std::cout << AtA << std::endl;
            //std::cout << AtB << std::endl;
            //std::cout << std::endl;
            bool solutionExist = solveSystem(AtA, AtB, determinantThreshold, ksi);
            if(!solutionExist)
                break;
            
            if(transfromType == OdometryTransformType::ROTATION)
            {
                Mat tmp(6, 1, CV_64FC1, Scalar(0));
                ksi.copyTo(tmp.rowRange(0,3));
                ksi = tmp;
            }
            else if(transfromType == OdometryTransformType::TRANSLATION)
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
}


void computeCorresps(const Matx33f& _K, const Matx33f& _K_inv, const Mat& Rt,
    const Mat& depth0, const Mat& validMask0,
    const Mat& depth1, const Mat& selectMask1, float maxDepthDiff,
    Mat& _corresps)
{
    //imshow("d0", depth0);
    //imshow("d1", depth1);
    //waitKey(10000);
    CV_Assert(Rt.type() == CV_64FC1);

    Mat corresps(depth1.size(), CV_16SC2, Scalar::all(-1));

    Matx33d K(_K), K_inv(_K_inv);
    //std::cout << K << std::endl;
    //std::cout << std::endl;
    Rect r(0, 0, depth1.cols, depth1.rows);
    Mat Kt = Rt(Rect(3, 0, 1, 3)).clone();
    Kt = K * Kt;
    const double* Kt_ptr = Kt.ptr<const double>();

    AutoBuffer<float> buf(3 * (depth1.cols + depth1.rows));
    float* KRK_inv0_u1 = buf.data();
    float* KRK_inv1_v1_plus_KRK_inv2 = KRK_inv0_u1 + depth1.cols;
    float* KRK_inv3_u1 = KRK_inv1_v1_plus_KRK_inv2 + depth1.rows;
    float* KRK_inv4_v1_plus_KRK_inv5 = KRK_inv3_u1 + depth1.cols;
    float* KRK_inv6_u1 = KRK_inv4_v1_plus_KRK_inv5 + depth1.rows;
    float* KRK_inv7_v1_plus_KRK_inv8 = KRK_inv6_u1 + depth1.cols;
    {
        Mat R = Rt(Rect(0, 0, 3, 3)).clone();

        Mat KRK_inv = K * R * K_inv;
        //std::cout << std::endl;
        //std::cout << K << std::endl;
        //std::cout << R << std::endl;
        //std::cout << K_inv << std::endl;
        //std::cout << KRK_inv << std::endl;
        const double* KRK_inv_ptr = KRK_inv.ptr<const double>();
        for (int u1 = 0; u1 < depth1.cols; u1++)
        {
            KRK_inv0_u1[u1] = (float)(KRK_inv_ptr[0] * u1);
            KRK_inv3_u1[u1] = (float)(KRK_inv_ptr[3] * u1);
            KRK_inv6_u1[u1] = (float)(KRK_inv_ptr[6] * u1);
            //std::cout << KRK_inv0_u1[u1]<<" " << KRK_inv3_u1[u1] <<" " << KRK_inv6_u1[u1] << std::endl;
        }

        for (int v1 = 0; v1 < depth1.rows; v1++)
        {
            KRK_inv1_v1_plus_KRK_inv2[v1] = (float)(KRK_inv_ptr[1] * v1 + KRK_inv_ptr[2]);
            KRK_inv4_v1_plus_KRK_inv5[v1] = (float)(KRK_inv_ptr[4] * v1 + KRK_inv_ptr[5]);
            KRK_inv7_v1_plus_KRK_inv8[v1] = (float)(KRK_inv_ptr[7] * v1 + KRK_inv_ptr[8]);
        }
    }

    int correspCount = 0;
    for (int v1 = 0; v1 < depth1.rows; v1++)
    {
        const float* depth1_row = depth1.ptr<float>(v1);
        const uchar* mask1_row = selectMask1.ptr<uchar>(v1);
        for (int u1 = 0; u1 < depth1.cols; u1++)
        {
            float d1 = depth1_row[u1];
            if (mask1_row[u1])
            {
                CV_DbgAssert(!cvIsNaN(d1));
                float transformed_d1 = static_cast<float>(d1 * (KRK_inv6_u1[u1] + KRK_inv7_v1_plus_KRK_inv8[v1]) +
                    Kt_ptr[2]);
                //std::cout << Vec2i(u1, v1) << " | "  << d1 << "*" << KRK_inv6_u1[u1] << "+" << KRK_inv7_v1_plus_KRK_inv8[v1] << "+" << Kt_ptr[2] << std::endl;
                if (transformed_d1 > 0)
                {
                    float transformed_d1_inv = 1.f / transformed_d1;
                    int u0 = cvRound(transformed_d1_inv * (d1 * (KRK_inv0_u1[u1] + KRK_inv1_v1_plus_KRK_inv2[v1]) +
                        Kt_ptr[0]));
                    int v0 = cvRound(transformed_d1_inv * (d1 * (KRK_inv3_u1[u1] + KRK_inv4_v1_plus_KRK_inv5[v1]) +
                        Kt_ptr[1]));
                    //std::cout << Vec2i(u0, v0) << " " << Vec2i(u1, v1) << std::endl;
                    if (r.contains(Point(u0, v0)))
                    {
                        float d0 = depth0.at<float>(v0, u0);
                        if (validMask0.at<uchar>(v0, u0) && std::abs(transformed_d1 - d0) <= maxDepthDiff)
                        {
                            //std::cout << d1 << " "<< transformed_d1 << " "<< d0 <<" "<< maxDepthDiff<< std::endl;
                            //std::cout << Vec2i(v0, u0) << " " << Vec2i(v1, u1) << std::endl;
                            //std::cout << std::endl;
                            CV_DbgAssert(!cvIsNaN(d0));
                            Vec2s& c = corresps.at<Vec2s>(v0, u0);
                            if (c[0] != -1)
                            {
                                int exist_u1 = c[0], exist_v1 = c[1];

                                float exist_d1 = (float)(depth1.at<float>(exist_v1, exist_u1) *
                                    (KRK_inv6_u1[exist_u1] + KRK_inv7_v1_plus_KRK_inv8[exist_v1]) + Kt_ptr[2]);

                                if (transformed_d1 > exist_d1)
                                    continue;
                            }
                            else
                            {
                                correspCount++;
                            }
                            c = Vec2s((short)u1, (short)v1);
                        }
                    }
                }
            }
        }
    }

    _corresps.create(correspCount, 1, CV_32SC4);
    Vec4i* corresps_ptr = _corresps.ptr<Vec4i>();
    for (int v0 = 0, i = 0; v0 < corresps.rows; v0++)
    {
        const Vec2s* corresps_row = corresps.ptr<Vec2s>(v0);
        for (int u0 = 0; u0 < corresps.cols; u0++)
        {
            const Vec2s& c = corresps_row[u0];
            if (c[0] != -1)
                corresps_ptr[i++] = Vec4i(u0, v0, c[0], c[1]);
            //if (c[0] != -1)
            //    std::cout << Vec4i(u0, v0, c[0], c[1]) << std::endl;
            
        }
    }
}

static
void calcRgbdLsmMatrices(const Mat& image0, const Mat& cloud0, const Mat& Rt,
    const Mat& image1, const Mat& dI_dx1, const Mat& dI_dy1,
    const Mat& corresps, double fx, double fy, double sobelScaleIn,
    Mat& AtA, Mat& AtB, CalcRgbdEquationCoeffsPtr func, int transformDim)
{
    AtA = Mat(transformDim, transformDim, CV_64FC1, Scalar(0));
    AtB = Mat(transformDim, 1, CV_64FC1, Scalar(0));
    double* AtB_ptr = AtB.ptr<double>();

    const int correspsCount = corresps.rows;

    CV_Assert(Rt.type() == CV_64FC1);
    const double* Rt_ptr = Rt.ptr<const double>();

    AutoBuffer<float> diffs(correspsCount);
    float* diffs_ptr = diffs.data();

    const Vec4i* corresps_ptr = corresps.ptr<Vec4i>();

    double sigma = 0;
    for (int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u0 = c[0], v0 = c[1];
        int u1 = c[2], v1 = c[3];

        diffs_ptr[correspIndex] = static_cast<float>(static_cast<int>(image0.at<uchar>(v0, u0)) -
                                                     static_cast<int>(image1.at<uchar>(v1, u1)));
        //std::cout << "diffs_ptr[correspIndex] "<< correspIndex << " " << diffs_ptr[correspIndex] << std::endl;
        sigma += diffs_ptr[correspIndex] * diffs_ptr[correspIndex];
    }
    sigma = std::sqrt(sigma / correspsCount);

    std::vector<double> A_buf(transformDim);
    double* A_ptr = &A_buf[0];

    for (int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u0 = c[0], v0 = c[1];
        int u1 = c[2], v1 = c[3];

        double w = sigma + std::abs(diffs_ptr[correspIndex]);
        w = w > DBL_EPSILON ? 1. / w : 1.;

        double w_sobelScale = w * sobelScaleIn;

        const Point3f& p0 = cloud0.at<Point3f>(v0, u0);
        Point3f tp0;
        tp0.x = (float)(p0.x * Rt_ptr[0] + p0.y * Rt_ptr[1] + p0.z * Rt_ptr[2] + Rt_ptr[3]);
        tp0.y = (float)(p0.x * Rt_ptr[4] + p0.y * Rt_ptr[5] + p0.z * Rt_ptr[6] + Rt_ptr[7]);
        tp0.z = (float)(p0.x * Rt_ptr[8] + p0.y * Rt_ptr[9] + p0.z * Rt_ptr[10] + Rt_ptr[11]);

        func(A_ptr,
            w_sobelScale * dI_dx1.at<short int>(v1, u1),
            w_sobelScale * dI_dy1.at<short int>(v1, u1),
            tp0, fx, fy);

        for (int y = 0; y < transformDim; y++)
        {
            double* AtA_ptr = AtA.ptr<double>(y);
            for (int x = y; x < transformDim; x++)
            {
                AtA_ptr[x] += A_ptr[y] * A_ptr[x];
                //std::cout << correspIndex << " + " << AtA_ptr[x] << " = " << A_ptr[y] << " " << A_ptr[x] << std::endl;
            }
            //std::cout << A_ptr[y] << " ";
            AtB_ptr[y] += A_ptr[y] * w * diffs_ptr[correspIndex];
            //std::cout << correspIndex << " + " << AtB_ptr[y] << " = " << A_ptr[y] << " " << w << " " << diffs_ptr[correspIndex] << std::endl;
            //std::cout << std::endl;
        }
        //std::cout << std::endl;
       
    }
    //std::cout << AtA << std::endl;
    //std::cout << AtB << std::endl;
    //std::cout << std::endl;
    for (int y = 0; y < transformDim; y++)
        for (int x = y + 1; x < transformDim; x++)
            AtA.at<double>(x, y) = AtA.at<double>(y, x);
}

static
void calcICPLsmMatrices(const Mat& cloud0, const Mat& Rt,
    const Mat& cloud1, const Mat& normals1,
    const Mat& corresps,
    Mat& AtA, Mat& AtB, CalcICPEquationCoeffsPtr func, int transformDim)
{
    AtA = Mat(transformDim, transformDim, CV_64FC1, Scalar(0));
    AtB = Mat(transformDim, 1, CV_64FC1, Scalar(0));
    double* AtB_ptr = AtB.ptr<double>();

    const int correspsCount = corresps.rows;

    CV_Assert(Rt.type() == CV_64FC1);
    const double* Rt_ptr = Rt.ptr<const double>();

    AutoBuffer<float> diffs(correspsCount);
    float* diffs_ptr = diffs.data();

    AutoBuffer<Point3f> transformedPoints0(correspsCount);
    Point3f* tps0_ptr = transformedPoints0.data();

    const Vec4i* corresps_ptr = corresps.ptr<Vec4i>();

    double sigma = 0;
    for (int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u0 = c[0], v0 = c[1];
        int u1 = c[2], v1 = c[3];

        const Point3f& p0 = cloud0.at<Point3f>(v0, u0);
        Point3f tp0;
        tp0.x = (float)(p0.x * Rt_ptr[0] + p0.y * Rt_ptr[1] + p0.z * Rt_ptr[2] + Rt_ptr[3]);
        tp0.y = (float)(p0.x * Rt_ptr[4] + p0.y * Rt_ptr[5] + p0.z * Rt_ptr[6] + Rt_ptr[7]);
        tp0.z = (float)(p0.x * Rt_ptr[8] + p0.y * Rt_ptr[9] + p0.z * Rt_ptr[10] + Rt_ptr[11]);

        Vec3f n1 = normals1.at<Vec3f>(v1, u1);
        Point3f v = cloud1.at<Point3f>(v1, u1) - tp0;

        tps0_ptr[correspIndex] = tp0;
        diffs_ptr[correspIndex] = n1[0] * v.x + n1[1] * v.y + n1[2] * v.z;
        sigma += diffs_ptr[correspIndex] * diffs_ptr[correspIndex];
    }

    sigma = std::sqrt(sigma / correspsCount);

    std::vector<double> A_buf(transformDim);
    double* A_ptr = &A_buf[0];
    for (int correspIndex = 0; correspIndex < corresps.rows; correspIndex++)
    {
        const Vec4i& c = corresps_ptr[correspIndex];
        int u1 = c[2], v1 = c[3];

        double w = sigma + std::abs(diffs_ptr[correspIndex]);
        w = w > DBL_EPSILON ? 1. / w : 1.;

        func(A_ptr, tps0_ptr[correspIndex], normals1.at<Vec3f>(v1, u1) * w);

        for (int y = 0; y < transformDim; y++)
        {
            double* AtA_ptr = AtA.ptr<double>(y);
            for (int x = y; x < transformDim; x++)
                AtA_ptr[x] += A_ptr[y] * A_ptr[x];

            AtB_ptr[y] += A_ptr[y] * w * diffs_ptr[correspIndex];
        }
    }

    for (int y = 0; y < transformDim; y++)
        for (int x = y + 1; x < transformDim; x++)
            AtA.at<double>(x, y) = AtA.at<double>(y, x);
}


static
bool solveSystem(const Mat& AtA, const Mat& AtB, double detThreshold, Mat& x)
{
    double det = determinant(AtA);
    //std::cout << AtA << std::endl;
    //std::cout << "det: " << det << std::endl;
    if (fabs(det) < detThreshold || cvIsNaN(det) || cvIsInf(det))
        return false;

    solve(AtA, AtB, x, DECOMP_CHOLESKY);

    return true;
}

static
void computeProjectiveMatrix(const Mat& ksi, Mat& Rt)
{
    CV_Assert(ksi.size() == Size(1, 6) && ksi.type() == CV_64FC1);

    const double* ksi_ptr = ksi.ptr<const double>();
    // 0.5 multiplication is here because (dual) quaternions keep half an angle/twist inside
    Matx44d matdq = (DualQuatd(0, ksi_ptr[0], ksi_ptr[1], ksi_ptr[2],
        0, ksi_ptr[3], ksi_ptr[4], ksi_ptr[5]) * 0.5).exp().toMat(QUAT_ASSUME_UNIT);
    Mat(matdq).copyTo(Rt);
}


static
bool testDeltaTransformation(const Mat& deltaRt, double maxTranslation, double maxRotation)
{
    double translation = norm(deltaRt(Rect(3, 0, 1, 3)));

    Mat rvec;
    Rodrigues(deltaRt(Rect(0, 0, 3, 3)), rvec);

    double rotation = norm(rvec) * 180. / CV_PI;

    return translation <= maxTranslation && rotation <= maxRotation;
}