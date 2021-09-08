#include "odometry_functions.hpp"
#include "utils.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

const std::vector<int> iterCounts = { 7, 7, 7, 10 };
const float minDepth = 0.f;
const float maxDepth = 4.f;

bool prepareRGBFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame)
{
    std::cout << "prepareRGBFrame()" << std::endl;
    prepareRGBFrameBase(srcFrame);
	return true;
}

bool prepareRGBFrameBase(OdometryFrame& frame)
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

    std::vector<TMat> ipyramids;
    preparePyramidImage(image, ipyramids, iterCounts.size());
    setPyramids(frame, OdometryFramePyramidType::PYR_IMAGE, ipyramids);

    std::vector<TMat> dpyramids;
    preparePyramidImage(depth, dpyramids, iterCounts.size());
    setPyramids(frame, OdometryFramePyramidType::PYR_DEPTH, dpyramids);

    std::vector<TMat> mpyramids;
    std::vector<TMat> npyramids;
    preparePyramidMask<TMat>(mask, dpyramids, (float)minDepth, (float)maxDepth,
        npyramids, mpyramids);
    setPyramids(frame, OdometryFramePyramidType::PYR_MASK, mpyramids);


/*
    preparePyramidMask<TMat>(mask, tframe->pyramids[OdometryFramePyramidType::PYR_DEPTH], (float)minDepth, (float)maxDepth,
        tframe->pyramids[OdometryFramePyramidType::PYR_NORM], tframe->pyramids[OdometryFramePyramidType::PYR_MASK]);
*/
	return true;
}

bool prepareRGBFrameSrc(OdometryFrame frame)
{
	return true;
}

bool prepareRGBFrameDst(OdometryFrame frame)
{
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

