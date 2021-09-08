#include "odometry_functions.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

const std::vector<int> iterCounts = { 7, 7, 7, 10 };

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

    std::vector<TMat> pyramids;
    preparePyramidImage(image, pyramids, iterCounts.size());
    setPyramids(frame, OdometryFramePyramidType::PYR_IMAGE, pyramids);
/*
    preparePyramidDepth(depth, tframe->pyramids[OdometryFramePyramidType::PYR_DEPTH], iterCounts.size());

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
