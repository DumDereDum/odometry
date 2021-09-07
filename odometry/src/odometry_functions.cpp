#include "odometry_functions.hpp"

bool prepareICPFrame(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
    std::cout << "prepareICPFrame()" << std::endl;
    prepareICPFrameBase(srcFrame);
	return true;
}

bool prepareICPFrameBase(OdometryFrame frame)
{
    std::cout << "prepareICPFrameBase()" << std::endl;
    // Can be transformed into template argument in the future
    // when this algorithm supports OCL UMats too
/*    
    typedef Mat TMat;

    TMat image;
    frame.getImage(image);
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
    frame->getDepth(depth);
    if (depth.empty())
    {
        if (frame->getPyramidLevels(OdometryFramePyramidType::PYR_DEPTH) > 0)
        {
            TMat pyr0;
            frame->getPyramidAt(pyr0, OdometryFramePyramidType::PYR_DEPTH, 0);
            frame->setDepth(pyr0);
        }
        else if (frame->getPyramidLevels(OdometryFramePyramidType::PYR_CLOUD) > 0)
        {
            TMat cloud;
            frame->getPyramidAt(cloud, OdometryFramePyramidType::PYR_CLOUD, 0);
            std::vector<TMat> xyz;
            split(cloud, xyz);
            frame->setDepth(xyz[2]);
        }
        else
            CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
    }
    checkDepth(depth, image.size());

    TMat mask;
    frame->getMask(mask);
    if (mask.empty() && frame->getPyramidLevels(OdometryFramePyramidType::PYR_MASK) > 0)
    {
        TMat pyr0;
        frame->getPyramidAt(pyr0, OdometryFramePyramidType::PYR_MASK, 0);
        frame->setMask(pyr0);
    }
    checkMask(mask, image.size());

    auto tframe = frame.dynamicCast<OdometryFrameImpl<TMat>>();
    preparePyramidImage(image, tframe->pyramids[OdometryFramePyramidType::PYR_IMAGE], iterCounts.size());

    preparePyramidDepth(depth, tframe->pyramids[OdometryFramePyramidType::PYR_DEPTH], iterCounts.size());

    preparePyramidMask<TMat>(mask, tframe->pyramids[OdometryFramePyramidType::PYR_DEPTH], (float)minDepth, (float)maxDepth,
        tframe->pyramids[OdometryFramePyramidType::PYR_NORM], tframe->pyramids[OdometryFramePyramidType::PYR_MASK]);
*/
	return true;
}

bool prepareICPFrameSrc(OdometryFrame frame)
{
	return true;
}

bool prepareICPFrameDst(OdometryFrame frame)
{
	return true;
}

