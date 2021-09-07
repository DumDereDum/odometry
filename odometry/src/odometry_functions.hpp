#ifndef ODOMETRY_FUNCTIONS_HPP
#define ODOMETRY_FUNCTIONS_HPP

#include "odometry_frame.hpp"

static inline
void checkImage(InputArray image)
{
    if (image.empty())
        CV_Error(Error::StsBadSize, "Image is empty.");
    if (image.type() != CV_8UC1)
        CV_Error(Error::StsBadSize, "Image type has to be CV_8UC1.");
}

//Size prepareFramesCache(OdometryFrame srcFrame, OdometryFrame dstFrame);
bool prepareICPFrame(OdometryFrame srcFrame, OdometryFrame dstFrame);
bool prepareICPFrameBase(OdometryFrame frame);
bool prepareICPFrameSrc (OdometryFrame frame);
bool prepareICPFrameDst (OdometryFrame frame);
//Size prepareRGBFrame(OdometryFrame srcFrame, OdometryFrame dstFrame);
//Size prepareRGBDFrame(OdometryFrame srcFrame, OdometryFrame dstFrame);

#endif //ODOMETRY_FUNCTIONS_HPP