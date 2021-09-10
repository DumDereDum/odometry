#ifndef ODOMETRY_FUNCTIONS_HPP
#define ODOMETRY_FUNCTIONS_HPP

#include "odometry_frame.hpp"

using namespace cv;

static inline
void checkImage(InputArray image)
{
    if (image.empty())
        //CV_Error(Error::StsBadSize, "Image is empty.");
        std::cout << "Image is empty." << std::endl;
    if (image.type() != CV_8UC1)
        //CV_Error(Error::StsBadSize, "Image type has to be CV_8UC1.");
        std::cout << "Image type has to be CV_8UC1." << image.type() << std::endl;
}
static inline
void checkDepth(InputArray depth, const Size& imageSize)
{
    if (depth.empty())
        //CV_Error(Error::StsBadSize, "Depth is empty.");
        std::cout << "Depth is empty." << std::endl;
    if (depth.size() != imageSize)
        //CV_Error(Error::StsBadSize, "Depth has to have the size equal to the image size.");
        std::cout << "Depth has to have the size equal to the image size." << std::endl;
    if (depth.type() != CV_32FC1)
        //CV_Error(Error::StsBadSize, "Depth type has to be CV_32FC1.");
        std::cout << "Depth type has to be CV_32FC1." << std::endl;
}

static inline
void checkMask(InputArray mask, const Size& imageSize)
{
    if (!mask.empty())
    {
        if (mask.size() != imageSize)
            //CV_Error(Error::StsBadSize, "Mask has to have the size equal to the image size.");
            std::cout << "Mask has to have the size equal to the image size." << std::endl;
        if (mask.type() != CV_8UC1)
            //CV_Error(Error::StsBadSize, "Mask type has to be CV_8UC1.");
            std::cout << "Mask type has to be CV_8UC1." << std::endl;
    }
}

static inline
void checkNormals(InputArray normals, const Size& depthSize)
{
    if (normals.size() != depthSize)
        //CV_Error(Error::StsBadSize, "Normals has to have the size equal to the depth size.");
        std::cout << "Normals has to have the size equal to the depth size." << std::endl;
    if (normals.type() != CV_32FC3)
        //CV_Error(Error::StsBadSize, "Normals type has to be CV_32FC3.");
        std::cout << "Normals type has to be CV_32FC3." << std::endl;
}
//Size prepareFramesCache(OdometryFrame srcFrame, OdometryFrame dstFrame);
bool prepareRGBFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame);
bool prepareRGBFrameBase(OdometryFrame& frame);
bool prepareRGBFrameSrc (OdometryFrame frame);
bool prepareRGBFrameDst (OdometryFrame frame);

void setPyramids(OdometryFrame& odf, OdometryFramePyramidType oftype, InputArrayOfArrays pyramidImage);
std::vector<Mat> getPyramids(OdometryFrame& odf, OdometryFramePyramidType oftype);

static void preparePyramidImage(InputArray image, InputOutputArrayOfArrays pyramidImage, size_t levelCount);

template<typename TMat>
void preparePyramidMask(InputArray mask, InputArrayOfArrays pyramidDepth, float minDepth, float maxDepth, InputArrayOfArrays pyramidNormal, InputOutputArrayOfArrays pyramidMask);

template<typename TMat>
void preparePyramidCloud(InputArrayOfArrays pyramidDepth, const Matx33f& cameraMatrix, InputOutputArrayOfArrays pyramidCloud, InputArrayOfArrays pyramidMask);

static
void buildPyramidCameraMatrix(const Matx33f& cameraMatrix, int levels, std::vector<Matx33f>& pyramidCameraMatrix);

//template<typename TMat>
//void preparePyramidSobel(InputArrayOfArrays pyramidImage, int dx, int dy, InputOutputArrayOfArrays pyramidSobel);






#endif //ODOMETRY_FUNCTIONS_HPP