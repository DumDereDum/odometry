#ifndef ODOMETRY_FUNCTIONS_HPP
#define ODOMETRY_FUNCTIONS_HPP

#include "odometry_frame.hpp"

using namespace cv;

enum OdometryTransformType
{
    ROTATION = 1, TRANSLATION = 2, RIGID_BODY_MOTION = 4
};

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


static inline
void calcRgbdEquationCoeffs(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz = 1. / p3d.z,
        v0 = dIdx * fx * invz,
        v1 = dIdy * fy * invz,
        v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;

    C[0] = -p3d.z * v1 + p3d.y * v2;
    C[1] = p3d.z * v0 - p3d.x * v2;
    C[2] = -p3d.y * v0 + p3d.x * v1;
    C[3] = v0;
    C[4] = v1;
    C[5] = v2;
}

static inline
void calcRgbdEquationCoeffsRotation(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz = 1. / p3d.z,
        v0 = dIdx * fx * invz,
        v1 = dIdy * fy * invz,
        v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;
    C[0] = -p3d.z * v1 + p3d.y * v2;
    C[1] = p3d.z * v0 - p3d.x * v2;
    C[2] = -p3d.y * v0 + p3d.x * v1;
}

static inline
void calcRgbdEquationCoeffsTranslation(double* C, double dIdx, double dIdy, const Point3f& p3d, double fx, double fy)
{
    double invz = 1. / p3d.z,
        v0 = dIdx * fx * invz,
        v1 = dIdy * fy * invz,
        v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;
    C[0] = v0;
    C[1] = v1;
    C[2] = v2;
}

typedef
void (*CalcRgbdEquationCoeffsPtr)(double*, double, double, const Point3f&, double, double);

static inline
void calcICPEquationCoeffs(double* C, const Point3f& p0, const Vec3f& n1)
{
    C[0] = -p0.z * n1[1] + p0.y * n1[2];
    C[1] = p0.z * n1[0] - p0.x * n1[2];
    C[2] = -p0.y * n1[0] + p0.x * n1[1];
    C[3] = n1[0];
    C[4] = n1[1];
    C[5] = n1[2];
}

static inline
void calcICPEquationCoeffsRotation(double* C, const Point3f& p0, const Vec3f& n1)
{
    C[0] = -p0.z * n1[1] + p0.y * n1[2];
    C[1] = p0.z * n1[0] - p0.x * n1[2];
    C[2] = -p0.y * n1[0] + p0.x * n1[1];
}

static inline
void calcICPEquationCoeffsTranslation(double* C, const Point3f& /*p0*/, const Vec3f& n1)
{
    C[0] = n1[0];
    C[1] = n1[1];
    C[2] = n1[2];
}

typedef
void (*CalcICPEquationCoeffsPtr)(double*, const Point3f&, const Vec3f&);




bool prepareRGBFrame(OdometryFrame& srcFrame, OdometryFrame& dstFrame);
bool prepareRGBFrameBase(OdometryFrame& frame);
bool prepareRGBFrameSrc (OdometryFrame& frame);
bool prepareRGBFrameDst (OdometryFrame& frame);

void setPyramids(OdometryFrame& odf, OdometryFramePyramidType oftype, InputArrayOfArrays pyramidImage);
std::vector<Mat> getPyramids(OdometryFrame& odf, OdometryFramePyramidType oftype);

static void preparePyramidImage(InputArray image, InputOutputArrayOfArrays pyramidImage, size_t levelCount);

template<typename TMat>
void preparePyramidMask(InputArray mask, InputArrayOfArrays pyramidDepth, float minDepth, float maxDepth, InputArrayOfArrays pyramidNormal, InputOutputArrayOfArrays pyramidMask);

template<typename TMat>
void preparePyramidCloud(InputArrayOfArrays pyramidDepth, const Matx33f& cameraMatrix, InputOutputArrayOfArrays pyramidCloud, InputArrayOfArrays pyramidMask);

static
void buildPyramidCameraMatrix(const Matx33f& cameraMatrix, int levels, std::vector<Matx33f>& pyramidCameraMatrix);

template<typename TMat>
void preparePyramidSobel(InputArrayOfArrays pyramidImage, int dx, int dy, InputOutputArrayOfArrays pyramidSobel);


static
bool RGBDICPOdometryImpl(OutputArray _Rt, const Mat& initRt,
    const OdometryFrame srcFrame,
    const OdometryFrame dstFrame,
    const Matx33f& cameraMatrix,
    float maxDepthDiff, const std::vector<int>& iterCounts,
    double maxTranslation, double maxRotation,
    int method, int transfromType);



#endif //ODOMETRY_FUNCTIONS_HPP