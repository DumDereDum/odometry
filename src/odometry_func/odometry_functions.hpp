#ifndef ODOMETRY_FUNCTIONS_HPP
#define ODOMETRY_FUNCTIONS_HPP

#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/traits.hpp>

using namespace cv;

typedef cv::Vec4f ptype;
inline cv::Vec3f fromPtype(const ptype& x)
{
    return cv::Vec3f(x[0], x[1], x[2]);
}

inline ptype toPtype(const cv::Vec3f& x)
{
    return ptype(x[0], x[1], x[2], 0);
}

typedef float depthType;

template<> class DataType<cv::Point3f>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum {
        generic_type = 0,
        depth = CV_32F,
        channels = 3,
        fmt = (int)'f',
        type = CV_MAKETYPE(depth, channels)
    };
};

template<> class DataType<cv::Vec3f>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum {
        generic_type = 0,
        depth = CV_32F,
        channels = 3,
        fmt = (int)'f',
        type = CV_MAKETYPE(depth, channels)
    };
};

template<> class DataType<cv::Vec4f>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum {
        generic_type = 0,
        depth = CV_32F,
        channels = 4,
        fmt = (int)'f',
        type = CV_MAKETYPE(depth, channels)
    };
};

enum
{
    DEPTH_TYPE = DataType<depthType>::type,
    POINT_TYPE = DataType<ptype    >::type,
    COLOR_TYPE = DataType<ptype    >::type
};

typedef cv::Mat_< ptype > Points;
typedef Points Normals;
typedef Points Colors;

typedef cv::Mat_< depthType > Depth;

const float qnan = std::numeric_limits<float>::quiet_NaN();
const cv::Vec3f nan3(qnan, qnan, qnan);
inline bool isNaN(cv::Point3f p)
{
    return (cvIsNaN(p.x) || cvIsNaN(p.y) || cvIsNaN(p.z));
}


template<typename TMat>
static TMat getTMat(InputArray, int = -1);

template<>
Mat getTMat<Mat>(InputArray a, int i)
{
	std::cout << "getTMat<Mat>" << std::endl;
	return a.getMat(i);
}

template<>
UMat getTMat<UMat>(InputArray a, int i)
{
	std::cout << "getTMat<UMat>" << std::endl;
	return a.getUMat(i);
}

template<typename TMat>
static TMat& getTMatRef(InputOutputArray, int = -1);

template<>
Mat& getTMatRef<Mat>(InputOutputArray a, int i)
{
    return a.getMatRef(i);
}

static const int sobelSize = 3;
static const double sobelScale = 1. / 8.;
static const int normalWinSize = 5;

template<typename TMat>
static
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
void preparePyramidCloud(InputArrayOfArrays pyramidDepth, const Matx33f& cameraMatrix, InputOutputArrayOfArrays pyramidCloud)
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
            depthTo3d(getTMat<TMat>(pyramidDepth, (int)i), pyramidCameraMatrix[i], cloud);
            getTMatRef<TMat>(pyramidCloud, (int)i) = cloud;
        }
    }
}

template<typename TMat>
static
void preparePyramidSobel(InputArrayOfArrays pyramidImage, int dx, int dy, InputOutputArrayOfArrays pyramidSobel)
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

static inline
void checkImage(InputArray image)
{
    if (image.empty())
        //CV_Error(Error::StsBadSize, "Image is empty.");
        std::cout << "Image is empty." << std::endl;
    if (image.type() != CV_8UC1)
        //CV_Error(Error::StsBadSize, "Image type has to be CV_8UC1.");
        std::cout << "Image type has to be CV_8UC1. " << image.type() << std::endl;
}

static inline
void checkDepth(InputArray depth, const Size& imageSize)
{
    if (depth.empty())
        CV_Error(Error::StsBadSize, "Depth is empty.");
    if (depth.size() != imageSize)
        CV_Error(Error::StsBadSize, "Depth has to have the size equal to the image size.");
    if (depth.type() != CV_32FC1)
        CV_Error(Error::StsBadSize, "Depth type has to be CV_32FC1.");
}

static inline
void checkMask(InputArray mask, const Size& imageSize)
{
    if (!mask.empty())
    {
        if (mask.size() != imageSize)
            CV_Error(Error::StsBadSize, "Mask has to have the size equal to the image size.");
        if (mask.type() != CV_8UC1)
            CV_Error(Error::StsBadSize, "Mask type has to be CV_8UC1.");
    }
}

static inline
void checkNormals(InputArray normals, const Size& depthSize)
{
    if (normals.size() != depthSize)
        CV_Error(Error::StsBadSize, "Normals has to have the size equal to the depth size.");
    if (normals.type() != CV_32FC3)
        CV_Error(Error::StsBadSize, "Normals type has to be CV_32FC3.");
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

static
void preparePyramidDepth(InputArray depth, InputOutputArrayOfArrays pyramidDepth, size_t levelCount)
{
    if (!pyramidDepth.empty())
    {
        size_t nLevels = pyramidDepth.size(-1).width;
        if (nLevels < levelCount)
            CV_Error(Error::StsBadSize, "Levels count of pyramidDepth has to be equal or less than size of iterCounts.");

        CV_Assert(pyramidDepth.size(0) == depth.size());
        for (size_t i = 0; i < nLevels; i++)
            CV_Assert(pyramidDepth.type((int)i) == depth.type());
    }
    else
        buildPyramid(depth, pyramidDepth, (int)levelCount - 1);
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

static
void preparePyramidTexturedMask(InputArrayOfArrays pyramid_dI_dx, InputArrayOfArrays pyramid_dI_dy,
    InputArray minGradMagnitudes, InputArrayOfArrays pyramidMask, double maxPointsPart,
    InputOutputArrayOfArrays pyramidTexturedMask)
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
        //CV_Assert(minGradMagnitudes.type() == CV_32F);
        if (minGradMagnitudes.type() != CV_32F)
            std::cout << "minGradMagnitudes.type() != CV_32F  " << minGradMagnitudes.type() << std::endl;
        Mat_<float> mgMags = minGradMagnitudes.getMat();

        const float sobelScale2_inv = 1.f / (float)(sobelScale * sobelScale);
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
void preparePyramidNormals(InputArray normals, InputArrayOfArrays pyramidDepth, InputOutputArrayOfArrays pyramidNormals)
{
    size_t depthLevels = pyramidDepth.size(-1).width;
    size_t normalsLevels = pyramidNormals.size(-1).width;
    if (!pyramidNormals.empty())
    {
        if (normalsLevels != depthLevels)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidNormals.");

        for (size_t i = 0; i < normalsLevels; i++)
        {
            CV_Assert(pyramidNormals.size((int)i) == pyramidDepth.size((int)i));
            CV_Assert(pyramidNormals.type((int)i) == CV_32FC3);
        }
    }
    else
    {
        buildPyramid(normals, pyramidNormals, (int)depthLevels - 1);
        // renormalize normals
        for (size_t i = 1; i < depthLevels; i++)
        {
            Mat& currNormals = pyramidNormals.getMatRef((int)i);
            for (int y = 0; y < currNormals.rows; y++)
            {
                Point3f* normals_row = currNormals.ptr<Point3f>(y);
                for (int x = 0; x < currNormals.cols; x++)
                {
                    double nrm = norm(normals_row[x]);
                    normals_row[x] *= 1. / nrm;
                }
            }
        }
    }
}

static
void preparePyramidNormalsMask(InputArray pyramidNormals, InputArray pyramidMask, double maxPointsPart,
    InputOutputArrayOfArrays /*std::vector<Mat>&*/ pyramidNormalsMask)
{
    size_t maskLevels = pyramidMask.size(-1).width;
    size_t norMaskLevels = pyramidNormalsMask.size(-1).width;
    if (!pyramidNormalsMask.empty())
    {
        if (norMaskLevels != maskLevels)
            CV_Error(Error::StsBadSize, "Incorrect size of pyramidNormalsMask.");

        for (size_t i = 0; i < norMaskLevels; i++)
        {
            CV_Assert(pyramidNormalsMask.size((int)i) == pyramidMask.size((int)i));
            CV_Assert(pyramidNormalsMask.type((int)i) == pyramidMask.type((int)i));
        }
    }
    else
    {
        pyramidNormalsMask.create((int)maskLevels, 1, CV_8U, -1);
        for (size_t i = 0; i < maskLevels; i++)
        {
            Mat& normalsMask = pyramidNormalsMask.getMatRef((int)i);
            normalsMask = pyramidMask.getMat((int)i).clone();

            const Mat normals = pyramidNormals.getMat((int)i);
            for (int y = 0; y < normalsMask.rows; y++)
            {
                const Vec3f* normals_row = normals.ptr<Vec3f>(y);
                uchar* normalsMask_row = normalsMask.ptr<uchar>(y);
                for (int x = 0; x < normalsMask.cols; x++)
                {
                    Vec3f n = normals_row[x];
                    if (cvIsNaN(n[0]))
                    {
                        CV_DbgAssert(cvIsNaN(n[1]) && cvIsNaN(n[2]));
                        normalsMask_row[x] = 0;
                    }
                }
            }
            randomSubsetOfMask(normalsMask, (float)maxPointsPart);
        }
    }
}
void pyrDownPointsNormals(const Points p, const Normals n, Points &pdown, Normals &ndown)
{
    for(int y = 0; y < pdown.rows; y++)
    {
        ptype* ptsRow = pdown[y];
        ptype* nrmRow = ndown[y];
        const ptype* pUpRow0 = p[2*y];
        const ptype* pUpRow1 = p[2*y+1];
        const ptype* nUpRow0 = n[2*y];
        const ptype* nUpRow1 = n[2*y+1];
        for(int x = 0; x < pdown.cols; x++)
        {
            Point3f point = nan3, normal = nan3;

            Point3f d00 = fromPtype(pUpRow0[2*x]);
            Point3f d01 = fromPtype(pUpRow0[2*x+1]);
            Point3f d10 = fromPtype(pUpRow1[2*x]);
            Point3f d11 = fromPtype(pUpRow1[2*x+1]);

            if(!(isNaN(d00) || isNaN(d01) || isNaN(d10) || isNaN(d11)))
            {
                point = (d00 + d01 + d10 + d11)*0.25f;

                Point3f n00 = fromPtype(nUpRow0[2*x]);
                Point3f n01 = fromPtype(nUpRow0[2*x+1]);
                Point3f n10 = fromPtype(nUpRow1[2*x]);
                Point3f n11 = fromPtype(nUpRow1[2*x+1]);

                normal = (n00 + n01 + n10 + n11)*0.25f;
            }

            ptsRow[x] = toPtype(point);
            nrmRow[x] = toPtype(normal);
        }
    }
}
void buildPyramidPointsNormals(InputArray _points, InputArray _normals,
    OutputArrayOfArrays pyrPoints, OutputArrayOfArrays pyrNormals,
    int levels)
{
    //CV_Assert(_points.type() == POINT_TYPE);
    //CV_Assert(_points.type() == _normals.type());
    //CV_Assert(_points.size() == _normals.size());

    int kp = pyrPoints.kind(), kn = pyrNormals.kind();
    CV_Assert(kp == _InputArray::STD_ARRAY_MAT || kp == _InputArray::STD_VECTOR_MAT);
    CV_Assert(kn == _InputArray::STD_ARRAY_MAT || kn == _InputArray::STD_VECTOR_MAT);

    Mat p0 = _points.getMat(), n0 = _normals.getMat();

    pyrPoints.create(levels, 1, POINT_TYPE);
    pyrNormals.create(levels, 1, POINT_TYPE);

    pyrPoints.getMatRef(0) = p0;
    pyrNormals.getMatRef(0) = n0;

    Size sz = _points.size();
    for (int i = 1; i < levels; i++)
    {
        Points  p1 = pyrPoints.getMat(i - 1);
        Normals n1 = pyrNormals.getMat(i - 1);

        sz.width /= 2; sz.height /= 2;

        pyrPoints.create(sz, POINT_TYPE, i);
        pyrNormals.create(sz, POINT_TYPE, i);
        Points  pd = pyrPoints.getMatRef(i);
        Normals nd = pyrNormals.getMatRef(i);

        pyrDownPointsNormals(p1, n1, pd, nd);
    }
}



#endif //ODOMETRY_FUNCTIONS_HPP