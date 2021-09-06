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


#endif //ODOMETRY_FUNCTIONS_HPP