#ifndef ODOMETRY_FRAME_IMPL
#define ODOMETRY_FRAME_IMPL

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/mat.hpp>

using namespace cv;

class OdometryFrameImpl
{
public:
	OdometryFrameImpl() {};
	~OdometryFrameImpl() {};
	virtual void setX(InputArray image) = 0;

private:

};


#endif // !ODOMETRY_FRAME_IMPL