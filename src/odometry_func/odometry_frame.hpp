#ifndef ODOMETRY_FRAME
#define ODOMETRY_FRAME

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/mat.hpp>

#include "odometry_frame_impl.hpp"

using namespace cv;

class OdometryFrame
{
public:
	OdometryFrame() {};
	~OdometryFrame() {};
	void create(InputArray image);
private:
	Ptr<OdometryFrameImpl> odometryFrame;
};

void OdometryFrame::create(InputArray _image)
{
	//bool allEmpty = _image.empty();
	//bool useOcl = _image.isUMat();
	//if (useOcl && !allEmpty)
	//	this->odometryFrame = makePtr<OdometryFrameImplTMat<UMat>>(_image);
	//else
	//	this->odometryFrame = makePtr<OdometryFrameImplTMat<Mat>>(_image);
}



#endif // !ODOMETRY_FRAME