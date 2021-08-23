#ifndef ODOMETRY_FRAME_HPP
#define ODOMETRY_FRAME_HPP

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
	void create(InputArray image)
	{
		bool allEmpty = image.empty();
		bool useOcl = image.isUMat();
		//if (useOcl && !allEmpty)
		if (useOcl)
			this->odometryFrame = makePtr<OdometryFrameImplTMat<UMat>>(image);
		else
			this->odometryFrame = makePtr<OdometryFrameImplTMat<Mat>>(image);
	}
private:
	Ptr<OdometryFrameImpl> odometryFrame;
};

#endif // !ODOMETRY_FRAME_HPP