#ifndef ODOMETRY_FRAME_HPP
#define ODOMETRY_FRAME_HPP

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/mat.hpp>

//#include "../odometry.hpp"
#include "odometry_frame_impl.hpp"

using namespace cv;

enum class OdometryType
{
	ICP = 0,
	RGB = 1,
	RGBD = 2
};

class OdometryFrame
{
public:
	OdometryFrame(InputArray image = noArray(),
				  OdometryType otype = OdometryType::ICP)
	{
		bool allEmpty = image.empty();
		bool useOcl   = image.isUMat();
		bool isNoArray  = (&image == &noArray());
		//if (useOcl && !allEmpty && !isNoArray)
		if (useOcl && !isNoArray)
			this->odometryFrame = makePtr<OdometryFrameImplTMat<UMat>>(image);
		else
			this->odometryFrame = makePtr<OdometryFrameImplTMat<Mat>>(image);
	};
	~OdometryFrame() {};
private:
	Ptr<OdometryFrameImpl> odometryFrame;
};

#endif // !ODOMETRY_FRAME_HPP