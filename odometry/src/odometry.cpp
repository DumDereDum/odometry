#include "odometry_icp.hpp"
#include "odometry_rgb.hpp"
#include "odometry_rgbd.hpp"


Odometry::Odometry(OdometryType otype, OdometrySettings settings)
{
	switch (otype)
	{
	case OdometryType::ICP:
		this->odometry = makePtr<OdometryICP>(settings);
		break;
	case OdometryType::RGB:
		this->odometry = makePtr<OdometryRGB>(settings);
		break;
	case OdometryType::RGBD:
		this->odometry = makePtr<OdometryRGBD>(settings);
		break;
	default:
		//CV_Error(Error::StsInternal,
		//	"Incorrect OdometryType, you are able to use only { ICP, RGB, RGBD }");
		break;
	}
	
}

Odometry::~Odometry()
{
}

bool Odometry::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt)
{
	this->odometry->compute(srcFrame, dstFrame, Rt);
	return true;
}

bool Odometry::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
	this->odometry->prepareFrames(srcFrame, dstFrame);
	return true;
}
