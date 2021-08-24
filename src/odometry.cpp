#include "odometry.hpp"

Odometry::Odometry(OdometryType otype)
{
	switch (otype)
	{
	case OdometryType::ICP:
		this->odometry = makePtr<OdometryICP>();
		break;
	case OdometryType::RGB:
		this->odometry = makePtr<OdometryRGB>();
		break;
	case OdometryType::RGBD:
		this->odometry = makePtr<OdometryRGBD>();
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

bool Odometry::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, Affine3f& Rt)
{
	this->odometry->compute(srcFrame, dstFrame, Rt);
	return true;
}