#include "odometry_func/odometry_icp.hpp"
#include "odometry_func/odometry_rgb.hpp"
#include "odometry_func/odometry_rgbd.hpp"


Odometry::Odometry(OdometryType otype, OdometrySettings settings)
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

bool Odometry::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt)
{
	this->odometry->compute(srcFrame, dstFrame, Rt);
	return true;
}
