#include "odometry.hpp"

Odometry::Odometry(OdometryType otype)
{
	switch (otype)
	{
	case OdometryType::ICP:
		this->odometry = new OdometryICP();
		break;
	case OdometryType::RGB:
		this->odometry = new OdometryRGB();
		break;
	case OdometryType::RGBD:
		this->odometry = new OdometryRGBD();
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

bool Odometry::compute()
{
	this->odometry->compute();
	return true;
}