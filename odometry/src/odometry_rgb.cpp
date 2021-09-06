#include "odometry_rgb.hpp"

OdometryRGB::OdometryRGB()
{
}

OdometryRGB::~OdometryRGB()
{
}

bool OdometryRGB::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
{
	std::cout << "OdometryRGB::compute()" << std::endl;
	this->compute_corresps();
	this->compute_Rt();
	return true;
}

bool OdometryRGB::compute_corresps() const
{
	std::cout << "OdometryRGB::compute_corresps()" << std::endl;
	return true;
}

bool OdometryRGB::compute_Rt() const
{
	std::cout << "OdometryRGB::compute_Rt()" << std::endl;
	return true;
}