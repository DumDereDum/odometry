#include "odometry_rgbd.hpp"
#include "odometry_functions.hpp"

OdometryRGBD::OdometryRGBD()
{
}

OdometryRGBD::~OdometryRGBD()
{
}

OdometryFrame OdometryRGBD::createOdometryFrame()
{
	std::cout << "OdometryRGBD::createOdometryFrame()" << std::endl;
	return OdometryFrame(Mat());
}

bool OdometryRGBD::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
{
	std::cout << "OdometryRGBD::compute()" << std::endl;
	this->compute_corresps();
	this->compute_Rt();
	return true;
}

bool OdometryRGBD::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
	std::cout << "OdometryRGBD::prepareFrames()" << std::endl;
	return true;
}

bool OdometryRGBD::compute_corresps() const
{
	std::cout << "OdometryRGBD::compute_corresps()" << std::endl;
	return true;
}

bool OdometryRGBD::compute_Rt() const
{
	std::cout << "OdometryRGBD::compute_Rt()" << std::endl;
	return true;
}