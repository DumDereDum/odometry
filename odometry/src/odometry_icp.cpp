#include "odometry_icp.hpp"
#include "odometry_functions.hpp"

OdometryICP::OdometryICP(OdometrySettings _settings)
{
	this->settings = _settings;
}

OdometryICP::~OdometryICP()
{
}

OdometryFrame OdometryICP::createOdometryFrame()
{
	std::cout << "OdometryICP::createOdometryFrame()" << std::endl;
	return OdometryFrame(Mat());
}

bool OdometryICP::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
{
	std::cout << "OdometryICP::compute()" << std::endl;
	this->compute_corresps();
	this->compute_Rt();
	return true;
}

bool OdometryICP::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
	std::cout << "OdometryICP::prepareFrames()" << std::endl;
	//prepareICPFrame(srcFrame, dstFrame);
	return true;
}

bool OdometryICP::compute_corresps() const
{
	std::cout << "OdometryICP::compute_corresps()" << std::endl;
	return true;
}

bool OdometryICP::compute_Rt() const
{
	std::cout << "OdometryICP::compute_Rt()" << std::endl;
	return true;
}
