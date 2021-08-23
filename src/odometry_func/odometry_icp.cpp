#include "odometry_icp.hpp"

OdometryICP::OdometryICP(/* args */)
{
}

OdometryICP::~OdometryICP()
{
}

bool OdometryICP::compute(OdometryFrame frame) const
{
	std::cout << "OdometryICP::compute()" << std::endl;
	this->compute_corresps();
	this->compute_Rt();
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
