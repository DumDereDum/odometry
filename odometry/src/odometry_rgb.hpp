#ifndef ODOMETRY_RGB_HPP
#define ODOMETRY_RGB_HPP

#include "../include/odometry.hpp"

class OdometryRGB : public OdometryImpl
{
private:
	virtual bool compute_corresps() const override;
	virtual bool compute_Rt() const override;
public:
	OdometryRGB();
	~OdometryRGB();

	virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const override;
};

#endif //ODOMETRY_RGB_HPP