#ifndef ODOMETRY_RGBD_HPP
#define ODOMETRY_RGBD_HPP

#include "../include/odometry.hpp"

class OdometryRGBD : public OdometryImpl
{
private:
	virtual bool compute_corresps() const override;
	virtual bool compute_Rt() const override;
public:
	OdometryRGBD();
	~OdometryRGBD();

	virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const override;
	virtual bool prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame);
};

#endif //ODOMETRY_RGBD_HPP