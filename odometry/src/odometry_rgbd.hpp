#ifndef ODOMETRY_RGBD_HPP
#define ODOMETRY_RGBD_HPP

#include "../include/odometry.hpp"

class OdometryRGBD : public OdometryImpl
{
private:
	OdometrySettings settings;

	virtual bool compute_corresps() const override;
	virtual bool compute_Rt(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const override;
public:
	OdometryRGBD(OdometrySettings settings);
	~OdometryRGBD();

	virtual OdometryFrame createOdometryFrame() override;
	virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const override;
	virtual bool prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame);
};

#endif //ODOMETRY_RGBD_HPP