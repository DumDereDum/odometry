#ifndef ODOMETRY_RGB_HPP
#define ODOMETRY_RGB_HPP

#include "../include/odometry.hpp"

class OdometryRGB : public OdometryImpl
{
private:
	OdometrySettings settings;

	virtual bool compute_corresps() const override;
	virtual bool compute_Rt() const override;
public:
	OdometryRGB(OdometrySettings settings);
	~OdometryRGB();

	virtual OdometryFrame createOdometryFrame() override;
	virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const override;
	virtual bool prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame);
};

#endif //ODOMETRY_RGB_HPP