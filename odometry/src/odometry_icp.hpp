#ifndef ODOMETRY_ICP_HPP
#define ODOMETRY_ICP_HPP

#include "../include/odometry.hpp"

class OdometryICP : public OdometryImpl
{
private:
	virtual bool compute_corresps() const override;
	virtual bool compute_Rt() const override;
public:
	OdometryICP();
	~OdometryICP();

	virtual OdometryFrame createOdometryFrame() override;
	virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const override;
	virtual bool prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame) override;
};

#endif //ODOMETRY_ICP_HPP