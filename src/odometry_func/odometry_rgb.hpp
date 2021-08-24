#ifndef ODOMETRY_RGB_HPP
#define ODOMETRY_RGB_HPP

#include "odometry_impl.hpp"

class OdometryRGB : public OdometryImpl
{
private:
    virtual bool compute_corresps() const override;
    virtual bool compute_Rt() const override;
public:
    OdometryRGB();
    ~OdometryRGB();

    virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, Affine3f& Rt) const override;
};

#endif //ODOMETRY_RGB_HPP