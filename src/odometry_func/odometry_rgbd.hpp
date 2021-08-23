#ifndef ODOMETRY_RGBD_HPP
#define ODOMETRY_RGBD_HPP

#include "odometry_impl.hpp"

class OdometryRGBD : public OdometryImpl
{
private:
    virtual bool compute_corresps() const override;
    virtual bool compute_Rt() const override;
public:
    OdometryRGBD();
    ~OdometryRGBD();

    virtual bool compute(OdometryFrame frame) const override;
};

#endif //ODOMETRY_RGBD_HPP