#ifndef ODOMETRY_ICP_HPP
#define ODOMETRY_ICP_HPP

#include "odometry_impl.hpp"

class OdometryICP : public OdometryImpl
{
private:
    virtual bool compute_corresps() const override;
    virtual bool compute_Rt() const override;
public:
    OdometryICP();
    ~OdometryICP();

    virtual bool compute(OdometryFrame frame) const override;
};

#endif //ODOMETRY_ICP_HPP