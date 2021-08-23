#ifndef ODOMETRY_IMPL_HPP
#define ODOMETRY_IMPL_HPP

#include <iostream>

#include "odometry_frame.hpp"

class OdometryImpl
{
private:
    virtual bool compute_corresps() const = 0;
    virtual bool compute_Rt() const = 0;
public:
    OdometryImpl();
    ~OdometryImpl();

    virtual bool compute(OdometryFrame frame) const = 0;
};

#endif //ODOMETRY_IMPL_HPP