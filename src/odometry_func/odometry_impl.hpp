#ifndef ODOMETRY_IMPL_HPP
#define ODOMETRY_IMPL_HPP

#include <iostream>

class OdometryImpl
{
private:
    virtual bool compute_corresps() const = 0;
    virtual bool compute_Rt() const = 0;
public:
    OdometryImpl();
    ~OdometryImpl();

    virtual bool compute() const = 0;
};

#endif //ODOMETRY_IMPL_HPP