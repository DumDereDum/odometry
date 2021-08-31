#include "odometry_settings.hpp"

void OdometrySettings::setCameraMatrix(InputArray val)
{
    if (!val.empty())
    {
        CV_Assert(val.rows() == 3 && val.cols() == 3 && val.channels() == 1);
        CV_Assert(val.type() == CV_32F);
        val.copyTo(cameraMatrix);
    }
}

void OdometrySettings::getCameraMatrix(OutputArray val) const
{
    Mat(this->cameraMatrix).copyTo(val);
}