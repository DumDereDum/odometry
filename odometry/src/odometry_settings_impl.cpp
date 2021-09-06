#include "odometry_settings.hpp"

class OdometrySettingsImplCommon : public OdometrySettingsImpl
{
public:
    OdometrySettingsImplCommon() {};
    ~OdometrySettingsImplCommon() {};
    virtual void setCameraMatrix(InputArray val);
    virtual void getCameraMatrix(OutputArray val) const;
private:
    Matx33f cameraMatrix;
};

OdometrySettings::OdometrySettings()
{
    this->odometrySettings = makePtr<OdometrySettingsImplCommon>();
}

const cv::Matx33f defaultCameraMatrix =
{ /* fx, 0, cx*/ 0, 0, 0,
  /* 0, fy, cy*/ 0, 0, 0,
  /* 0,  0,  0*/ 0, 0, 0 };


void OdometrySettingsImplCommon::setCameraMatrix(InputArray val)
{
    if (!val.empty())
    {
        CV_Assert(val.rows() == 3 && val.cols() == 3 && val.channels() == 1);
        CV_Assert(val.type() == CV_32F);
        val.copyTo(cameraMatrix);
    }
}

void OdometrySettingsImplCommon::getCameraMatrix(OutputArray val) const
{
    Mat(this->cameraMatrix).copyTo(val);
}
