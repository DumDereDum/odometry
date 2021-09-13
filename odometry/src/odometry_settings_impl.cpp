#include "odometry_settings.hpp"

const cv::Matx33f defaultCameraMatrix =
{ /* fx, 0, cx*/ 0, 0, 0,
  /* 0, fy, cy*/ 0, 0, 0,
  /* 0,  0,  0*/ 0, 0, 0 };
const std::vector<int> defaultIterCounts = { 7, 7, 7, 10 };

const float defaultMinDepth = 0.f;
const float defaultMaxDepth = 4.f;
const float defaultMaxDepthDiff = 0.07f;
const float defaultMaxPointsPart = 0.07f;

static const int defaultSobelSize = 3;
static const double defaultSobelScale = 1. / 8.;
static const int defaultNormalWinSize = 5;

const float defaultMaxTranslation = 0.15f;
const float defaultMaxRotation = 15.f;

const float defaultMinGradientMagnitude = 10.f;
std::vector<float> defaultMinGradientMagnitudes = std::vector<float>(defaultIterCounts.size(), defaultMinGradientMagnitude);


class OdometrySettingsImplCommon : public OdometrySettingsImpl
{
public:
    OdometrySettingsImplCommon() {};
    ~OdometrySettingsImplCommon() {};
	virtual void setCameraMatrix(InputArray val);
	virtual void getCameraMatrix(OutputArray val) const;
	virtual void setIterCounts(InputArrayOfArrays val);
	virtual void getIterCounts(OutputArrayOfArrays val) const;

	virtual void  setMinDepth(float val);
	virtual float getMinDepth() const;
	virtual void  setMaxDepth(float val);
	virtual float getMaxDepth() const;
	virtual void  setMaxDepthDiff(float val);
	virtual float getMaxDepthDiff() const;
	virtual void  setMaxPointsPart(float val);
	virtual float getMaxPointsPart() const;

	virtual void setSobelSize(int val);
	virtual int  getSobelSize() const;
	virtual void   setSobelScale(double val);
	virtual double getSobelScale() const;
	virtual void setNormalWinSize(int val);
	virtual int  getNormalWinSize() const;

	virtual void  setMaxTranslation(float val);
	virtual float getMaxTranslation() const;
	virtual void  setMaxRotation(float val);
	virtual float getMaxRotation() const;

	virtual void  setMinGradientMagnitude(float val);
	virtual float getMinGradientMagnitude() const;
	virtual void setMinGradientMagnitudes(InputArrayOfArrays val);
	virtual void getMinGradientMagnitudes(OutputArrayOfArrays val) const;

private:
    Matx33f cameraMatrix;
};

OdometrySettings::OdometrySettings()
{
    this->odometrySettings = makePtr<OdometrySettingsImplCommon>();
}

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

void OdometrySettingsImplCommon::setIterCounts(InputArrayOfArrays val)
{

}

void OdometrySettingsImplCommon::getIterCounts(OutputArrayOfArrays val) const
{

}

void  OdometrySettingsImplCommon::setMinDepth(float val)
{

}
float OdometrySettingsImplCommon::getMinDepth() const
{
	return 1.f;
}
void  OdometrySettingsImplCommon::setMaxDepth(float val)
{

}
float OdometrySettingsImplCommon::getMaxDepth() const
{
	return 1.f;
}
void  OdometrySettingsImplCommon::setMaxDepthDiff(float val)
{

}
float OdometrySettingsImplCommon::getMaxDepthDiff() const
{
	return 1.f;
}
void  OdometrySettingsImplCommon::setMaxPointsPart(float val)
{

}
float OdometrySettingsImplCommon::getMaxPointsPart() const
{
	return 1.f;
}

void OdometrySettingsImplCommon::setSobelSize(int val)
{

}
int  OdometrySettingsImplCommon::getSobelSize() const
{
	return 1;
}
void   OdometrySettingsImplCommon::setSobelScale(double val)
{

}
double OdometrySettingsImplCommon::getSobelScale() const
{
	return 1.;
}
void OdometrySettingsImplCommon::setNormalWinSize(int val)
{

}
int  OdometrySettingsImplCommon::getNormalWinSize() const
{
	return 1;
}

void  OdometrySettingsImplCommon::setMaxTranslation(float val)
{

}
float OdometrySettingsImplCommon::getMaxTranslation() const
{
	return 1.f;
}
void  OdometrySettingsImplCommon::setMaxRotation(float val)
{

}
float OdometrySettingsImplCommon::getMaxRotation() const
{
	return 1.f;
}

void  OdometrySettingsImplCommon::setMinGradientMagnitude(float val)
{

}
float OdometrySettingsImplCommon::getMinGradientMagnitude() const
{
	return 1.f;
}
void OdometrySettingsImplCommon::setMinGradientMagnitudes(InputArrayOfArrays val)
{

}
void OdometrySettingsImplCommon::getMinGradientMagnitudes(OutputArrayOfArrays val) const
{

}
