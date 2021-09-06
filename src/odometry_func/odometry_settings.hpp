#ifndef ODOMETRY_SETTINGS_HPP
#define ODOMETRY_SETTINGS_HPP

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/mat.hpp>

using namespace cv;

class OdometrySettingsImpl
{
public:
	OdometrySettingsImpl() {};
	~OdometrySettingsImpl() {};
	virtual void setCameraMatrix(InputArray val) = 0;
	virtual void getCameraMatrix(OutputArray val) const = 0;
private:
	Matx33f cameraMatrix;
};

class OdometrySettings
{
public:
	OdometrySettings() {};
	~OdometrySettings() {};
	void setCameraMatrix(InputArray val) { this->odometrySettings->setCameraMatrix(val); }
	void getCameraMatrix(OutputArray val) const { this->odometrySettings->getCameraMatrix(val); }
private:
	Ptr<OdometrySettingsImpl> odometrySettings;
};


#endif //ODOMETRY_SETTINGS_HPP