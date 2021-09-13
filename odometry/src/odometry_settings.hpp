#ifndef ODOMETRY_SETTINGS_HPP
#define ODOMETRY_SETTINGS_HPP

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cvstd.hpp>

using namespace cv;

class OdometrySettingsImpl
{
public:
	OdometrySettingsImpl() {};
	~OdometrySettingsImpl() {};
	virtual void setCameraMatrix(InputArray val) = 0;
	virtual void getCameraMatrix(OutputArray val) const = 0;
	virtual void setIterCounts(InputArrayOfArrays val) = 0;
	virtual void getIterCounts(OutputArrayOfArrays val) const = 0;
	
	virtual void  setMinDepth(float val) = 0;
	virtual float getMinDepth() const = 0;
	virtual void  setMaxDepth(float val) = 0;
	virtual float getMaxDepth() const = 0;
	virtual void  setMaxDepthDiff(float val) = 0;
	virtual float getMaxDepthDiff() const = 0;
	virtual void  setMaxPointsPart(float val) = 0;
	virtual float getMaxPointsPart() const = 0;

	virtual void setSobelSize(int val) = 0;
	virtual int  getSobelSize() const = 0;
	virtual void   setSobelScale(double val) = 0;
	virtual double getSobelScale() const = 0;
	virtual void setNormalWinSize(int val) = 0;
	virtual int  getNormalWinSize() const = 0;

	virtual void  setMaxTranslation(float val) = 0;
	virtual float getMaxTranslation() const = 0;
	virtual void  setMaxRotation(float val) = 0;
	virtual float getMaxRotation() const = 0;

	virtual void  setMinGradientMagnitude(float val) = 0;
	virtual float getMinGradientMagnitude() const = 0;
	virtual void setMinGradientMagnitudes(InputArrayOfArrays val) = 0;
	virtual void getMinGradientMagnitudes(OutputArrayOfArrays val) const = 0;

private:
	Matx33f cameraMatrix;
};

class OdometrySettings
{
public:
	OdometrySettings();
	~OdometrySettings() {};
	void setCameraMatrix(InputArray val) { this->odometrySettings->setCameraMatrix(val); }
	void getCameraMatrix(OutputArray val) const { this->odometrySettings->getCameraMatrix(val); }
private:
	Ptr<OdometrySettingsImpl> odometrySettings;
};


#endif //ODOMETRY_SETTINGS_HPP