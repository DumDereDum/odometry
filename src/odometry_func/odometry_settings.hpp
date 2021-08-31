#ifndef ODOMETRY_SETTINGS_HPP
#define ODOMETRY_SETTINGS_HPP

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/mat.hpp>

using namespace cv;

const cv::Matx33f defaultCameraMatrix = { /* fx, 0, cx*/ 0, 0, 0,
										  /* 0, fy, cy*/ 0, 0, 0,
										  /* 0,  0,  0*/ 0, 0, 0 };

class OdometrySettings
{
public:
	OdometrySettings() {};
	~OdometrySettings() {};
	void setCameraMatrix(InputArray val);
	void getCameraMatrix(OutputArray val) const;
private:
	Matx33f cameraMatrix;
};


#endif //ODOMETRY_SETTINGS_HPP