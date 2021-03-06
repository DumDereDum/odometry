#include "odometry_rgbd.hpp"
#include "odometry_functions.hpp"

OdometryRGBD::OdometryRGBD(OdometrySettings _settings)
{
	this->settings = _settings;
}

OdometryRGBD::~OdometryRGBD()
{
}

OdometryFrame OdometryRGBD::createOdometryFrame()
{
	//std::cout << "OdometryRGBD::createOdometryFrame()" << std::endl;
	return OdometryFrame(Mat());
}

bool OdometryRGBD::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
	//std::cout << "OdometryRGBD::prepareFrames()" << std::endl;
	prepareRGBDFrame(srcFrame, dstFrame, this->settings);
	return true;
}

bool OdometryRGBD::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt, OdometryAlgoType algtype) const
{
	//std::cout << "OdometryRGBD::compute()" << std::endl;
	Matx33f cameraMatrix;
	settings.getCameraMatrix(cameraMatrix);
	std::vector<int> iterCounts;
	Mat miterCounts;
	settings.getIterCounts(miterCounts);
	for (int i = 0; i < miterCounts.size().height; i++)
		iterCounts.push_back(miterCounts.at<int>(i));
	RGBDICPOdometryImpl(Rt, Mat(), srcFrame, dstFrame, cameraMatrix,
		this->settings.getMaxDepthDiff(), this->settings.getAngleThreshold(),
		iterCounts, this->settings.getMaxTranslation(),
		this->settings.getMaxRotation(), settings.getSobelScale(),
		OdometryType::RGBD, OdometryTransformType::RIGID_TRANSFORMATION, algtype);
	return true;
}
