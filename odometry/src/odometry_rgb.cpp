#include "odometry_functions.hpp"
#include "odometry_rgb.hpp"

OdometryRGB::OdometryRGB(OdometrySettings _settings)
{
	this->settings = _settings;
}

OdometryRGB::~OdometryRGB()
{
}

OdometryFrame OdometryRGB::createOdometryFrame()
{
	//std::cout << "OdometryRGB::createOdometryFrame()" << std::endl;
	return OdometryFrame(Mat());
}

bool OdometryRGB::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
{
	//std::cout << "OdometryRGB::compute()" << std::endl;
	this->compute_corresps();
	this->compute_Rt(srcFrame, dstFrame, Rt);
	return true;
}

bool OdometryRGB::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
	//std::cout << "OdometryRGB::prepareFrames()" << std::endl;
	prepareRGBFrame(srcFrame, dstFrame, this->settings);
	return true;
}

bool OdometryRGB::compute_corresps() const
{
	//std::cout << "OdometryRGB::compute_corresps()" << std::endl;
	return true;
}

bool OdometryRGB::compute_Rt(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
{
	//std::cout << "OdometryRGB::compute_Rt()" << std::endl;
	Matx33f cameraMatrix;
	settings.getCameraMatrix(cameraMatrix);
	std::vector<int> iterCounts;
	Mat miterCounts;
	settings.getIterCounts(miterCounts);
	for (int i = 0; i < miterCounts.size().height; i++)
		iterCounts.push_back(miterCounts.at<int>(i));
	RGBDICPOdometryImpl(Rt, Mat(), srcFrame, dstFrame, cameraMatrix,
		this->settings.getMaxDepthDiff(), iterCounts, this->settings.getMaxTranslation(),
		this->settings.getMaxRotation(), settings.getSobelScale(),
		OdometryType::RGB, OdometryTransformType::RIGID_TRANSFORMATION);
	return true;
}