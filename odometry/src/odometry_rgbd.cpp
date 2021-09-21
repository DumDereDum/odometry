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

bool OdometryRGBD::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
{
	//std::cout << "OdometryRGBD::compute()" << std::endl;
	this->compute_corresps();
	this->compute_Rt(srcFrame, dstFrame, Rt);
	return true;
}

bool OdometryRGBD::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
	//std::cout << "OdometryRGBD::prepareFrames()" << std::endl;
	prepareRGBDFrame(srcFrame, dstFrame, this->settings);
	return true;
}

bool OdometryRGBD::compute_corresps() const
{
	//std::cout << "OdometryRGBD::compute_corresps()" << std::endl;
	return true;
}

bool OdometryRGBD::compute_Rt(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
{
	//std::cout << "OdometryRGBD::compute_Rt()" << std::endl;
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
		OdometryType::RGBD, OdometryTransformType::RIGID_TRANSFORMATION);
	return true;
}