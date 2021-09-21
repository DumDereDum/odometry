#include "odometry_icp.hpp"
#include "odometry_functions.hpp"

OdometryICP::OdometryICP(OdometrySettings _settings)
{
	this->settings = _settings;
}

OdometryICP::~OdometryICP()
{
}

OdometryFrame OdometryICP::createOdometryFrame()
{
	//std::cout << "OdometryICP::createOdometryFrame()" << std::endl;
	return OdometryFrame(Mat());
}

bool OdometryICP::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
{
	//std::cout << "OdometryICP::compute()" << std::endl;
	this->compute_corresps();
	this->compute_Rt(srcFrame, dstFrame, Rt);
	return true;
}

bool OdometryICP::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
	//std::cout << "OdometryICP::prepareFrames()" << std::endl;
	prepareICPFrame(srcFrame, dstFrame, this->settings);
	return true;
}

bool OdometryICP::compute_corresps() const
{
	//std::cout << "OdometryICP::compute_corresps()" << std::endl;
	return true;
}

bool OdometryICP::compute_Rt(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
{
	//std::cout << "OdometryICP::compute_Rt()" << std::endl;
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
		OdometryType::ICP, OdometryTransformType::RIGID_TRANSFORMATION);
	return true;
}
