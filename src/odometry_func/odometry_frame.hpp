#ifndef ODOMETRY_FRAME_HPP
#define ODOMETRY_FRAME_HPP

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/mat.hpp>

//#include "../odometry.hpp"
#include "odometry_frame_impl.hpp"

using namespace cv;

enum class OdometryType
{
	ICP = 0,
	RGB = 1,
	RGBD = 2
};

/** Indicates what pyramid is to access using get/setPyramid... methods:
* @param PYR_IMAGE The pyramid of RGB images
* @param PYR_DEPTH The pyramid of depth images
* @param PYR_MASK  The pyramid of masks
* @param PYR_CLOUD The pyramid of point clouds, produced from the pyramid of depths
* @param PYR_DIX   The pyramid of dI/dx derivative images
* @param PYR_DIY   The pyramid of dI/dy derivative images
* @param PYR_TEXMASK The pyramid of textured masks
* @param PYR_NORM  The pyramid of normals
* @param PYR_NORMMASK The pyramid of normals masks
*/

/*
enum OdometryFramePyramidType
{
	PYR_IMAGE = 0, PYR_DEPTH = 1, PYR_MASK = 2, PYR_CLOUD = 3, PYR_DIX = 4, PYR_DIY = 5, PYR_TEXMASK = 6, PYR_NORM = 7, PYR_NORMMASK = 8,
	N_PYRAMIDS
};
*/

class OdometryFrame
{
private:
	Ptr<OdometryFrameImpl> odometryFrame;
public:
	OdometryFrame(InputArray image,
				  OdometryType otype = OdometryType::ICP)
	{
		bool allEmpty = image.empty();
		bool useOcl   = image.isUMat();
		bool isNoArray  = (&image == &noArray());
		//if (useOcl && !allEmpty && !isNoArray)
		std::cout << " allEmpty:" << allEmpty<<" useOcl:" << useOcl << " isNoArray:" << isNoArray << std::endl;
		if (useOcl)
			this->odometryFrame = makePtr<OdometryFrameImplTMat<UMat>>(image);
		else
			this->odometryFrame = makePtr<OdometryFrameImplTMat<Mat>>(image);
	};
	~OdometryFrame() {};
	void setImage(InputArray  image) { this->odometryFrame->setImage(image); }
	void getImage(OutputArray image) { this->odometryFrame->getImage(image); }
	void setDepth(InputArray  depth) { this->odometryFrame->setDepth(depth); }
	void getDepth(OutputArray depth) { this->odometryFrame->getDepth(depth); }
	void setMask(InputArray  mask) { this->odometryFrame->setMask(mask); }
	void getMask(OutputArray mask) { this->odometryFrame->getMask(mask); }
	void setNormals(InputArray  normals) { this->odometryFrame->setNormals(normals); }
	void getNormals(OutputArray normals) { this->odometryFrame->getNormals(normals); }
	void setPyramidAt(InputArray  img, OdometryFramePyramidType pyrType, size_t level) {};
	void getPyramidAt(OutputArray img, OdometryFramePyramidType pyrType, size_t level) {};



	void findMask(InputArray depth) { this->odometryFrame->findMask(depth); }

	void prepareAsSrcFrame() {};
	void prepareAsDstFrame() {};
};

#endif // !ODOMETRY_FRAME_HPP