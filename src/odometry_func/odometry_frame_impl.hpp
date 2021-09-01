#ifndef ODOMETRY_FRAME_IMPL_HPP
#define ODOMETRY_FRAME_IMPL_HPP

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/mat.hpp>

#include "odometry_functions.hpp"

using namespace cv;

class OdometryFrameImpl
{
public:
	OdometryFrameImpl() {};
	~OdometryFrameImpl() {};
	virtual void setImage(InputArray  image) = 0;
	virtual void getImage(OutputArray image) = 0;
	virtual void setDepth(InputArray  depth) = 0;
	virtual void getDepth(OutputArray depth) = 0;
	virtual void setMask(InputArray  mask) = 0;
	virtual void getMask(OutputArray mask) = 0;
	virtual void setNormals(InputArray  normals) = 0;
	virtual void getNormals(OutputArray normals) = 0;

	virtual void findMask(InputArray image) = 0;

private:

};

template<typename TMat>
class OdometryFrameImplTMat : public OdometryFrameImpl
{
public:
	OdometryFrameImplTMat(InputArray image = noArray());
	~OdometryFrameImplTMat() {};

	virtual void setImage(InputArray  image) override;
	virtual void getImage(OutputArray image) override;
	virtual void setDepth(InputArray  depth) override;
	virtual void getDepth(OutputArray depth) override;
	virtual void setMask(InputArray  mask) override;
	virtual void getMask(OutputArray mask) override;
	virtual void setNormals(InputArray  normals) override;
	virtual void getNormals(OutputArray normals) override;

	virtual void findMask(InputArray image) override;

private:
	TMat image;
	TMat depth;
	TMat mask;
	TMat normals;
};

template<typename TMat>
OdometryFrameImplTMat<TMat>::OdometryFrameImplTMat(InputArray _image)
{
	image = getTMat<TMat>(_image);
};

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setImage(InputArray _image)
{
	this->image = getTMat<TMat>(_image);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::getImage(OutputArray _image)
{
	_image.assign(this->image);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setDepth(InputArray _depth)
{
	this->depth = getTMat<TMat>(_depth);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::getDepth(OutputArray _depth)
{
	_depth.assign(this->depth);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setMask(InputArray _mask)
{
	this->mask = getTMat<TMat>(_mask);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::getMask(OutputArray _mask)
{
	_mask.assign(this->mask);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setNormals(InputArray _normals)
{
	this->normals = getTMat<TMat>(_normals);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::getNormals(OutputArray _normals)
{
	_normals.assign(this->normals);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::findMask(InputArray _depth)
{
	Mat depth = _depth.getMat();
	Mat mask(depth.size(), CV_8UC1, Scalar(255));
	for (int y = 0; y < depth.rows; y++)
		for (int x = 0; x < depth.cols; x++)
			if (cvIsNaN(depth.at<float>(y, x)) || depth.at<float>(y, x) > 10 || depth.at<float>(y, x) <= FLT_EPSILON)
				mask.at<uchar>(y, x) = 0;
	this->setMask(mask);
}

#endif // !ODOMETRY_FRAME_IMPL_HPP