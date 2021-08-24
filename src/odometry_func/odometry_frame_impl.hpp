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

private:
	TMat image;
	TMat depth;
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
	//_image = getTMat<TMat>(this->image);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setDepth(InputArray _depth)
{
	this->depth = getTMat<TMat>(_depth);
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::getDepth(OutputArray _depth)
{
	//_depth = getTMat<TMat>(this->depth);
}


#endif // !ODOMETRY_FRAME_IMPL_HPP