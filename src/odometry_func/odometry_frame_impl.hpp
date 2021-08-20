#ifndef ODOMETRY_FRAME_IMPL
#define ODOMETRY_FRAME_IMPL

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
	virtual void setImage(InputArray image) = 0;

private:

};

template<typename TMat>
class OdometryFrameImplTMat : public OdometryFrameImpl
{
public:
	OdometryFrameImplTMat(InputArray image = noArray());
	~OdometryFrameImplTMat() {};

	virtual void setImage(InputArray image) override;

private:
	TMat image;
};

template<typename TMat>
OdometryFrameImplTMat<TMat>::OdometryFrameImplTMat(InputArray _image)
{
	image = getTMat<TMat>(_image);
};

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setImage(InputArray _image)
{
	image = getTMat<TMat>(_image);
}
/*
Ptr<OdometryFrameImpl> createOdometryFrame(InputArray image)
{
	//bool allEmpty = image.empty();
	//bool useOcl = image.isUMat();
	//Ptr<OdometryFrameImpl> res;
	//if (useOcl && !allEmpty)
	//	makePtr<OdometryFrameImplTMat<UMat>>(image);
	//else
	//	makePtr<OdometryFrameImplTMat<Mat>>(image);
	//return makePtr<OdometryFrameImplTMat<Mat>>();
}
*/

#endif // !ODOMETRY_FRAME_IMPL