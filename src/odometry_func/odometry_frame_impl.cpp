#include "odometry_frame_impl.hpp"


template<typename TMat>
static TMat getTMat(InputArray, int = -1);

template<>
Mat getTMat<Mat>(InputArray a, int i)
{
	return a.getMat(i);
}

template<>
UMat getTMat<UMat>(InputArray a, int i)
{
	return a.getUMat(i);
}


template<typename TMat>
class OdometryFrameImplTMat : public OdometryFrameImpl
{
public:
	OdometryFrameImplTMat(InputArray image);
	~OdometryFrameImplTMat() {};

	virtual void setX(InputArray image) override;

private:
	TMat image;
};

template<typename TMat>
OdometryFrameImplTMat<TMat>::OdometryFrameImplTMat(InputArray image)
{
	image = getTMat<TMat>(_image);
};

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setX(InputArray _image)
{
	image = getTMat<TMat>(_image);
}