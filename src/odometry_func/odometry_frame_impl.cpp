#include "odometry_frame_impl.hpp"



template<typename TMat>
void OdometryFrameImplTMat<TMat>::setImage(InputArray _image)
{
	image = getTMat<TMat>(_image);
}