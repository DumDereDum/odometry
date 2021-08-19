#ifndef ODOMETRY_FUNCTIONS_HPP
#define ODOMETRY_FUNCTIONS_HPP

#include <iostream>
#include <opencv2/core/mat.hpp>

using namespace cv;


template<typename TMat>
static TMat getTMat(InputArray, int = -1);

template<>
Mat getTMat<Mat>(InputArray a, int i)
{
	std::cout << "getTMat<Mat>" << std::endl;
	return a.getMat(i);
}

template<>
UMat getTMat<UMat>(InputArray a, int i)
{
	std::cout << "getTMat<UMat>" << std::endl;
	return a.getUMat(i);
}


#endif //ODOMETRY_FUNCTIONS_HPP