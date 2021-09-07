#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/mat.hpp>

#include "utils.hpp"
#include "odometry_frame.hpp"

using namespace cv;

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
	virtual void   setPyramidLevels(size_t _nLevels) override;
	virtual size_t getPyramidLevels(OdometryFramePyramidType oftype) override;
	virtual void setPyramidAt(InputArray  img,
		OdometryFramePyramidType pyrType, size_t level) override;
	virtual void getPyramidAt(OutputArray img,
		OdometryFramePyramidType pyrType, size_t level) override;


	virtual void findMask(InputArray image) override;

private:

	TMat image;
	TMat depth;
	TMat mask;
	TMat normals;
	std::vector< std::vector<TMat> > pyramids;

	const float maxTranslation = 0.15f;
	const float maxRotation = 15.f;
	const float minGradientMagnitude = 10.f;
	const std::vector<int> iterCounts = { 7, 7, 7, 10 };
	const cv::Matx33f cameraMatrix = { /* fx, 0, cx*/ 0, 0, 0, /* 0, fy, cy */ 0, 0, 0, /**/ 0, 0, 0 };
	const float minDepth = 0.f;
	const float maxDepth = 4.f;
	const float maxDepthDiff = 0.07f;
	const float maxPointsPart = 0.07f;
	//const InputArray minGradientMagnitudes = noArray();
	std::vector<float> minGradientMagnitudes = std::vector<float>(iterCounts.size(), minGradientMagnitude);

};

OdometryFrame::OdometryFrame(InputArray image)
{
	bool allEmpty = image.empty();
	bool useOcl = image.isUMat();
	bool isNoArray = (&image == &noArray());
	//if (useOcl && !allEmpty && !isNoArray)
	std::cout << " allEmpty:" << allEmpty << " useOcl:" << useOcl << " isNoArray:" << isNoArray << std::endl;
	if (useOcl)
		this->odometryFrame = makePtr<OdometryFrameImplTMat<UMat>>(image);
	else
		this->odometryFrame = makePtr<OdometryFrameImplTMat<Mat>>(image);
};

template<typename TMat>
OdometryFrameImplTMat<TMat>::OdometryFrameImplTMat(InputArray _image)
	: pyramids(OdometryFramePyramidType::N_PYRAMIDS)
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
void OdometryFrameImplTMat<TMat>::setPyramidLevels(size_t _nLevels)
{
	for (auto& p : pyramids)
	{
		p.resize(_nLevels, TMat());
	}
}

template<typename TMat>
size_t OdometryFrameImplTMat<TMat>::getPyramidLevels(OdometryFramePyramidType oftype)
{
	if (oftype < OdometryFramePyramidType::N_PYRAMIDS)
		return pyramids[oftype].size();
	else
		return 0;
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

template<typename TMat>
void OdometryFrameImplTMat<TMat>::setPyramidAt(InputArray  _img, OdometryFramePyramidType pyrType, size_t level)
{
	TMat img = getTMat<TMat>(_img);
	pyramids[pyrType][level] = img;
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::getPyramidAt(OutputArray _img, OdometryFramePyramidType pyrType, size_t level)
{
	TMat img = pyramids[pyrType][level];
	_img.assign(img);
}
