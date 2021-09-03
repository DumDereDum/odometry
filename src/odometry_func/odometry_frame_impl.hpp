#ifndef ODOMETRY_FRAME_IMPL_HPP
#define ODOMETRY_FRAME_IMPL_HPP

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/mat.hpp>

#include "odometry_functions.hpp"

using namespace cv;


enum OdometryFramePyramidType
{
	PYR_IMAGE = 0, PYR_DEPTH = 1, PYR_MASK = 2, PYR_CLOUD = 3, PYR_DIX = 4, PYR_DIY = 5, PYR_TEXMASK = 6, PYR_NORM = 7, PYR_NORMMASK = 8,
	N_PYRAMIDS
};

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
	virtual void   setPyramidLevels(size_t _nLevels) = 0;
	virtual size_t getPyramidLevels(OdometryFramePyramidType oftype) = 0;
	virtual void setPyramidAt(InputArray  img,
		OdometryFramePyramidType pyrType, size_t level) = 0;
	virtual void getPyramidAt(OutputArray img,
		OdometryFramePyramidType pyrType, size_t level) = 0;


	virtual void findMask(InputArray image) = 0;

	virtual void prepareICPFrame()  = 0;
	virtual void prepareRGBFrame()  = 0;
	virtual void prepareRGBDFrame() = 0;

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
	virtual void   setPyramidLevels(size_t _nLevels) override;
	virtual size_t getPyramidLevels(OdometryFramePyramidType oftype) override;
	virtual void setPyramidAt(InputArray  img,
		OdometryFramePyramidType pyrType, size_t level) override;
	virtual void getPyramidAt(OutputArray img,
		OdometryFramePyramidType pyrType, size_t level) override;


	virtual void findMask(InputArray image) override;

	virtual void prepareICPFrame()  override;
	virtual void prepareRGBFrame()  override;
	virtual void prepareRGBDFrame() override;

private:
	Size prepareFrameCacheRGB();
	Size prepareFrameCacheICP();

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

template<typename TMat>
OdometryFrameImplTMat<TMat>::OdometryFrameImplTMat(InputArray _image)
	: pyramids(N_PYRAMIDS)
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
	if (oftype < N_PYRAMIDS)
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

template<typename TMat>
void OdometryFrameImplTMat<TMat>::prepareICPFrame()
{
	this->prepareFrameCacheICP();
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::prepareRGBFrame()
{
	this->prepareFrameCacheRGB();
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::prepareRGBDFrame()
{
	this->prepareFrameCacheICP();
	this->prepareFrameCacheRGB();
}


template<typename TMat>
Size OdometryFrameImplTMat<TMat>::prepareFrameCacheRGB()
{
	///*
	// Can be transformed into template argument in the future
	// when this algorithm supports OCL UMats too
	typedef Mat TMat;

	//if (this.dynamicCast<OdometryFrameImpl<UMat>>())
	//	CV_Error(cv::Error::Code::StsBadArg, "RgbdOdometry does not support UMats yet");

	TMat image;
	this->getImage(image);
	if (image.empty())
	{
		if (this->getPyramidLevels(OdometryFramePyramidType::PYR_IMAGE) > 0)
		{
			TMat pyr0;
			this->getPyramidAt(pyr0, OdometryFramePyramidType::PYR_IMAGE, 0);
			this->setImage(pyr0);
		}
		else
			CV_Error(Error::StsBadSize, "Image or pyramidImage have to be set.");
	}
	//CV_ERROR doesn't work on my ps :(
	// Why image must to be CV_8UC1 ?!
	checkImage(image);
	image.convertTo(image, CV_8UC1);


	TMat depth;
	this->getDepth(depth);
	if (depth.empty())
	{
		if (this->getPyramidLevels(OdometryFramePyramidType::PYR_DEPTH) > 0)
		{
			TMat pyr0;
			this->getPyramidAt(pyr0, OdometryFramePyramidType::PYR_DEPTH, 0);
			this->setDepth(pyr0);
		}
		else if (this->getPyramidLevels(OdometryFramePyramidType::PYR_CLOUD) > 0)
		{
			TMat cloud;
			this->getPyramidAt(cloud, OdometryFramePyramidType::PYR_CLOUD, 0);
			std::vector<TMat> xyz;
			split(cloud, xyz);
			this->setDepth(xyz[2]);
		}
		else
			CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
	}
	checkDepth(depth, image.size());

	TMat mask;
	this->getMask(mask);
	if (mask.empty() && this->getPyramidLevels(OdometryFramePyramidType::PYR_MASK) > 0)
	{
		TMat pyr0;
		this->getPyramidAt(pyr0, OdometryFramePyramidType::PYR_MASK, 0);
		this->setMask(pyr0);
	}
	checkMask(mask, image.size());

	preparePyramidImage(image, this->pyramids[OdometryFramePyramidType::PYR_IMAGE], iterCounts.size());

	preparePyramidDepth(depth, this->pyramids[OdometryFramePyramidType::PYR_DEPTH], iterCounts.size());

	preparePyramidMask<TMat>(mask, this->pyramids[OdometryFramePyramidType::PYR_DEPTH], (float)minDepth, (float)maxDepth,
		this->pyramids[OdometryFramePyramidType::PYR_NORM], this->pyramids[OdometryFramePyramidType::PYR_MASK]);

	//if (cacheType & OdometryFramePyramidType::CACHE_SRC)
	//	preparePyramidCloud<TMat>(this->pyramids[OdometryFramePyramidType::PYR_DEPTH], cameraMatrix, this->pyramids[OdometryFramePyramidType::PYR_CLOUD]);

	//if (cacheType & OdometryFramePyramidType::CACHE_DST)
	//{
		preparePyramidSobel<TMat>(this->pyramids[OdometryFramePyramidType::PYR_IMAGE], 1, 0, this->pyramids[OdometryFramePyramidType::PYR_DIX]);
		preparePyramidSobel<TMat>(this->pyramids[OdometryFramePyramidType::PYR_IMAGE], 0, 1, this->pyramids[OdometryFramePyramidType::PYR_DIY]);
		preparePyramidTexturedMask(this->pyramids[OdometryFramePyramidType::PYR_DIX], this->pyramids[OdometryFramePyramidType::PYR_DIY], minGradientMagnitudes,
			this->pyramids[OdometryFramePyramidType::PYR_MASK], maxPointsPart, this->pyramids[OdometryFramePyramidType::PYR_TEXMASK]);
	//}

	return image.size();
	//*/
	//return Mat().size();
}

template<typename TMat>
Size OdometryFrameImplTMat<TMat>::prepareFrameCacheICP()
{
	/*
	TMat depth;
	this->getDepth(depth);
	if (depth.empty())
	{
		if (this->getPyramidLevels(OdometryFramePyramidType::PYR_CLOUD))
		{
			if (this->getPyramidLevels(OdometryFramePyramidType::PYR_NORM))
			{
				TMat points, normals;
				this->getPyramidAt(points, OdometryFramePyramidType::PYR_CLOUD, 0);
				this->getPyramidAt(normals, OdometryFramePyramidType::PYR_NORM, 0);
				std::vector<TMat> pyrPoints, pyrNormals;
				// in, in, out, out
				size_t nLevels = iterCounts.size();
				buildPyramidPointsNormals(points, normals, pyrPoints, pyrNormals, (int)nLevels);
				for (size_t i = 1; i < nLevels; i++)
				{
					this->setPyramidAt(pyrPoints[i], OdometryFramePyramidType::PYR_CLOUD, i);
					this->setPyramidAt(pyrNormals[i], OdometryFramePyramidType::PYR_NORM, i);
				}

				return points.size();
			}
			else
			{
				TMat cloud;
				this->getPyramidAt(cloud, OdometryFramePyramidType::PYR_CLOUD, 0);
				std::vector<TMat> xyz;
				split(cloud, xyz);
				this->setDepth(xyz[2]);
			}
		}
		else if (this->getPyramidLevels(OdometryFramePyramidType::PYR_DEPTH))
		{
			TMat d0;
			this->getPyramidAt(d0, OdometryFramePyramidType::PYR_DEPTH, 0);
			this->setDepth(d0);
		}
		else
			CV_Error(Error::StsBadSize, "Depth or pyramidDepth or pyramidCloud have to be set.");
	}
	this->getDepth(depth);
	checkDepth(depth, depth.size());

	makeFrameFromDepth(depth, this->pyramids[OdometryFramePyramidType::PYR_CLOUD], this->pyramids[OdometryFramePyramidType::PYR_NORM], cameraMatrix, (int)iterCounts.size(),
		depthFactor, sigmaDepth, sigmaSpatial, kernelSize, truncateThreshold);

	return depth.size();
	*/
	return Mat().size();
}

#endif // !ODOMETRY_FRAME_IMPL_HPP