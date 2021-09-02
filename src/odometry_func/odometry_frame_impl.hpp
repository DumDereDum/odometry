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

	virtual void prepareAsSrcFrame() = 0;
	virtual void prepareAsDstFrame() = 0;
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

	virtual void prepareAsSrcFrame() override;
	virtual void prepareAsDstFrame() override;

private:
	TMat image;
	TMat depth;
	TMat mask;
	TMat normals;
	std::vector< std::vector<TMat> > pyramids;
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
void OdometryFrameImplTMat<TMat>::prepareAsSrcFrame()
{
}

template<typename TMat>
void OdometryFrameImplTMat<TMat>::prepareAsDstFrame()
{
}

#endif // !ODOMETRY_FRAME_IMPL_HPP