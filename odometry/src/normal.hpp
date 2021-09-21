#ifndef NORMAL_HPP
#define NORMAL_HPP

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/utility.hpp>

using namespace cv;

class RgbdNormals
{
public:
    enum RgbdNormalsMethod
    {
        RGBD_NORMALS_METHOD_FALS = 0,
        RGBD_NORMALS_METHOD_LINEMOD = 1,
        RGBD_NORMALS_METHOD_SRI = 2
    };

    RgbdNormals() { }
    virtual ~RgbdNormals() { }

    /** Creates new RgbdNormals object
     * @param rows the number of rows of the depth image normals will be computed on
     * @param cols the number of cols of the depth image normals will be computed on
     * @param depth the depth of the normals (only CV_32F or CV_64F)
     * @param K the calibration matrix to use
     * @param window_size the window size to compute the normals: can only be 1,3,5 or 7
     * @param method one of the methods to use: RGBD_NORMALS_METHOD_SRI, RGBD_NORMALS_METHOD_FALS
     */
    static Ptr<RgbdNormals> create(int rows = 0, int cols = 0, int depth = 0, InputArray K = noArray(), int window_size = 5,
        RgbdNormals::RgbdNormalsMethod method = RgbdNormals::RgbdNormalsMethod::RGBD_NORMALS_METHOD_FALS);

    /** Given a set of 3d points in a depth image, compute the normals at each point.
     * @param points a rows x cols x 3 matrix of CV_32F/CV64F or a rows x cols x 1 CV_U16S
     * @param normals a rows x cols x 3 matrix
     */
    virtual void apply(InputArray points, OutputArray normals) const = 0;

    /** Prepares cached data required for calculation
    * If not called by user, called automatically at first calculation
    */
    virtual void cache() const = 0;

    virtual int getRows() const = 0;
    virtual void setRows(int val) = 0;
    virtual int getCols() const = 0;
    virtual void setCols(int val) = 0;
    virtual int getWindowSize() const = 0;
    virtual void setWindowSize(int val) = 0;
    virtual int getDepth() const = 0;
    virtual void getK(OutputArray val) const = 0;
    virtual void setK(InputArray val) = 0;
    virtual RgbdNormals::RgbdNormalsMethod getMethod() const = 0;
    virtual void setMethod(RgbdNormals::RgbdNormalsMethod val) = 0;
};


#endif // !NORMAL_HPP
