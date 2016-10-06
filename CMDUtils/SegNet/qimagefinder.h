#ifndef QIMAGEFINDER_H
#define QIMAGEFINDER_H

#include <QObject>
#include <QDir>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "cnnsegmentnet.h"

class QImageFinder : public QObject
{
    Q_OBJECT
public:
    explicit QImageFinder(QObject *parent = 0);

    static bool readImagesFromPath(const char *_dirname, std::vector<cv::Mat> &_vvis, std::vector<cv::Mat> &_vseg, bool _cvt2gray=false, cv::Size _targetSize=cv::Size(0,0), segnet::ImageResizeMethod _irm=segnet::ImageResizeMethod::PaddZeroAndResize);
    static cv::Mat readImage(const QString &fileName);

private:   
    static cv::Mat __preprocessImage(const cv::Mat &input, bool cvt2gray, cv::Size targetSize, segnet::ImageResizeMethod irm);
};


#endif // QIMAGEFINDER_H
