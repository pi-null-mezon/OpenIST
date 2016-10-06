#ifndef QIMAGEFINDER_H
#define QIMAGEFINDER_H

#include <QObject>
#include <QDir>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

class QImageFinder : public QObject
{
    Q_OBJECT
public:
    explicit QImageFinder(QObject *parent = 0);

    static bool readImagesFromPath(const char *_dirname, std::vector<cv::Mat> &_vvis, std::vector<cv::Mat> &_vseg);
    static cv::Mat readImage(const QString &fileName);  
};


#endif // QIMAGEFINDER_H
