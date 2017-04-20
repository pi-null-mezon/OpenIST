#include "qimagefinder.h"
//--------------------------------------------------------------------------------
QImageFinder::QImageFinder(QObject *parent) : QObject(parent)
{   
}
//--------------------------------------------------------------------------------
bool QImageFinder::readImagesFromPath(const char *_dirname, std::vector<cv::Mat> &_vvis, std::vector<cv::Mat> &_vseg, bool _cvt2gray, cv::Size _targetSize, segnet::ImageResizeMethod _irm)
{
    QDir _dir(_dirname);
    if(_dir.exists()) {
        QStringList v_namefiltersList;
        v_namefiltersList << "*.png" << "*.jpg" << "*.bmp";
        QStringList _filesList =  _dir.entryList(v_namefiltersList, QDir::Files | QDir::NoDotAndDotDot);
        QString filename;
        //qWarning("\nFiles list in training dir:");
        for(int i = 0; i < _filesList.size(); i++) {
            filename = _filesList[i];
            //qWarning("%d) %s", i, filename.toLocal8Bit().constData());
            cv::Mat _mat = __preprocessImage( readImage(_dir.absoluteFilePath(filename)), _cvt2gray, _targetSize, _irm);
            if(!_mat.empty())
                if(filename.contains('@') == true)
                    _vseg.push_back( _mat );
                else
                    _vvis.push_back( _mat );
        }
        return true;
    }
    return false;
}
//--------------------------------------------------------------------------------
cv::Mat QImageFinder::readImage(const QString &fileName)
{
    return cv::imread(fileName.toLocal8Bit().constData(), CV_LOAD_IMAGE_UNCHANGED);
}
//--------------------------------------------------------------------------------
cv::Mat QImageFinder::__preprocessImage(const cv::Mat &input, bool cvt2gray, cv::Size targetSize, segnet::ImageResizeMethod irm)
{
    cv::Mat output;

    if(!input.empty()) {

        if(cvt2gray && input.channels() > 1) {
            switch(input.channels()) {
                case 3:
                    cv::cvtColor(input, output, CV_BGR2GRAY);
                    break;
                case 4:
                    cv::cvtColor(input, output, CV_BGRA2GRAY);
                    break;
            }
        } else if(input.channels() == 4)
            cv::cvtColor(input, output, CV_BGRA2BGR);
        else
            output = input;

        if(targetSize.area() > 0)
            if(output.cols != targetSize.width || output.rows != targetSize.height)
                switch(irm){
                    case segnet::ImageResizeMethod::CropAndResizeFromCenter:
                        output = segnet::__cropresize(output, targetSize);
                        break;
                    case segnet::ImageResizeMethod::PaddZeroAndResize:
                        output = segnet::__propresize(output, targetSize);
                        break;
                }
    }

    return output;
}
//--------------------------------------------------------------------------------

