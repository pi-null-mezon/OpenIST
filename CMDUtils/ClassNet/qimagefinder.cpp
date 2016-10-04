#include "qimagefinder.h"
//--------------------------------------------------------------------------------
QImageFinder::QImageFinder(QObject *parent) : QObject(parent)
{   
}
//--------------------------------------------------------------------------------
bool QImageFinder::readImagesFromPath(const char *_dirname, std::vector<cv::Mat> &_vraw, std::vector<tiny_cnn::label_t> &_vlabels, bool _cvt2gray, cv::Size _targetSize)
{
    QDir _dir(_dirname);
    if(_dir.exists()) {
        QStringList v_namefiltersList;
        v_namefiltersList << "*.png" << "*.jpg" << "*.bmp";
        QStringList _filesList =  _dir.entryList(v_namefiltersList, QDir::Files | QDir::NoDotAndDotDot);
        QString filename;
        qWarning("\nFiles list in training dir:");
        for(int i = 0; i < _filesList.size(); i++) {
            filename = _filesList[i];
            // last number (i.e. last before file extension) after '_' in filename determines label of the example
            tiny_cnn::label_t _label = static_cast<tiny_cnn::label_t>( (filename.section('_',-1,-1).section('.',0,0)).toUInt() );
            qWarning("%d) %s, label: %d", i, filename.toLocal8Bit().constData(), _label);
            cv::Mat _mat = __preprocessImage( readImage(_dir.absoluteFilePath(filename)), _cvt2gray, _targetSize);
            if(!_mat.empty()) {
                _vraw.push_back( _mat );
                _vlabels.push_back( _label );
            }
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
cv::Mat QImageFinder::__preprocessImage(const cv::Mat &input, bool cvt2gray, cv::Size targetSize)
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
                output = __cropresizeImage(output, targetSize);
    }

    return output;
}
//--------------------------------------------------------------------------------
cv::Mat QImageFinder::__cropresizeImage(const cv::Mat &input, const cv::Size size)
{
    cv::Rect2f roiRect(0,0,0,0);
    if( (float)input.cols/input.rows > (float)size.width/size.height) {
        roiRect.height = input.rows;
        roiRect.width = input.rows * (float)size.width/size.height;
        roiRect.x = (input.cols - roiRect.width)/2.0f;
    } else {
        roiRect.width = input.cols;
        roiRect.height = input.cols * (float)size.height/size.width;
        roiRect.y = (input.rows - roiRect.height)/2.0f;
    }
    roiRect &= cv::Rect2f(0, 0, input.cols, input.rows);
    cv::Mat output;
    if(roiRect.area() > 0)  {
        cv::Mat croppedImg(input, roiRect);
        int interpolationMethod = 0;
        if(size.area() > roiRect.area())
            interpolationMethod = CV_INTER_CUBIC;
        else
            interpolationMethod = CV_INTER_AREA;
        cv::resize(croppedImg, output, size, 0, 0, interpolationMethod);
    }
    return output;
}
//--------------------------------------------------------------------------------

