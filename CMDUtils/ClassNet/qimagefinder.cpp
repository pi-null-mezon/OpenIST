#include "qimagefinder.h"
//--------------------------------------------------------------------------------
QImageFinder::QImageFinder(QObject *parent) : QObject(parent)
{   
}
//--------------------------------------------------------------------------------
bool QImageFinder::readImagesFromPath(const char *_dirname, std::vector<cv::Mat> &_vraw, std::vector<tiny_dnn::label_t> &_vlabels, bool _cvt2gray, cv::Size _targetSize, ImageResizeMethod _irm)
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
            // last number (i.e. last before file extension) after '_' in filename determines label of the example
            tiny_dnn::label_t _label = static_cast<tiny_dnn::label_t>( (filename.section('_',-1,-1).section('.',0,0)).toUInt() );
            //qWarning("%d) %s, label: %d", i, filename.toLocal8Bit().constData(), _label);
            cv::Mat _mat = __preprocessImage( readImage(_dir.absoluteFilePath(filename)), _cvt2gray, _targetSize, _irm);
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
cv::Mat QImageFinder::__preprocessImage(const cv::Mat &input, bool cvt2gray, cv::Size targetSize, ImageResizeMethod _irm)
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
                switch(_irm){
                    case ImageResizeMethod::CropAndResizeFromCenter:
                        output =__cropresize(output, targetSize);
                        break;
                    case ImageResizeMethod::PaddZeroAndResize:
                        output = __propresize(output, targetSize);
                        break;
                }
    }

    return output;
}
//--------------------------------------------------------------------------------


