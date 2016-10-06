#include "qimagefinder.h"
//--------------------------------------------------------------------------------
QImageFinder::QImageFinder(QObject *parent) : QObject(parent)
{   
}
//--------------------------------------------------------------------------------
bool QImageFinder::readImagesFromPath(const char *_dirname, std::vector<cv::Mat> &_vvis, std::vector<cv::Mat> &_vseg)
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
            qWarning("%d) %s", i, filename.toLocal8Bit().constData());
            cv::Mat _mat = readImage(_dir.absoluteFilePath(filename));
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



