#include <iostream>

#include <opencv2/opencv.hpp>

#include <QStringList>
#include <QString>
#include <QDir>

int main(int argc, char *argv[])
{
    const char *firstdirname = 0, *seconddirname = 0, *outputdirname = 0;
    while(--argc > 0 && (*++argv)[0] == '-') {
        char option = *(++argv[0]);
        switch(option) {
            case 'l':
                firstdirname = ++(*argv);
                break;
            case 'r':
                seconddirname = ++(*argv);
                break;
            case 'o':
                outputdirname = ++(*argv);
                break;
        }
    }

    if(firstdirname == 0 || seconddirname == 0 || outputdirname == 0) {
        std::cerr << "Empty directory name. Abort...";
        return -1;
    }
    QDir _fdir(firstdirname), _sdir(seconddirname), _odir(outputdirname);
    if(_fdir.exists() == false || _sdir.exists() == false) {
        std::cerr << "One or both of directories do not exist. Abort...";
        return -2;
    } else {
        if(_odir.exists() == false)
            _odir.mkpath(outputdirname);
    }

    QStringList _lformats;
    _lformats << "*.png" << "*.jpg" << "*.bmp";
    QStringList _lfs1 = _fdir.entryList(_lformats, QDir::NoDotAndDotDot | QDir::Files);
    QStringList _lfs2 = _sdir.entryList(_lformats, QDir::NoDotAndDotDot | QDir::Files);

    if(_lfs1.size() != _lfs2.size()) {
        std::cerr << "Directories content mismatch. Abort...";
        return -3;
    }

    for(int i = 0; i < _lfs1.size(); i++) {

        cv::Mat _mat1 = cv::imread(_fdir.absoluteFilePath(_lfs1[i]).toLocal8Bit().constData(), CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat _mat2 = cv::imread(_sdir.absoluteFilePath(_lfs2[i]).toLocal8Bit().constData(), CV_LOAD_IMAGE_UNCHANGED);

        cv::Mat _tempmat = _mat2+_mat1;
        cv::namedWindow("probe", CV_WINDOW_NORMAL);
        cv::imshow("probe", _tempmat);
        cv::waitKey(1);

        QString _outputfilename = (_lfs1[i]).section('.',0,0) + QString("@.") + (_lfs1[i]).section('.',1,1);

       cv::imwrite((_odir.absolutePath() + QString("/") + _outputfilename).toLocal8Bit().constData(), _tempmat);
    }
    std::cout << "Operation has been performed successfuly.";
    return 0;
}
