#include <iostream>
#include <QDir>
#include <QStringList>
#include "cnnsegmentnet.h"
#include <opencv2/imgcodecs.hpp>

int main(int argc, char *argv[])
{
    char *sourcedirname = 0;
    char *outputdirname = 0;;
    char *networkname = 0;

    while((--argc > 0) && (**(++argv) == '-')) {
        char option = *++argv[0];
        switch(option) {
            case 'i':
                sourcedirname = ++argv[0];
                break;
            case 'o':
                outputdirname = ++argv[0];
                break;
            case 'n':
                networkname = ++argv[0];
                break;
            case 'h':
                //TO DO help section
                return 0;
        }
    }

    if(sourcedirname == 0) {
        return -1;
    }
    if(outputdirname == 0) {
        return -2;
    }
    if(networkname == 0) {
        return -3;
    }

    segnet::SegNetForLungs _segnet;
    if(_segnet.load(networkname) == false) {
        return -4;
    }

    QDir sourcedir(sourcedirname);
    if(sourcedir.exists() == false)
        return -5;

    QDir outdir(outputdirname);
    if(outdir.exists() == false) {
        std::cout << "Output directory does not exist. Thus, it will be created in " << outputdirname << std::endl;
        outdir.mkpath(outputdirname);
        outdir.cd(outputdirname);
    }

    QStringList filters;
    filters << "*.png" << "*.jpg" << "*.bmp";

    QStringList l_files = sourcedir.entryList(filters, QDir::NoDotAndDotDot | QDir::Files);
    for(int i = 0; i < l_files.size(); i++) {
        cv::Mat _img = cv::imread(sourcedir.absoluteFilePath(l_files[i]).toLocal8Bit().constData(), cv::IMREAD_UNCHANGED);
        if(_img.empty()) {
            return -6;
        } else {
            cv::Mat _simg = _segnet.predict(_img);
            _simg.convertTo(_simg, CV_8U, 255, 0);
            // In most cases Otsu works better than fixed threshold
            cv::threshold(_simg,_simg,0.0,255.0,CV_THRESH_OTSU);
            _img.copyTo(_simg,_simg);
            cv::imwrite((outdir.absolutePath() + "/" + l_files[i]).toLocal8Bit().constData(), _simg);
            std::cout << " " << i << ") "<< l_files[i].toLocal8Bit().constData() << " has been enrolled and saved" << std::endl;
        }
    }



    return 0;
}
