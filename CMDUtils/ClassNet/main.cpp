#include <QString>
#include <iostream>

#include "cnnclassnet.h"
#include "qimagefinder.h"

int main(int argc, char *argv[])
{
    #ifdef Q_OS_WIN32
    setlocale(LC_CTYPE,"");
    #endif

    const char *_indirname = 0, *_outcnnfilename = 0;
    int _epoch = 30, _minibatch = 4, _width = 32, _height = 32, _inchannels = 0, _outchannels = 0, _t2c = 9;
    const char *_incnnfilename = 0, *_inimgfilename = 0;
    while(--argc > 0 && (*++argv)[0] == '-') {
        char _opt = *++argv[0];
        switch(_opt) {
        case 'i':
            _indirname = ++(*argv);
            break;
        case 'o':
            _outcnnfilename = ++(*argv);
            break;
        case 'e':
            _epoch = QString(++(*argv)).toInt();
            break;
        case 'm':
            _minibatch = QString(++(*argv)).toInt();
            break;
        case 'c':
            _width = QString(++(*argv)).toInt();
            break;
        case 'r':
            _height = QString(++(*argv)).toInt();
            break;
        case 'x':
            _inchannels = QString(++(*argv)).toInt();
            break;
        case 'y':
            _outchannels = QString(++(*argv)).toInt();
            break;
        case 'n':
            _incnnfilename = ++(*argv);
            break;
        case 's':
            _inimgfilename = ++(*argv);
            break;
        case 'p':
            _t2c = QString(++(*argv)).toInt();
            break;
        case 'h':
            QString helpstr( "Comand line utility for the image segmentation by means of CNN"
                             "\n To train CNN use:"
                             "\n -i[dirname]  - directory with training data*"
                             "\n -o[filename] - name for the output file with trained network"
                             "\n -e[int] - number of training epoch (ref. to tiny-dnn doc.)"
                             "\n -m[int] - minibatch size for training (ref. to tiny-dnn doc.)"
                             "\n -c[int] - to what cols training images should be resized (default %1)"
                             "\n -r[int] - to what rows training images should be resized (default %2)"
                             "\n -x[int] - desired number of input channels (0 means same as in source)"
                             "\n -p[int] - each p-th sample will goes to control set"
                             "\n To segment image by CNN:"
                             "\n -n[filename] - name for the file with pretrained network"
                             "\n -s[filename] - image for segmentation"                             
                             "\nAlex A. Taranov, based on Qt, Opencv and tiny-dnn");
            std::cout << helpstr.arg(QString::number(_width),QString::number(_height)).toLocal8Bit().constData();
            return 0;
        }
    }

    if(_indirname != 0) {

        if(_outcnnfilename == 0) {
            std::cout << "You have not provide filname for the output network. Abort...";
            return -3;
        }

        std::cout << "Training mode selected. Data preprocessing started...\n";

        std::vector<cv::Mat> vimgs;
        std::vector<tiny_dnn::label_t> vlabels;
        QImageFinder::readImagesFromPath(_indirname, vimgs, vlabels, false, cv::Size(_width, _height),ImageResizeMethod::PaddZeroAndResize);
        std::cout << "Total " << vimgs.size() << " images have been found and preprocessed..." << std::endl;

        CNNClassificator net;
        net.setImageResizeMethod(ImageResizeMethod::PaddZeroAndResize);
        net.setInputChannels(_inchannels);
        net.setInputSize(cv::Size(_width, _height));
        net.setBackupFilePrefix(QString(_outcnnfilename).section('.',0,0).toLocal8Bit().constData());

        if(_incnnfilename != 0) {
            if(net.load(_incnnfilename)) {
                std::cout << "Updating started...\n";
                net.update(vimgs, vlabels, _epoch, _minibatch, _t2c);
                std::cout << "Updating finished.\n";
            } else {
                std::cout << "Can not load network from file " << _incnnfilename;
                return -1;
            }
        } else {
            std::cout << "Training started...\n";
            net.train(vimgs, vlabels, _epoch, _minibatch, _t2c);
            std::cout << "Training finished.\n";
        }

        net.save(_outcnnfilename);
        std::cout << "Network has been saved in " << _outcnnfilename << "\n";
    }

    if(_incnnfilename != 0 && _inimgfilename != 0) {
        std::cout << "Prediction mode selected\n";

        CNNClassificator net;
        if(net.load(_incnnfilename))
            std::cout << "Network loaded from " << _incnnfilename << "\n";
        else {
            std::cout << "Can not load network from file " << _incnnfilename;
            return -1;
        }

        cv::Mat img = QImageFinder::readImage(_inimgfilename);
        if(img.empty()) {
            std::cout << "Can not open image to segment! Abort...";
            return -2;
        }

        std::cout << "Predicted class label is " << net.predict(img) << std::endl;
    }

    return 0;
}
