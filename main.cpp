#include <QString>
#include <iostream>

#include "cnnsegmentnet.h"
#include "qimagefinder.h"

int main(int argc, char *argv[])
{
    #ifdef Q_OS_WIN32
    setlocale(LC_CTYPE,"");
    #endif

    const char *_indirname = 0, *_outcnnfilename = 0;
    int _epoch = 10, _minibatch = 1, _width = 320, _height = 240, _inchannels = 0, _outchannels = 0;
    const char *_incnnfilename = 0, *_inimgfilename = 0, *_outimgfilename = 0;
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
        case 'a':
            _outimgfilename = ++(*argv);
            break;
        case 'h':
            QString helpstr( "Comand line utility for the image segmentation by means of CNN"
                             "\n To train CNN use:"
                             "\n -i[dirname]  - directory with training data*"
                             "\n -o[filename] - name for the output file with trained network"
                             "\n -e[int] - number of training epoch (ref. to tiny-dnn doc.)"
                             "\n -m[int] - minibatch size for training (ref. to tiny-dnn doc.)"
                             "\n -c[int] - to what cols training images should be resized"
                             "\n -r[int] - to what rows training images should be resized"
                             "\n -x[int] - desired number of input channels (0 means same as in source)"
                             "\n -y[int] - desired number of output channels (0 means same as in source)"
                             "\n To segment image by CNN:"
                             "\n -n[filename] - name for the file with pretrained network"
                             "\n -s[filename] - image for segmentation"
                             "\n -a[filename] - where to save segmented image"
                             "\nAlex A. Taranov, based on Qt, Opencv and tiny-dnn");
            std::cout << helpstr.toLocal8Bit().constData();
            return 0;
        }
    }

    if(_indirname != 0) {
        std::cout << "Training mode selected\n";

        std::vector<cv::Mat> vimgs, vlbls;
        QImageFinder::readImagesFromPath(_indirname, vimgs, vlbls);
        std::cout << "Found " << vimgs.size() << " images and " << vlbls.size() << " segmented images.\n";

        CNNSegmentnet net;
        net.setInputChannels(_inchannels);
        net.setOutputChannels(_outchannels);
        net.setInputSize(cv::Size(_width, _height));

        std::cout << "Training started...\n";
        net.train(vimgs, vlbls, _epoch, _minibatch);
        std::cout << "Training finished.\n";

        if(_outcnnfilename != 0) {
            net.save(_outcnnfilename);
            std::cout << "Network has been saved in " << _outcnnfilename << ".\n";
        }
    }

    if(_incnnfilename != 0 && _inimgfilename != 0) {
        std::cout << "Prediction mode selected\n";

        CNNSegmentnet net;
        if(net.load(_incnnfilename))
            std::cout << "Network loaded from " << _incnnfilename << ".\n";
        else {
            std::cout << "Can not load network from file " << _incnnfilename;
            return -1;
        }

        cv::Mat img = QImageFinder::readImage(_inimgfilename);
        if(img.empty()) {
            std::cout << "Can not open image to segment! Abort...";
            return -2;
        }

        cv::Mat _outmat = net.predict(img);
        std::cout << "Press 'q' to quit from the app.";
        if(!_outmat.empty()) {
            while(true) {
                cv::namedWindow("Segmented image", CV_WINDOW_NORMAL);
                cv::imshow("Segmented image", _outmat);
                if(cv::waitKey(33) == 'q')
                    break;
            }
        }
    }


    return 0;
}
