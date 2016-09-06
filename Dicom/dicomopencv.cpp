#include "dicomopencv.h"
#include <QDebug>

namespace dicom {

cv::Mat dicomRead(const char *_filename, EI_Status *_rcode)
{
    DicomImage _srcdcm(_filename);

    if(_srcdcm.getStatus() == EIS_Normal)   {
        if(_rcode != 0)
            *_rcode = EIS_Normal;

        if (_srcdcm.isMonochrome()) {
            int _mattype;
            qDebug() << _srcdcm.getDepth();

            int depth = _srcdcm.getDepth();
            if(depth <= 8)
                _mattype = CV_8UC1; // Monochrome 8-bit
            else if(depth <= 16)
                _mattype = CV_16UC1; // Monochrome 16-bit
            else
                _mattype = CV_32SC1; // Monochrome 32-bit signed

            // Enable intensity transformations if needed
            //_srcdcm.setMinMaxWindow();

            void *pixelData = (void *)_srcdcm.getOutputData();
            if(pixelData) {
                cv::Mat _mat(_srcdcm.getHeight(), _srcdcm.getWidth(), _mattype, pixelData);
                _mat = _mat.clone();
                _srcdcm.deleteOutputData();
                return _mat;
            }

        } else {
            // TO DO enroll color images convertation
        }
    }
    if(_rcode != 0)
        *_rcode = _srcdcm.getStatus();
    return cv::Mat(); // return an empty image if something goes wrong
}



} // end of namespace dicom
