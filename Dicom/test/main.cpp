#include <iostream>
#include <opencv2/opencv.hpp>

#include "dicomopencv.h"

void readNameFromDicom(const char *filename);

int main(int argc, char *argv[])
{
    const char * filename = "C:/X_RAY/Demo/000000.dcm";

    readNameFromDicom(filename);
    cv::Mat _mat = dicom::dicomRead(filename);

    if(_mat.empty() == false) {

        std::cout << "\nMat depth: " << _mat.depth();

        cv::namedWindow("DICOM", CV_WINDOW_NORMAL);
        cv::imshow("DICOM", _mat);

        double _max,_min;
        int _minidx,_maxidx;
        cv::minMaxIdx(_mat,&_min,&_max,&_minidx,&_maxidx);
        std::cout << "\nMax: " << _max << "\tMin: " << _min;

        cv::Mat _nmat;
        _mat.convertTo(_nmat, CV_32F, 1.0/_max);
        cv::namedWindow("Normalized", CV_WINDOW_NORMAL);
        cv::imshow("Normalized", _nmat);

        while(true) {
            if(cv::waitKey(33) == 'q')
                break;
        }
    } else {
        std::cerr << "\nCould not open image from dicom file";
    }

    return 0;
}

void readNameFromDicom(const char *filename) {

    DcmFileFormat fileformat;
    OFCondition status = fileformat.loadFile(filename);
    if(status.good())
    {
        OFString patientName;
        if (fileformat.getDataset()->findAndGetOFString(DCM_PatientName, patientName).good())
            std::cout << "Patient's Name: " << patientName << std::endl;
        else
            std::cerr << "Error: cannot access Patient's Name!" << std::endl;

    } else
        std::cerr << "Error: cannot read DICOM file (" << status.text() << ")" << std::endl;


}




