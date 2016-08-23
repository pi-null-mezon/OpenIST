#ifndef DICOMOPENCV_H
#define DICOMOPENCV_H

#include <opencv2/core.hpp>

#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>

namespace dicom {
/**
 * @brief dicomRead should be used to extract images from DICOM files
 * @param _filename - input filename
 * @param _rcode - result code
 * @return image from DICOM file in opencv basic image format, empty image could be returned!
 */
cv::Mat dicomRead(const char *_filename, EI_Status *_rcode=0);

}

#endif // DICOMOPENCV_H
