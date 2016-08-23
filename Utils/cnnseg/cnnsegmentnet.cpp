#include "cnnsegmentnet.h"
#include <opencv2/highgui.hpp>

#include <qDebug>

using namespace activation;
//-------------------------------------------------------------------------------------------------------
CNNSegmentnet::CNNSegmentnet()
{
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
network<sequential> CNNSegmentnet::__initNet(const cv::Size &size, int inchannels, int outchannels)
{  
    int _kernels = 3;
    network<sequential> _net;

    /*_net << convolutional_layer<elu>(size.width, size.height, 5, inchannels, _kernels, padding::same)
         << convolutional_layer<elu>(size.width, size.height, 3, _kernels, 2*_kernels, padding::same)
            << average_pooling_layer<elu>(size.width, size.height, 2*_kernels, 2)
            << convolutional_layer<relu>(size.width/2, size.height/2, 3, 2*_kernels, 4*_kernels, padding::same)
            << convolutional_layer<relu>(size.width/2, size.height/2, 3, 4*_kernels, 4*_kernels, padding::same)
            << average_unpooling_layer<elu>(size.width/2, size.height/2, 4*_kernels, 2)
         << convolutional_layer<elu>(size.width, size.height, 3, 4*_kernels, 2*_kernels, padding::same)
         << convolutional_layer<tan_h>(size.width, size.height, 5, 2*_kernels, outchannels, padding::same);
*/
    _net << convolutional_layer<relu>(size.width, size.height, 5, inchannels, _kernels, padding::same)
         << convolutional_layer<relu>(size.width, size.height, 3, _kernels, 2*_kernels, padding::same)
         << average_pooling_layer<relu>(size.width, size.height, 2*_kernels, 2)
         << convolutional_layer<relu>(size.width/2, size.height/2, 3, 2*_kernels, 4*_kernels, padding::same)
         << convolutional_layer<relu>(size.width/2, size.height/2, 3, 4*_kernels, 4*_kernels, padding::same)
         << average_unpooling_layer<relu>(size.width/2, size.height/2, 4*_kernels, 2)
         << convolutional_layer<relu>(size.width, size.height, 3, 4*_kernels, 4*_kernels, padding::same)
         << convolutional_layer<tan_h>(size.width, size.height, 5, 4*_kernels, outchannels, padding::same);

    return _net;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNSegmentnet::train(cv::InputArrayOfArrays _vvis, cv::InputArrayOfArrays _vseg, int _epoch, int _minibatch)
{
    if(_vvis.kind() != cv::_InputArray::STD_VECTOR_MAT && _vseg.kind() != cv::_InputArray::STD_VECTOR_MAT) {
        cv::String error_message = "The images are expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_MAT (a std::vector<Mat>).";
        CV_Error(cv::Error::StsBadArg, error_message);
    }
    if(_vvis.total() == 0 || _vseg.total() == 0) {
        cv::String error_message = cv::format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(cv::Error::StsUnsupportedFormat, error_message);
    }

    // Get the vector of visual images
    std::vector<cv::Mat> srcmats;
    std::vector<vec_t> srcvec_t;
    _vvis.getMatVector(srcmats);
    for(size_t it = 0; it < srcmats.size(); it++) {
        if(it == 0) {
            if(m_inputchannels == 0)
                m_inputchannels = (srcmats[it]).channels();
            if(m_inputsize.area() == 0) {
                setInputSize(cv::Size((srcmats[it]).cols,(srcmats[it]).rows));
            }
        }

        srcvec_t.push_back( __mat2vec_t(srcmats[it], m_inputsize, m_inputchannels) );
    }

    // Get the vector of segmented images
    srcmats.clear();
    _vseg.getMatVector(srcmats);
    std::vector<vec_t> segvec_t;
    for(size_t it = 0; it < srcmats.size(); it++) {
        if(it == 0 && m_outputchannels == 0)
            m_outputchannels = (srcmats[it]).channels();

        segvec_t.push_back( __mat2vec_t(srcmats[it], m_inputsize, m_outputchannels) );
    }

    // Check if data is well-aligned
    if(segvec_t.size() != srcvec_t.size()) {
        cv::String error_message = cv::format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", srcvec_t.size(), segvec_t.size());
        CV_Error(cv::Error::StsBadArg, error_message);
    }   

    m_net = __initNet(m_inputsize, m_inputchannels, m_outputchannels);

    // Batch_size is a number of samples enrolled per parameter update
    adagrad _opt;
    m_net.train<mse>(_opt, srcvec_t, segvec_t, _minibatch, _epoch);
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNSegmentnet::setInputChannels(int _value)
{
    m_inputchannels = _value;
}
void CNNSegmentnet::setOutputChannels(int _value)
{
    m_outputchannels = _value;
}
int CNNSegmentnet::getInputChannels() const
{
    return m_inputchannels;
}
int CNNSegmentnet::getOutputChannels() const
{
    return m_outputchannels;
}
void CNNSegmentnet::setInputSize(const cv::Size &_size)
{
    m_inputsize = _size;
}
cv::Size CNNSegmentnet::getInputSize() const
{
    return m_inputsize;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNSegmentnet::save(const char *filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if(fs.isOpened()) {

        fs << "width" << m_inputsize.width;
        fs << "height" << m_inputsize.height;
        fs << "inchannels" << m_inputchannels;
        fs << "outchannels" << m_outputchannels;

        std::vector<tiny_cnn::float_t> _weights;
        std::vector<tiny_cnn::vec_t*> _w;
        for(size_t i = 0; i < m_net.depth(); i++) {
            _w = m_net[i]->get_weights();
            tiny_cnn::vec_t *_v;
            for(size_t j = 0; j < _w.size(); j++) {
                _v = _w[j];
                _weights.insert(_weights.end(), _v->begin(), _v->end());
            }
        }
        fs << "weights" << _weights;

        fs.release();
    }
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
bool CNNSegmentnet::load(const char *filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if(fs.isOpened()) {

        int w, h;
        fs["width"] >> w;
        fs["height"] >> h;
        setInputSize(cv::Size(w,h));
        fs["inchannels"] >> m_inputchannels;
        fs["outchannels"] >> m_outputchannels;

        m_net = __initNet(m_inputsize, m_inputchannels, m_outputchannels);

        std::vector<tiny_cnn::float_t> _weights;
        fs["weights"] >> _weights;
        int idx = 0;
        std::vector<tiny_cnn::vec_t*> _w;
        for(size_t i = 0; i < m_net.depth(); i++) {
            _w = m_net[i]->get_weights();
            tiny_cnn::vec_t *_v;
            for(size_t j = 0; j < _w.size(); j++) {
                _v = _w[j];
                for(size_t k = 0; k < _v->size(); k++)
                    _v->at(k) = _weights[idx++];
            }
        }

        fs.release();
        return true;
    }
    return false;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
cv::Mat CNNSegmentnet::predict(const cv::Mat &image) const
{
    vec_t vect = m_net.predict( __mat2vec_t(image, m_inputsize, m_inputchannels) );

    visualizeActivations(m_net);

    cv::Mat _outmat;
    int _type = (sizeof(tiny_cnn::float_t) == sizeof(double)) ? CV_64FC1 : CV_32FC1;
    switch(m_outputchannels){
        case 1:
            _outmat = cv::Mat( m_inputsize, _type, &vect[0]);
            break;
        case 3:
            std::vector<cv::Mat> vmats;
            vmats.push_back( cv::Mat( m_inputsize, _type, &vect[2*m_inputsize.area()]) );
            vmats.push_back( cv::Mat( m_inputsize, _type, &vect[m_inputsize.area()]) );
            vmats.push_back( cv::Mat( m_inputsize, _type, &vect[0]) );
            cv::merge(vmats,_outmat);
            break;
    }
    cv::normalize(_outmat, _outmat, 0.0, 1.0, cv::NORM_MINMAX);
    return _outmat;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
tiny_cnn::vec_t CNNSegmentnet::__mat2vec_t(const cv::Mat &img, const cv::Size targetSize, int targetChannels) const
{
    // Resize if needed
    cv::Mat _mat;
    if(img.cols != targetSize.width || img.rows != targetSize.height)
        _mat = __cropresize(img, targetSize);
    else
        _mat = img;
    // Change channels quantity if needed
    if((targetChannels > 0) && (targetChannels != _mat.channels())) {
        switch(targetChannels) {
            case 1:
                cv::cvtColor(_mat, _mat, CV_BGR2GRAY);
                break;
            case 3:
                cv::cvtColor(_mat, _mat, CV_GRAY2BGR);
                break;
        }
    }
    // Convert to float_t type    
    int _maxval = 1;
    switch(_mat.depth()) {
        case CV_8U:
            _maxval = 255;
            break;
        case CV_16U:
            _maxval = 65535;
            break;
    }
    _mat.convertTo(_mat, (sizeof(tiny_cnn::float_t) == sizeof(double)) ? CV_64F : CV_32F, 2.0/_maxval, -1.0);

    // Visualize
    cv::namedWindow("CNNSegmentnet", CV_WINDOW_NORMAL);
    cv::imshow("CNNSegmentnet", _mat);
    cv::waitKey(300);

    // Construct vec_t image representation
    tiny_cnn::vec_t ovect;
    switch(_mat.channels()) {
        case 1: {
            tiny_cnn::float_t *ptr = _mat.ptr<tiny_cnn::float_t>(0);
            ovect = tiny_cnn::vec_t(ptr, ptr + _mat.cols * _mat.rows );
        } break;
        case 3: {
            std::vector<cv::Mat> _vmats;
            cv::split(_mat, _vmats);
            for(int i = 0; i < 3; i++) {
                cv::Mat _chanmat = _vmats[i];
                tiny_cnn::float_t *ptr = _chanmat.ptr<tiny_cnn::float_t>(0);
                ovect.insert(ovect.end(), ptr, ptr + _mat.cols * _mat.rows);
            }
        } break;
    }

    return ovect;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
cv::Mat CNNSegmentnet::__cropresize(const cv::Mat &input, const cv::Size size) const
{
    cv::Mat output;
    if(size.area() > 0){
        cv::Rect2f roiRect(0,0,0,0);
        if( (float)input.cols/input.rows > (float)size.width/size.height) {
            roiRect.height = (float)input.rows;
            roiRect.width = input.rows * (float)size.width/size.height;
            roiRect.x = (input.cols - roiRect.width)/2.0f;
        } else {
            roiRect.width = (float)input.cols;
            roiRect.height = input.cols * (float)size.height/size.width;
            roiRect.y = (input.rows - roiRect.height)/2.0f;
        }
        roiRect &= cv::Rect2f(0,0,(float)input.cols,(float)input.rows);
        if(roiRect.area() > 0)  {
            cv::Mat croppedImg(input, roiRect);
            int interpolationMethod = 0;
            if(size.area() > roiRect.area())
                interpolationMethod = CV_INTER_CUBIC;
            else
                interpolationMethod = CV_INTER_AREA;
            cv::resize(croppedImg, output, size, 0, 0, interpolationMethod);
        }
    } else {
        output = input;
    }
    return output;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void visualizeActivations(const tiny_cnn::network<tiny_cnn::sequential> &_net)
{
    for(size_t i = 0; i <_net.depth(); i++) {
        tiny_cnn::image<unsigned char> img = _net[i]->output_to_image(); // visualize activations of recent input
        cv::Mat mat = tinyimage2mat(img);
        if(mat.empty() == false) {
            cv::String windowname = (std::string("Activation of layer ") + std::to_string(i)).c_str();
            cv::namedWindow(windowname, CV_WINDOW_NORMAL);
            cv::imshow(windowname, mat);
        }
    }

}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
template<typename T>
cv::Mat tinyimage2mat(const tiny_cnn::image<T> &_image)
{
    std::vector<T> data = _image.data();
    cv::Mat mat = cv::Mat(_image.height(), _image.width(), CV_8UC1, &data[0]);
    return mat.clone();
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------




