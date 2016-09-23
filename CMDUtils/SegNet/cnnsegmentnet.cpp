#include "cnnsegmentnet.h"
#include <opencv2/highgui.hpp>

namespace segnet {

using namespace activation;
//-------------------------------------------------------------------------------------------------------
CNNSegmentnet::CNNSegmentnet()
{}
//-------------------------------------------------------------------------------------------------------
CNNSegmentnet::~CNNSegmentnet()
{}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
network<sequential> CNNSegmentnet::__initNet(const cv::Size &size, int inchannels, int outchannels)
{  
    network<sequential> _net;
    return _net;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNSegmentnet::train(cv::InputArrayOfArrays _vvis, cv::InputArrayOfArrays _vseg, int _epoch, int _minibatch)
{
    __train(_vvis, _vseg, _epoch, _minibatch, false);
}
void CNNSegmentnet::update(cv::InputArrayOfArrays _vvis, cv::InputArrayOfArrays _vseg, int _epoch, int _minibatch)
{
    __train(_vvis, _vseg, _epoch, _minibatch, true);
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNSegmentnet::__train(cv::InputArrayOfArrays _vvis, cv::InputArrayOfArrays _vseg, int _epoch, int _minibatch, bool preservedata)
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

        srcvec_t.push_back( __mat2vec_t(srcmats[it], m_inputsize, m_irm, m_inputchannels) );
    }

    // Get the vector of segmented images
    srcmats.clear();
    _vseg.getMatVector(srcmats);
    std::vector<vec_t> segvec_t;
    for(size_t it = 0; it < srcmats.size(); it++) {
        if(it == 0 && m_outputchannels == 0)
            m_outputchannels = (srcmats[it]).channels();

        segvec_t.push_back( __mat2vec_t(srcmats[it], m_inputsize, m_irm, m_outputchannels, 0.0, 1.0) );
    }

    // Check if data is well-aligned
    if(segvec_t.size() != srcvec_t.size()) {
        cv::String error_message = cv::format("The number of samples (src) must be equal to the number of the labels. Was len(samples)=%d, len(labels)=%d.", srcvec_t.size(), segvec_t.size());
        CV_Error(cv::Error::StsBadArg, error_message);
    }   

    // Shuffle input pairs in random order, it should prevent overfitting when training samples is taken from different sources
    __random_shuffle(srcvec_t.begin(),srcvec_t.end(),segvec_t.begin(),segvec_t.end());

    if(preservedata == false)
        m_net = __initNet(m_inputsize, m_inputchannels, m_outputchannels);
    
    adam _opt;
	    
    // Note that for right learning by the fit() function, the values in label data (segvec_t here) should be normalized to [0.0; 1.0] interval
	m_net.fit<cross_entropy>(_opt, srcvec_t, segvec_t, _minibatch, _epoch,
									[&](){/*visualizeLastLayerActivation(m_net);*/},
									[&](){visualizeLastLayerActivation(m_net);});

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
    // Save input image size
    cv::Size _size(image.cols, image.rows);

    m_net.set_netphase(tiny_cnn::net_phase::test);
    vec_t vect = m_net.predict( __mat2vec_t(image, m_inputsize, m_irm, m_inputchannels) );
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

    // Restore input size with respect to ImageResizeMethod
    _outmat = __restoreSize(_outmat, _size, m_irm);
    return _outmat;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
tiny_cnn::vec_t __mat2vec_t(const cv::Mat &img, const cv::Size targetSize, ImageResizeMethod resizeMethod, int targetChannels, double min, double max)
{
    // Resize if needed
    cv::Mat _mat;
    if(img.cols != targetSize.width || img.rows != targetSize.height) {
        switch(resizeMethod){
            case ImageResizeMethod::CropAndResizeFromCenter:
                _mat =__cropresize(img, targetSize);
                break;
            case ImageResizeMethod::PaddZeroAndResize:
                _mat = __propresize(img, targetSize);
                break;
        }
    } else {
        _mat = img;
    }
    // Change channels quantity if needed
    if((targetChannels > 0) && (targetChannels != _mat.channels()))
        switch(targetChannels) {
            case 1:
                if(_mat.channels() == 3)
                    cv::cvtColor(_mat, _mat, CV_BGR2GRAY);
                else
                    cv::cvtColor(_mat, _mat, CV_BGRA2GRAY);
                break;
            case 3:
                cv::cvtColor(_mat, _mat, CV_GRAY2BGR);
                break;
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
    _mat.convertTo(_mat, (sizeof(tiny_cnn::float_t) == sizeof(double)) ? CV_64F : CV_32F, (max-min)/_maxval, min);

    // Visualize
    cv::namedWindow("CNNSegmentnet", CV_WINDOW_NORMAL);
    cv::imshow("CNNSegmentnet", _mat);
    cv::waitKey(1);

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
cv::Mat __cropresize(const cv::Mat &input, const cv::Size size)
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
                interpolationMethod = CV_INTER_LANCZOS4;
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
cv::Mat __propresize(const cv::Mat &input, const cv::Size size)
{
    cv::Mat output;
    if(size.area() > 0){
        cv::Rect2f roiRect(0,0,0,0);
        cv::Point shift(0,0);
        if( (float)input.cols/input.rows > (float)size.width/size.height) {
            roiRect.width = (float)input.cols;
            roiRect.height = input.cols * (float)size.height/size.width;
            shift.y = static_cast<int>((roiRect.height - (float)input.rows)/2.0f);
        } else {
            roiRect.height = (float)input.rows;
            roiRect.width = input.rows * (float)size.width/size.height;
            shift.x = static_cast<int>((roiRect.width - (float)input.cols)/2.0f);
        }
        output = cv::Mat::zeros(roiRect.size(),input.type());
        cv::Mat imgcontent = output(cv::Rect(0,0,input.cols,input.rows)+shift);
        input.copyTo(imgcontent);
        if(roiRect.area() > 0)  {
            int interpolationMethod = 0;
            if(size.area() > roiRect.area())
                interpolationMethod = CV_INTER_LANCZOS4;
            else
                interpolationMethod = CV_INTER_AREA;
            cv::resize(output, output, size, 0, 0, interpolationMethod);
        }
    } else {
        output = input;
    }
    return output;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
cv::Mat __restoreSize (const cv::Mat &img, const cv::Size &dstSize, ImageResizeMethod resizeMethod)
{
    if(resizeMethod == ImageResizeMethod::PaddZeroAndResize)
        return __cropresize(img, dstSize);
    else
        return __propresize(img, dstSize);
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNSegmentnet::setImageResizeMethod(ImageResizeMethod method)
{
    m_irm = method;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
ImageResizeMethod CNNSegmentnet::getImageResizeMethod() const
{
    return m_irm;
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
    cv::waitKey(1);

}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void visualizeLastLayerActivation(const tiny_cnn::network<tiny_cnn::sequential> &_net)
{
    tiny_cnn::image<unsigned char> img = _net[_net.depth()-1]->output_to_image(); // visualize activations of recent input
    cv::Mat mat = tinyimage2mat(img);
    if(mat.empty() == false) {
        cv::namedWindow("Last layer activation", CV_WINDOW_NORMAL);
        cv::imshow("Last layer activation", mat);
    }
    cv::waitKey(1);
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
template<typename T>
cv::Mat tinyimage2mat(const tiny_cnn::image<T> &_image)
{
    std::vector<T> data = _image.data();
    cv::Mat mat = cv::Mat(static_cast<int>(_image.height()), static_cast<int>(_image.width()), CV_8UC1, &data[0]);
    return mat.clone();
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
template <typename Iterator1, typename Iterator2>
void __random_shuffle (Iterator1 v1first, Iterator1 v1last, Iterator2 v2first, Iterator2 v2last)
{
    std::iterator_traits<Iterator1>::difference_type i, v1length = v1last - v1first;
    std::iterator_traits<Iterator2>::difference_type v2length = v2last - v2first;
    if(v1length != v2length) {
        CV_Error(cv::Error::StsBadArg, "Error in __random_shuffle(): input vectors have different sizes");
        return;
    } else {
        int pos;
        for(i = 0; i < v1length; i++) {
            pos = std::rand() % v1length;
            std::swap(v1first[i],v1first[pos]);
            std::swap(v2first[i],v2first[pos]);
        }
    }
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------
SegNetForLungs::SegNetForLungs()
{   }
//------------------------------------------------------------------------------------------
network<sequential> SegNetForLungs::__initNet(const cv::Size &size, int inchannels, int outchannels)
{
    cnn_size_t _kernels = 8,
               width = static_cast<cnn_size_t>(size.width),
               height = static_cast<cnn_size_t>(size.height);

    network<sequential> _net;
    _net << convolutional_layer<relu>(width, height, 3, static_cast<cnn_size_t>(inchannels), _kernels, padding::same)
         << convolutional_layer<relu>(width, height, 3, _kernels, _kernels, padding::same)
         << average_pooling_layer<identity>(width, height, _kernels, 2)
            << convolutional_layer<relu>(width/2, height/2, 3, _kernels, 2*_kernels, padding::same)
            << convolutional_layer<relu>(width/2, height/2, 3, 2*_kernels, 2*_kernels, padding::same)
            << average_pooling_layer<identity>(width/2, height/2, 2*_kernels, 2)
               << convolutional_layer<relu>(width/4, height/4, 3, 2*_kernels, 4*_kernels, padding::same)
               << convolutional_layer<relu>(width/4, height/4, 3, 4*_kernels, 4*_kernels, padding::same)
               << average_pooling_layer<identity>(width/4, height/4, 4*_kernels, 2)
                  << convolutional_layer<relu>(width/8, height/8, 3, 4*_kernels, 8*_kernels, padding::same)
                  << convolutional_layer<relu>(width/8, height/8, 3, 8*_kernels, 4*_kernels, padding::same)
               << average_unpooling_layer<identity>(width/8, height/8, 4*_kernels, 2)
               << convolutional_layer<relu>(width/4, height/4, 3, 4*_kernels, 4*_kernels, padding::same)
               << convolutional_layer<relu>(width/4, height/4, 3, 4*_kernels, 4*_kernels, padding::same)
            << average_unpooling_layer<identity>(width/4, height/4, 4*_kernels, 2)
            << convolutional_layer<relu>(width/2, height/2, 3, 4*_kernels, 4*_kernels, padding::same)
            << convolutional_layer<relu>(width/2, height/2, 3, 4*_kernels, 2*_kernels, padding::same)
         << average_unpooling_layer<identity>(width/2, height/2, 2*_kernels, 2)
         << convolutional_layer<relu>(width, height, 3, 2*_kernels, 2*_kernels, padding::same)
         << convolutional_layer<relu>(width, height, 3, 2*_kernels, _kernels, padding::same)
         << convolutional_layer<softmax>(width, height, 3, _kernels, static_cast<cnn_size_t>(outchannels), padding::same);
    return _net;
}
//-----------------------------------------------------------------------------------------


} // end of the segnet namespace



