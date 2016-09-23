#include "cnnconvsegmentnet.h"
#include <opencv2/highgui.hpp>

namespace segnet {

using namespace activation;
//-------------------------------------------------------------------------------------------------------
CNNConvSegmentnet::CNNConvSegmentnet()
{}
//-------------------------------------------------------------------------------------------------------
CNNConvSegmentnet::~CNNConvSegmentnet()
{}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
network<sequential> CNNConvSegmentnet::__initNet(const cv::Size &size, int inchannels, int outchannels)
{  
    network<sequential> _net;
    cnn_size_t _width = static_cast<cnn_size_t>(size.width),
               _height = static_cast<cnn_size_t>(size.height),
               _kernels = 16;

    _net << convolutional_layer<identity>(_width, _height, 3, static_cast<cnn_size_t>(inchannels), _kernels)
         << fully_connected_layer<softmax>((_width-2)*(_height-2)*_kernels, 2);

    return _net;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNConvSegmentnet::train(cv::InputArrayOfArrays _vraw, cv::InputArrayOfArrays _vlabel, int _epoch, int _minibatch)
{
    __train(_vraw, _vlabel, _epoch, _minibatch, false);
}
void CNNConvSegmentnet::update(cv::InputArrayOfArrays _vraw, cv::InputArrayOfArrays _vlabel, int _epoch, int _minibatch)
{
    __train(_vraw, _vlabel, _epoch, _minibatch, true);
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNConvSegmentnet::__train(cv::InputArrayOfArrays _vraw, cv::InputArrayOfArrays _vlabel, int _epoch, int _minibatch, bool preservedata)
{
    if(_vraw.kind() != cv::_InputArray::STD_VECTOR_MAT && _vlabel.kind() != cv::_InputArray::STD_VECTOR_MAT) {
        cv::String error_message = "CNNConvSegmentnet! - > The images are expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>).";
        CV_Error(cv::Error::StsBadArg, error_message);
    }
    if(_vraw.total() == 0 || _vlabel.total() == 0) {
        cv::String error_message = cv::format("CNNConvSegmentnet! - > Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(cv::Error::StsUnsupportedFormat, error_message);
    }

    std::vector<cv::Mat> _vmatrawimg;
    _vraw.getMatVector(_vmatrawimg);
    std::vector<cv::Mat> _vlabelimg;
    _vlabel.getMatVector(_vlabelimg);

    // Check if data is well-aligned
    if(_vmatrawimg.size() != _vlabelimg.size()) {
        cv::String error_message = cv::format("CNNConvSegmentnet! - > The number of samples (src) must be equal to the number of the labels. Was len(samples)=%d, len(labels)=%d.", _vmatrawimg.size(), _vlabelimg.size());
        CV_Error(cv::Error::StsBadArg, error_message);
    }

    std::vector<vec_t> srcvec_t;
    std::vector<label_t> srclabel_t;

    for(size_t it = 0; it < _vmatrawimg.size(); it++) {
        if(it == 0 && m_inputchannels == 0)
                setInputChannels( (_vmatrawimg[it]).channels() );

        __mosaic(m_inputsize, _vmatrawimg[it], _vlabelimg[it], srcvec_t, srclabel_t);
    }

    // Shuffle input pairs in random order, it should prevent overfitting when training samples is taken from different sources
    __random_shuffle(srcvec_t.begin(),srcvec_t.end(),srclabel_t.begin(),srclabel_t.end());

    if(preservedata == false)
        m_net = __initNet(m_inputsize, m_inputchannels, m_outputchannels);

    /*for(size_t i = 0; i < srcvec_t.size(); i++) {
        std::cout << std::endl << srclabel_t[i];
        for(size_t j = 0; j < srcvec_t[i].size(); j++) {
            std::cout << j << ":" << srcvec_t[i][j] << " ";
        }

    }*/

    // Batch_size is a number of samples enrolled per parameters update
    adam _opt;
    m_net.train<cross_entropy>(_opt, srcvec_t, srclabel_t, static_cast<cnn_size_t>(_minibatch), _epoch,
                                        [&](){ /*visualizeActivations(m_net);*/},
                                        [&](){

                                                static int _epoch = 0;
                                                std::cout << "\nEpoch " << ++_epoch << " has been passed";
                                              });
}
//-------------------------------------------------------------------------------------------------------
void CNNConvSegmentnet::__mosaic(const cv::Size &_msize, const cv::Mat &_rawimg, const cv::Mat &_labelimg, std::vector<vec_t> &_vparts, std::vector<label_t> &_vmarks)
{
    if(_rawimg.size() != _labelimg.size()) {
        cv::String error_message = cv::format("CNNConvSegmentnet! - > Raw and Label images should have equal sizes but does not have!");
        CV_Error(cv::Error::StsBadArg, error_message);
    }

    cv::Mat _rawmat, _labelmat;
    for(int y = 0 ; y < _rawimg.rows - _msize.height; y++) {
        for(int x = 0; x < _rawimg.cols - _msize.width; x++) {
            cv::Rect _roirect = cv::Rect(0,0,_msize.width,_msize.height)+cv::Point(x,y);
            _rawmat = _rawimg( _roirect );
            _labelmat = _labelimg( _roirect );
            _vparts.push_back( __mat2vec_t(_rawmat, m_inputchannels) );
            _vmarks.push_back( __getLabel(_labelmat) );
        }
    }

}
//-------------------------------------------------------------------------------------------------------
label_t CNNConvSegmentnet::__getLabel(const cv::Mat &img)
{
    if(img.at<unsigned char>(img.rows/2, img.cols/2) > 128)
        return label_t(1);
    else
        return label_t(0);
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNConvSegmentnet::setInputChannels(int _value)
{
    m_inputchannels = _value;
}
void CNNConvSegmentnet::setOutputChannels(int _value)
{
    m_outputchannels = _value;
}
int CNNConvSegmentnet::getInputChannels() const
{
    return m_inputchannels;
}
int CNNConvSegmentnet::getOutputChannels() const
{
    return m_outputchannels;
}
void CNNConvSegmentnet::setInputSize(const cv::Size &_size)
{
    m_inputsize = _size;
}
cv::Size CNNConvSegmentnet::getInputSize() const
{
    return m_inputsize;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void CNNConvSegmentnet::save(const char *filename) const
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
bool CNNConvSegmentnet::load(const char *filename)
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
cv::Mat CNNConvSegmentnet::predict(const cv::Mat &image) const
{
    m_net.set_netphase(tiny_cnn::net_phase::test);

    cv::Mat _output = cv::Mat::zeros(image.rows - m_inputsize.height, image.cols - m_inputsize.width, CV_8UC1);

    cv::Rect _roirect(0,0,m_inputsize.width, m_inputsize.height);
    for(int y = 0; y < image.rows - m_inputsize.height; y++) {
        for(int x = 0; x < image.cols - m_inputsize.width; x++) {
            cv::Mat _tempmat = image(_roirect + cv::Point(x,y));
            vec_t _vp = m_net.predict( __mat2vec_t(_tempmat, m_inputchannels) );
            _output.at<unsigned char>(y,x) = static_cast<unsigned char>(_vp[1]*255);
        }
    }
    return _output;
}
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
tiny_cnn::vec_t __mat2vec_t(const cv::Mat &img, int targetChannels, double min, double max)
{
    // Resize if needed
    cv::Mat _mat = img;

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
    /*cv::namedWindow("CNNConvSegmentnet", CV_WINDOW_NORMAL);
    cv::imshow("CNNConvSegmentnet", _mat);
    cv::waitKey(1);*/

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

} // end of the segnet namespace



